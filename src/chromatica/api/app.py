"""FastAPI application for Chromatica."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import RGB, Compose, PILToTensor, Resize, ToDtype

from chromatica.datasets.transform import LAB2RGB, RGB2LAB
from chromatica.nn.loader import load_cnn_class

if TYPE_CHECKING:
    from chromatica.nn.base import BaseCNN


class AvailableDatasets(StrEnum):
    """Supported datasets."""

    COCO = "COCO"
    FOOD101 = "FOOD101"


class AvailableModels(StrEnum):
    """Supported model architectures."""

    U_NET_WITH_SKIP_CONNECTIONS = "U_NET_WITH_SKIP_CONNECTIONS"
    U_NET_WITHOUT_SKIP_CONNECTIONS = "U_NET_WITHOUT_SKIP_CONNECTIONS"


class TrainingRequest(BaseModel):
    """Request payload for training a model."""

    dataset: AvailableDatasets
    model: AvailableModels


class TrainingMetrics(BaseModel):
    """Training metrics snapshot."""

    train_error: float
    test_error: float
    history: list[float] = Field(default_factory=list)


class TrainingRecord(BaseModel):
    """Registered model entry."""

    id: str
    dataset: AvailableDatasets
    model: AvailableModels
    status: str
    model_path: str | None = None
    metrics: TrainingMetrics | None = None
    error: str | None = None
    created_at: float = Field(default_factory=lambda: time.time())


class ModelRegistry:
    """Thread-safe registry for trained models."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(".data") / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / "registry.json"
        self._lock = threading.Lock()
        self._models: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {}

    def _flush(self) -> None:
        self._registry_path.write_text(json.dumps(self._models, indent=2))

    def list_models(self) -> list[dict[str, Any]]:
        """Return snapshot of all registered models."""
        with self._lock:
            return list(self._models.values())

    def get(self, model_id: str) -> dict[str, Any] | None:
        """Return model record by id or None if missing."""
        with self._lock:
            return self._models.get(model_id)

    def register_pending(
        self,
        model_id: str,
        dataset: AvailableDatasets,
        model: AvailableModels,
    ) -> TrainingRecord:
        """Create a running record for the given model_id."""
        record = TrainingRecord(
            id=model_id, dataset=dataset, model=model, status="running"
        )
        with self._lock:
            self._models[model_id] = record.model_dump()
            self._flush()
        return record

    def mark_completed(
        self,
        model_id: str,
        *,
        model_path: Path,
        metrics: TrainingMetrics,
    ) -> TrainingRecord:
        """Mark a model as completed and persist metrics."""
        with self._lock:
            payload = self._models.get(model_id)
            if payload is None:
                payload = TrainingRecord(
                    id=model_id,
                    dataset=AvailableDatasets.COCO,
                    model=AvailableModels.U_NET_WITHOUT_SKIP_CONNECTIONS,
                    status="running",
                ).model_dump()
            payload["status"] = "completed"
            payload["model_path"] = str(model_path)
            payload["metrics"] = metrics.model_dump()
            self._models[model_id] = payload
            self._flush()
            return TrainingRecord(**payload)

    def mark_failed(self, model_id: str, error: str) -> TrainingRecord:
        """Mark a model as failed and store the error."""
        with self._lock:
            payload = self._models.get(model_id)
            if payload is None:
                payload = TrainingRecord(
                    id=model_id,
                    dataset=AvailableDatasets.COCO,
                    model=AvailableModels.U_NET_WITHOUT_SKIP_CONNECTIONS,
                    status="failed",
                    error=error,
                ).model_dump()
            else:
                payload["status"] = "failed"
                payload["error"] = error
            self._models[model_id] = payload
            self._flush()
            return TrainingRecord(**payload)


class SyntheticColorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Tiny synthetic dataset for quick training runs."""

    def __init__(self, length: int = 12, seed: int = 0) -> None:
        super().__init__()
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        """Return dataset length."""
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Generate a random LAB tensor triple for index."""
        l_channel = torch.rand((1, 64, 64), generator=self.generator)
        ab_channels = torch.rand((2, 64, 64), generator=self.generator) * 2 - 1
        return l_channel, ab_channels, idx


def _create_model(model: AvailableModels) -> BaseCNN:
    version = "v2" if model is AvailableModels.U_NET_WITH_SKIP_CONNECTIONS else "v1"
    cls = load_cnn_class(version)
    return cls()


def _train_once(model: nn.Module, loader: DataLoader) -> float:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    loss_sum = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        l_batch, ab_batch, _ = batch
        preds = model(l_batch)
        loss = loss_fn(preds, ab_batch)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(len(loader), 1)


def _evaluate(model: nn.Module, loader: DataLoader) -> float:
    loss_fn = nn.MSELoss()
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for l_batch, ab_batch, _ in loader:
            preds = model(l_batch)
            total += loss_fn(preds, ab_batch).item()
    return total / max(len(loader), 1)


def _run_training(
    dataset: AvailableDatasets, model: AvailableModels
) -> tuple[BaseCNN, TrainingMetrics]:
    del dataset  # currently unused but kept for API symmetry
    torch.manual_seed(42)
    net = _create_model(model)
    train_loader = DataLoader(SyntheticColorDataset(), batch_size=3, shuffle=True)
    history = [_train_once(net, train_loader) for _ in range(2)]
    test_error = _evaluate(net, train_loader)
    metrics = TrainingMetrics(
        train_error=history[-1],
        test_error=test_error,
        history=history,
    )
    return net, metrics


def _encode_image(rgb_tensor: torch.Tensor) -> str:
    rgb_clamped = torch.clamp(rgb_tensor, 0.0, 1.0)
    array = (rgb_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    with BytesIO() as buffer:
        from PIL import Image  # local import to reduce startup cost

        Image.fromarray(array).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded


def _prepare_image(
    tensor: torch.Tensor, model: BaseCNN
) -> tuple[torch.Tensor, dict[str, float]]:
    l_channel = tensor[:1, :, :].unsqueeze(0)
    with torch.inference_mode():
        ab_pred = model(l_channel)
    full_lab = torch.cat([l_channel.squeeze(0), ab_pred.squeeze(0)], dim=0)
    rgb_tensor = LAB2RGB()(full_lab).clamp(0.0, 1.0)
    metrics = {
        "mean_abs_ab": float(ab_pred.abs().mean().item()),
        "max_ab": float(ab_pred.abs().max().item()),
    }
    return rgb_tensor, metrics


def _to_lab_tensor(image: UploadFile) -> torch.Tensor:
    content = image.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image payload")

    from PIL import Image

    pil_image = Image.open(BytesIO(content)).convert("RGB")
    transform = Compose(
        [
            PILToTensor(),
            Resize((256, 256)),
            RGB(),
            ToDtype(torch.float32, scale=True),
            RGB2LAB(),
        ]
    )
    return transform(pil_image)


def train_and_register(
    request: TrainingRequest,
    model_id: str,
    registry: ModelRegistry,
) -> None:
    """Run training in a background thread and persist record."""
    registry.register_pending(model_id, request.dataset, request.model)
    try:
        model, metrics = _run_training(request.dataset, request.model)
        model_path = registry.root / f"{model_id}.pt"
        torch.save(model.state_dict(), model_path)
        registry.mark_completed(model_id, model_path=model_path, metrics=metrics)
    except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
        registry.mark_failed(model_id, error=str(exc))


def build_app(registry: ModelRegistry | None = None) -> FastAPI:
    """Build and return configured FastAPI instance."""
    registry = registry or ModelRegistry()
    app = FastAPI(title="Chromatica API")

    @app.get("/datasets")
    def list_datasets() -> dict[str, list[str]]:
        return {"datasets": [d.value for d in AvailableDatasets]}

    @app.get("/models")
    def list_models() -> dict[str, list[str]]:
        return {"models": [m.value for m in AvailableModels]}

    @app.post("/train")
    def train_endpoint(request: TrainingRequest) -> dict[str, str]:
        model_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=train_and_register,
            args=(request, model_id, registry),
            daemon=True,
        )
        thread.start()
        return {"model_id": model_id, "status": "scheduled"}

    @app.get("/trained-models")
    def trained_models() -> dict[str, list[dict[str, Any]]]:
        return {"models": registry.list_models()}

    @app.post("/predict")
    async def predict(
        model_id: str, file: Annotated[UploadFile, File(...)]
    ) -> dict[str, Any]:
        record = registry.get(model_id)
        if record is None or record.get("status") != "completed":
            raise HTTPException(status_code=404, detail="Model not found or not ready")

        model_variant = AvailableModels(record["model"])
        model = _create_model(model_variant)

        model_path = record.get("model_path")
        if model_path is None:
            raise HTTPException(status_code=400, detail="Model path missing")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        lab_tensor = _to_lab_tensor(file)
        rgb_tensor, metrics = _prepare_image(lab_tensor, model)
        encoded_image = _encode_image(rgb_tensor)
        return {
            "model_id": model_id,
            "metrics": metrics,
            "image_base64": encoded_image,
        }

    return app


app = build_app()
