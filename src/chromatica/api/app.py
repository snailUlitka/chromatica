"""FastAPI application for Chromatica backed by a relational database."""

from __future__ import annotations

import base64
import threading
import uuid
from collections.abc import Iterator  # noqa: TC003
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session  # noqa: TC002
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import RGB, Compose, PILToTensor, Resize, ToDtype

from chromatica.api.bootstrap import ensure_seed_records, sync_datasets_from_disk
from chromatica.api.config import Settings, get_settings
from chromatica.api.db import SessionLocal, session_scope
from chromatica.api.models import (
    Architecture,
    Dataset,
    Status,
    TrainedModel,
)
from chromatica.api.models import (
    TrainingMetrics as TrainingMetricsORM,
)
from chromatica.datasets.transform import LAB2RGB, RGB2LAB
from chromatica.nn.loader import load_cnn_class

if TYPE_CHECKING:
    from chromatica.nn.base import BaseCNN


class AvailableModels(str):
    """Supported model architectures."""

    __slots__ = ()

    U_NET_WITH_SKIP_CONNECTIONS = "U_NET_WITH_SKIP_CONNECTIONS"
    U_NET_WITHOUT_SKIP_CONNECTIONS = "U_NET_WITHOUT_SKIP_CONNECTIONS"


ARCHITECTURE_TO_MODEL = {
    "u_net_v1": AvailableModels.U_NET_WITHOUT_SKIP_CONNECTIONS,
    "u_net_v2": AvailableModels.U_NET_WITH_SKIP_CONNECTIONS,
}

LEGACY_MODEL_ALIASES = {
    AvailableModels.U_NET_WITHOUT_SKIP_CONNECTIONS: "u_net_v1",
    AvailableModels.U_NET_WITH_SKIP_CONNECTIONS: "u_net_v2",
}


class TrainingRequest(BaseModel):
    """Request payload for training a model."""

    dataset: str
    model: str


class TrainingMetrics(BaseModel):
    """Training metrics snapshot."""

    train_error: float
    test_error: float
    history: list[float] = Field(default_factory=list)


class SyntheticColorDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor, int]]):
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


def get_db_session() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        SessionLocal.remove()


def _create_model(model: AvailableModels) -> BaseCNN:
    version = "v2" if model == AvailableModels.U_NET_WITH_SKIP_CONNECTIONS else "v1"
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


def _run_training(model: AvailableModels) -> tuple[BaseCNN, TrainingMetrics]:
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


def _serialize_model(record: TrainedModel) -> dict[str, Any]:
    metrics = None
    if record.metrics:
        metrics = TrainingMetrics(
            train_error=record.metrics.train_error,
            test_error=record.metrics.test_error,
            history=record.metrics.history or [],
        ).model_dump()

    created_ts = record.created_at.timestamp() if record.created_at else None
    completed_ts = record.completed_at.timestamp() if record.completed_at else None
    return {
        "id": record.id,
        "dataset": record.dataset.code,
        "model": record.architecture.code,
        "status": record.status.value,
        "model_path": record.model_path,
        "metrics": metrics,
        "error": record.error,
        "created_at": created_ts,
        "completed_at": completed_ts,
    }


def _resolve_architecture(
    session: Session, raw_model: str
) -> tuple[Architecture, AvailableModels]:
    normalized = raw_model.strip()
    fallback_code = LEGACY_MODEL_ALIASES.get(normalized)
    candidate_codes = [normalized]
    if fallback_code:
        candidate_codes.append(fallback_code)

    for candidate in candidate_codes:
        architecture = session.scalar(
            select(Architecture).where(Architecture.code == candidate)
        )
        if architecture is None:
            continue
        mapped_model = ARCHITECTURE_TO_MODEL.get(architecture.code)
        if mapped_model is None:
            break
        return architecture, mapped_model

    raise HTTPException(status_code=404, detail="Model architecture not found")


def _get_dataset(session: Session, code: str) -> Dataset:
    dataset = session.scalar(select(Dataset).where(Dataset.code == code))
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{code}' not found")
    return dataset


def _train_and_persist(
    model_id: str,
    architecture: AvailableModels,
    model_store: Path,
) -> None:
    """Run training in a background thread and persist record changes."""
    try:
        model, metrics = _run_training(architecture)
        model_store.mkdir(parents=True, exist_ok=True)
        model_path = model_store / f"{model_id}.pt"
        torch.save(model.state_dict(), model_path)

        with session_scope() as session:
            record = session.get(TrainedModel, model_id)
            if record is None:
                return
            record.status = Status.COMPLETED
            record.model_path = str(model_path)
            record.completed_at = datetime.now(tz=UTC)
            record.metrics = TrainingMetricsORM(
                model_id=model_id,
                train_error=metrics.train_error,
                test_error=metrics.test_error,
                history=metrics.history,
            )
    except Exception as exc:  # pragma: no cover - defensive branch  # noqa: BLE001
        with session_scope() as session:
            record = session.get(TrainedModel, model_id)
            if record is None:
                return
            record.status = Status.FAILED
            record.error = str(exc)


def build_app(settings: Settings | None = None) -> FastAPI:  # noqa: C901
    """Build and return configured FastAPI instance."""
    resolved_settings = settings or get_settings()
    model_store = Path(resolved_settings.model_store_path)
    dataset_root = Path(resolved_settings.datasets_path)

    cors_origins = resolved_settings.cors_origins
    allow_credentials = "*" not in cors_origins
    app = FastAPI(title="Chromatica API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def bootstrap() -> None:
        ensure_seed_records()
        sync_datasets_from_disk(dataset_root)

    @app.get("/datasets")
    def list_datasets(
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, list[str]]:
        datasets = session.scalars(select(Dataset.code)).all()
        return {"datasets": datasets}

    @app.get("/models")
    def list_models(
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, list[str]]:
        architectures = session.scalars(select(Architecture.code)).all()
        return {"models": architectures}

    @app.post("/train")
    def train_endpoint(
        request: TrainingRequest,
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, str]:
        dataset = _get_dataset(session, request.dataset)
        architecture, model_enum = _resolve_architecture(session, request.model)
        model_id = str(uuid.uuid4())

        record = TrainedModel(
            id=model_id,
            dataset_id=dataset.id,
            architecture_id=architecture.id,
            status=Status.RUNNING,
        )
        session.add(record)
        session.commit()

        thread = threading.Thread(
            target=_train_and_persist,
            args=(model_id, model_enum, model_store),
            daemon=True,
        )
        thread.start()
        return {"model_id": model_id, "status": "scheduled"}

    @app.get("/trained-models")
    def trained_models(
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, list[dict[str, Any]]]:
        records = session.scalars(select(TrainedModel)).all()
        return {"models": [_serialize_model(record) for record in records]}

    @app.post("/predict")
    async def predict(
        model_id: str,
        file: Annotated[UploadFile, File(...)],
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, Any]:
        record = session.get(TrainedModel, model_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Model not found")
        if record.status != Status.COMPLETED:
            raise HTTPException(status_code=409, detail="Model is not ready")

        model_enum = ARCHITECTURE_TO_MODEL.get(record.architecture.code)
        if model_enum is None:
            raise HTTPException(status_code=400, detail="Unknown architecture")

        model = _create_model(model_enum)
        model_path = record.model_path
        if model_path is None:
            raise HTTPException(status_code=400, detail="Model path missing")

        path_obj = Path(model_path)
        if not path_obj.exists():
            raise HTTPException(status_code=404, detail="Stored model weights missing")

        state_dict = torch.load(path_obj, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

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
