"""FastAPI application for Chromatica backed by a relational database."""

from __future__ import annotations

import base64
import logging
import math
import threading
import uuid
from collections.abc import Iterator  # noqa: TC003
from contextlib import nullcontext
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session  # noqa: TC002
from torch import amp, nn
from torch.utils.data import DataLoader, random_split
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
from chromatica.datasets.dataset import DirectoryImageDataset
from chromatica.datasets.transform import LAB2RGB, RGB2LAB
from chromatica.metrics.color_diversity import compute_delta_a_stats
from chromatica.nn.loader import load_cnn_class

if TYPE_CHECKING:
    from chromatica.nn.base import BaseCNN


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    uvicorn_logger = logging.getLogger("uvicorn.error")
    if uvicorn_logger.handlers:
        logger.handlers = uvicorn_logger.handlers
        logger.setLevel(uvicorn_logger.level)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _setup_logger()


def _log_info(message: str, *args: Any) -> None:
    """Log info-level messages and flush handlers to stream immediately."""
    logger.info(message, *args)
    for handler in logger.handlers:
        handler.flush()


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


class TrainingConfig(BaseModel):
    """Hyperparameters for a training run."""

    epochs: int = Field(2, ge=1, le=100)
    batch_size: int = Field(4, ge=1, le=64)
    learning_rate: float = Field(1e-3, gt=0.0)
    val_split: float = Field(0.2, ge=0.0, lt=1.0)
    num_workers: int = Field(4, ge=0, le=8)
    seed: int = 42
    use_amp: bool | None = None
    max_train_batches: int | None = Field(
        None, ge=1, description="Optional cap for train batches per epoch."
    )
    max_val_batches: int | None = Field(
        None, ge=1, description="Optional cap for val batches per epoch."
    )


class EpochMetrics(BaseModel):
    """Loss snapshot for an epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None = None


class TrainingHistory(BaseModel):
    """History and config attached to a training run."""

    config: TrainingConfig | None = None
    epochs: list[EpochMetrics] = Field(default_factory=list)


class TrainingRequest(BaseModel):
    """Request payload for training a model."""

    dataset: str
    model: str
    config: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="before")
    @classmethod
    def _merge_flat_config(cls, data: Any) -> Any:
        """Fold top-level hyperparameters into the nested config block.

        The frontend posts training params alongside dataset/model instead of
        under a `config` key. Allow both shapes by merging known hyperparameter
        fields into the config payload before validation.
        """
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        config_data = payload.get("config") or {}
        if not isinstance(config_data, dict):
            config_data = dict(config_data)

        hyperparams = {
            key: payload.pop(key)
            for key in list(payload)
            if key in TrainingConfig.model_fields
        }
        if hyperparams:
            payload["config"] = {**config_data, **hyperparams}
        elif "config" in payload:
            payload["config"] = config_data
        return payload


class TrainingMetrics(BaseModel):
    """Training metrics snapshot."""

    train_error: float
    val_error: float | None
    history: TrainingHistory = Field(default_factory=TrainingHistory)
    delta_a: dict[str, float] | None = None


def get_db_session() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _create_model(model: AvailableModels) -> BaseCNN:
    version = "v2" if model == AvailableModels.U_NET_WITH_SKIP_CONNECTIONS else "v1"
    cls = load_cnn_class(version)
    return cls()


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _build_dataloaders(
    dataset_path: Path, config: TrainingConfig, device: torch.device
) -> tuple[DataLoader, DataLoader | None]:
    dataset = DirectoryImageDataset(dataset_path)
    if len(dataset) == 0:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    val_size = 0
    if config.val_split > 0.0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * config.val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise HTTPException(status_code=400, detail="Not enough samples to train")

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=config.num_workers,
        )
    return train_loader, val_loader


def _train_model(  # noqa: C901, PLR0912, PLR0915
    model: AvailableModels,
    dataset_path: Path,
    config: TrainingConfig,
    run_id: str | None = None,
) -> tuple[BaseCNN, TrainingMetrics]:
    run_label = f"[train:{run_id}]" if run_id else "[train]"
    torch.manual_seed(config.seed)
    device = _select_device()
    net = _create_model(model).to(device)

    train_loader, val_loader = _build_dataloaders(dataset_path, config, device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    use_amp = (
        config.use_amp if config.use_amp is not None else device.type == "cuda"
    ) and device.type == "cuda"
    scaler = amp.GradScaler("cuda") if use_amp else None

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset) if val_loader else 0
    total_batches = len(train_loader)
    if config.max_train_batches:
        total_batches = min(total_batches, config.max_train_batches)
    total_steps = total_batches * config.epochs
    progress_marks = [0.25, 0.5, 0.75, 1.0]
    progress_steps = [max(1, math.ceil(total_steps * mark)) for mark in progress_marks]
    next_mark_idx = 0
    _log_info(
        "%s Starting training: model=%s dataset=%s device=%s epochs=%s "
        "batch_size=%s train_samples=%s val_samples=%s max_train_batches=%s "
        "max_val_batches=%s amp=%s",
        run_label,
        model,
        dataset_path,
        device,
        config.epochs,
        config.batch_size,
        train_size,
        val_size,
        config.max_train_batches or "all",
        config.max_val_batches or "all",
        use_amp,
    )
    history: list[EpochMetrics] = []
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        net.train()
        train_loss = 0.0
        train_batches = 0
        val_batches = 0
        for batch_idx, (l_batch, ab_batch, _) in enumerate(train_loader, start=1):
            if config.max_train_batches and batch_idx > config.max_train_batches:
                break
            l_tensor = l_batch.to(device, non_blocking=True)
            ab_tensor = ab_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                amp.autocast(device_type="cuda", enabled=True)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                preds = net(l_tensor)
                loss = loss_fn(preds, ab_tensor)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            global_step += 1
            while (
                next_mark_idx < len(progress_steps)
                and global_step >= (progress_steps[next_mark_idx])
            ):
                progress_percent = int(progress_marks[next_mark_idx] * 100)
                _log_info(
                    "%s Progress %s%% (epoch %s/%s, batch %s/%s): train_loss=%.4f",
                    run_label,
                    progress_percent,
                    epoch,
                    config.epochs,
                    batch_idx,
                    total_batches,
                    train_loss / max(train_batches, 1),
                )
                next_mark_idx += 1

        train_loss = train_loss / max(train_batches, 1)

        val_loss = None
        if val_loader is not None:
            net.eval()
            running_val = 0.0
            with torch.inference_mode():
                for batch_idx, (l_batch, ab_batch, _) in enumerate(val_loader, start=1):
                    if config.max_val_batches and batch_idx > config.max_val_batches:
                        break
                    l_tensor = l_batch.to(device, non_blocking=True)
                    ab_tensor = ab_batch.to(device, non_blocking=True)
                    preds = net(l_tensor)
                    running_val += loss_fn(preds, ab_tensor).item()
                    val_batches += 1
            if val_batches:
                val_loss = running_val / val_batches

        history.append(
            EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        )
        progress_fraction = epoch / config.epochs
        while next_mark_idx < len(progress_marks) and progress_fraction >= (
            progress_marks[next_mark_idx] - 1e-9
        ):
            progress_percent = int(progress_marks[next_mark_idx] * 100)
            _log_info(
                "%s Progress %s%% (epoch %s/%s): train_loss=%.4f "
                "%s batches(train=%s,val=%s)",
                run_label,
                progress_percent,
                epoch,
                config.epochs,
                train_loss,
                (f"val_loss={val_loss:.4f}" if val_loss is not None else "val_loss=NA"),
                train_batches,
                val_batches if val_loader is not None else 0,
            )
            next_mark_idx += 1

    delta_metrics = None
    if val_loader is not None:
        delta_metrics = compute_delta_a_stats(
            val_loader, net, device=device, use_amp=use_amp
        )

    metrics = TrainingMetrics(
        train_error=history[-1].train_loss,
        val_error=history[-1].val_loss,
        history=TrainingHistory(config=config, epochs=history),
        delta_a=delta_metrics,
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
        history_payload = record.metrics.history or {}
        epochs_raw = history_payload.get("epochs", [])
        config_raw = history_payload.get("config")
        delta_a = history_payload.get("delta_a")
        history = TrainingHistory(
            config=TrainingConfig.model_validate(config_raw) if config_raw else None,
            epochs=[EpochMetrics.model_validate(entry) for entry in epochs_raw or []],
        )
        metrics = TrainingMetrics(
            train_error=record.metrics.train_error,
            val_error=record.metrics.test_error,
            history=history,
            delta_a=delta_a,
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
    dataset_path: Path,
    config: TrainingConfig,
) -> None:
    """Run training in a background thread and persist record changes."""
    try:
        _log_info(
            "Training run %s started: model=%s dataset=%s",
            model_id,
            architecture,
            dataset_path,
        )
        model, metrics = _train_model(
            architecture, dataset_path, config, run_id=model_id
        )
        _log_info(
            "Training run %s finished; persisting weights to %s", model_id, model_store
        )
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
                test_error=metrics.val_error,
                history={
                    "config": metrics.history.config.model_dump()
                    if metrics.history.config
                    else None,
                    "epochs": [entry.model_dump() for entry in metrics.history.epochs],
                    "delta_a": metrics.delta_a,
                },
            )
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.exception("Training run %s failed", model_id)
        with session_scope() as session:
            record = session.get(TrainedModel, model_id)
            if record is None:
                return
            record.status = Status.FAILED
            record.error = str(exc)


def build_app(settings: Settings | None = None) -> FastAPI:  # noqa: C901, PLR0915
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
        # Sync new folders into the DB, then expose only datasets that have a directory.
        sync_datasets_from_disk(dataset_root)
        datasets = session.scalars(select(Dataset.code)).all()
        available = [code for code in datasets if (dataset_root / code).is_dir()]
        return {"datasets": available}

    @app.get("/models")
    def list_models(
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, list[str]]:
        architectures = session.scalars(select(Architecture.code)).all()
        return {"models": architectures}

    @app.get("/train/config")
    def default_training_config() -> dict[str, Any]:
        """Expose default training hyperparameters."""
        defaults = TrainingConfig()
        return {"config": defaults.model_dump()}

    @app.post("/train")
    def train_endpoint(
        request: TrainingRequest,
        session: Annotated[Session, Depends(get_db_session)],
    ) -> dict[str, str]:
        dataset = _get_dataset(session, request.dataset)
        architecture, model_enum = _resolve_architecture(session, request.model)
        model_id = str(uuid.uuid4())

        dataset_path = dataset_root / dataset.code
        if not dataset_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset directory '{dataset_path}' not found",
            )
        try:
            DirectoryImageDataset(dataset_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _log_info(
            "Scheduling training run %s: dataset=%s model=%s epochs=%s batch_size=%s",
            model_id,
            dataset.code,
            architecture.code,
            request.config.epochs,
            request.config.batch_size,
        )
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
            args=(model_id, model_enum, model_store, dataset_path, request.config),
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
