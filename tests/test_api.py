# ruff: noqa: S101,PLR2004
"""FastAPI surface smoke tests."""

from __future__ import annotations

import time
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from PIL import Image

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def _sample_image_bytes(color: tuple[int, int, int] = (120, 130, 140)) -> bytes:
    image = Image.new("RGB", (32, 32), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _wait_for_model(client: TestClient, model_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    last_seen: dict | None = None
    while time.time() < deadline:
        response = client.get("/trained-models")
        response.raise_for_status()
        models = response.json()["models"]
        for record in models:
            if record["id"] == model_id:
                if record["status"] == "completed":
                    return record
                last_seen = record
        time.sleep(0.1)
    msg = f"Model {model_id} not ready. Last state: {last_seen}"
    raise AssertionError(msg)


def test_list_endpoints(api_client: TestClient) -> None:
    """Datasets and model discovery endpoints respond with supported values."""
    datasets = api_client.get("/datasets")
    assert datasets.status_code == 200
    dataset_codes = set(datasets.json()["datasets"])
    assert {"COCO", "FOOD101"}.issubset(dataset_codes)
    assert "demo" in dataset_codes

    models = api_client.get("/models")
    assert models.status_code == 200
    assert set(models.json()["models"]) == {"u_net_v1", "u_net_v2"}

    defaults = api_client.get("/train/config")
    assert defaults.status_code == 200
    config = defaults.json()["config"]
    assert config["epochs"] == 2
    assert config["batch_size"] == 4
    assert "learning_rate" in config


@pytest.mark.continues
def test_train_and_predict(api_client: TestClient) -> None:
    """Training a model returns a completed record and enables prediction."""
    train_request = {
        "dataset": "demo",
        "model": "u_net_v1",
    }
    response = api_client.post("/train", json=train_request)
    assert response.status_code == 200
    model_id = response.json()["model_id"]

    trained = _wait_for_model(api_client, model_id)
    history = trained["metrics"]["history"]
    assert history["epochs"]
    assert history["config"]["epochs"] >= 1
    assert trained["metrics"]["train_error"] >= 0
    if trained["metrics"]["val_error"] is not None:
        assert trained["metrics"]["val_error"] >= 0
    if trained["metrics"]["delta_a"] is not None:
        assert "mean" in trained["metrics"]["delta_a"]
    assert trained["status"] == "completed"

    predict_response = api_client.post(
        f"/predict?model_id={model_id}",
        files={"file": ("sample.png", _sample_image_bytes(), "image/png")},
    )
    assert predict_response.status_code == 200
    body = predict_response.json()
    assert body["model_id"] == model_id
    assert body["metrics"]["mean_abs_ab"] >= 0
    assert body["image_base64"]
