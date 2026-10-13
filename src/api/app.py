from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, StreamingResponse

from src.rules.field_candidate_generator import classify_field_profile
from src.service.classification_service import classify_village_request


app = FastAPI(
    title="Maharashtra Crop Classifier API",
    version="0.7.0",
    description="Time Series Clustered Rule-based crop classification service for village and field profiles.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
INDEX_HTML = Path(__file__).resolve().parents[2] / "index.html"


def _analysis_end_date() -> str:
    return date.today().isoformat()


def _analysis_start_date() -> str:
    return (date.today() - timedelta(days=365)).isoformat()


class FieldProfileInput(BaseModel):
    field_id: str | None = None
    geometry: dict[str, Any] | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    dominant_top_crop: str | None = None
    avg_kharif_share: float | None = None
    avg_rabi_share: float | None = None
    avg_zaid_share: float | None = None
    avg_other_share: float | None = None
    top_crop_share: float | None = None

    @property
    def profile(self) -> dict[str, Any]:
        values = dict(self.properties)
        for key in [
            "field_id",
            "dominant_top_crop",
            "avg_kharif_share",
            "avg_rabi_share",
            "avg_zaid_share",
            "avg_other_share",
            "top_crop_share",
        ]:
            if getattr(self, key) is not None:
                values[key] = getattr(self, key)
        if self.field_id is not None:
            values["field_id"] = self.field_id
        return values


class VillageClassificationRequest(BaseModel):
    village_id: str = "unknown"
    district_name: str | None = None
    fields: list[FieldProfileInput] | dict[str, Any] = Field(default_factory=list)
    start_date: str = Field(default_factory=_analysis_start_date)
    end_date: str = Field(default_factory=_analysis_end_date)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "crop_classifier_api"}


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/classify-field")
def classify_field(payload: FieldProfileInput) -> dict[str, Any]:
    try:
        result = classify_field_profile(payload.model_dump())
        return {
            "field_id": payload.field_id,
            "best_crop": result["best_crop"],
            "best_score": result["best_score"],
            "season_label": result["season_label"],
            "evidence_summary": result["evidence_summary"],
            "ranked_candidates": result["ranked_candidates"],
        }
    except Exception as exc:  # pragma: no cover - defensive API layer
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/classify-village")
def classify_village(payload: VillageClassificationRequest) -> dict[str, Any]:
    try:
        village_payload = _service_payload(payload)
        return classify_village_request(village_payload)
    except Exception as exc:  # pragma: no cover - defensive API layer
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _service_payload(payload: VillageClassificationRequest) -> dict[str, Any]:
    return {
        "village_id": payload.village_id,
        "district_name": payload.district_name or "",
        "fields": [field.model_dump(exclude_none=True) for field in payload.fields]
        if isinstance(payload.fields, list)
        else payload.fields,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
    }


@app.post("/classify-village/stream")
def classify_village_stream(payload: VillageClassificationRequest) -> StreamingResponse:
    events: Queue[dict[str, Any]] = Queue()
    finished = False

    # Immediate acknowledgement so HTTP stream begins instantly
    events.put({
        "type": "progress",
        "stage": "geojson_validated",
        "completed": 0,
        "total": 1,
        "message": f"GeoJSON received ({len(payload.fields) if isinstance(payload.fields, list) else 'FeatureCollection'}). Preparing classification...",
    })

    def run_classification() -> None:
        nonlocal finished
        try:
            result = classify_village_request(
                _service_payload(payload),
                progress_callback=lambda progress: events.put({"type": "progress", **progress}),
            )
            events.put({"type": "result", "result": result})
        except Exception as exc:
            events.put({"type": "error", "detail": str(exc)})
        finally:
            finished = True

    worker = Thread(target=run_classification, daemon=True)

    def stream_events():
        worker.start()
        while not finished or not events.empty():
            try:
                event = events.get(timeout=1.0)
                yield f"data: {json.dumps(event, allow_nan=False)}\n\n"
            except Empty:
                if not finished:
                    yield ": keep-alive\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
