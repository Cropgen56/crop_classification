from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering.sentinel_field_extractor import build_village_field_profiles
from src.ingestion.village_geojson import normalize_village_fields
from src.rules.field_candidate_generator import classify_field_profile
from src.service.classification_service import classify_village_request


app = FastAPI(
    title="Maharashtra Crop Classifier API",
    version="0.7.0",
    description="Rule-based crop classification service for village and field profiles.",
)


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
    fields: list[FieldProfileInput] | dict[str, Any] = Field(default_factory=list)

    @property
    def normalized_fields(self) -> list[dict[str, Any]]:
        profiles = build_village_field_profiles(self.fields)
        if profiles:
            return profiles
        return normalize_village_fields(self.fields)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "crop_classifier_api"}


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
        village_payload = {
            "village_id": payload.village_id,
            "fields": payload.normalized_fields,
        }
        return classify_village_request(village_payload)
    except Exception as exc:  # pragma: no cover - defensive API layer
        raise HTTPException(status_code=400, detail=str(exc)) from exc
