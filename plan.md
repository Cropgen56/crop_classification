# Implementation Plan: Rule-Based Satellite Crop Classification for Maharashtra

## 1. Project Overview

Build a satellite-based crop classification system for Maharashtra using multi-temporal satellite imagery and an explainable, rule-based classification engine.

The system will accept a village location or boundary, determine the analysis timestamp, retrieve historical satellite observations, calculate vegetation and radar features, analyse their temporal patterns, and predict the likely crop type.

The initial version will use deterministic rules rather than a trained machine learning model. The architecture should allow machine learning to be added later without rewriting the satellite data pipeline.

### Primary objectives

- Classify agricultural land using satellite time-series data.
- Distinguish seasonal cropping systems from perennial/orchard crops.
- Determine likely Kharif, Rabi and Zaid cropping seasons.
- Identify individual crops using temporal vegetation and radar patterns.
- Support Maharashtra-specific crop calendars and local rules.
- Return predictions with supporting evidence and uncertainty information.
- Eventually map predictions and estimated crop area at village level.

### Initial target crops

**Seasonal crops:** paddy, jowar, bajra, maize, ragi, tur, moong, urad, groundnut, soybean, sunflower, sesame, nigerseed, cotton, wheat, barley, gram, masur, rapeseed and mustard, safflower and jute.

**Perennial/orchard crops:** orange (Nagpur mandarin), sapota and other locally relevant orchard crops.

**Additional classes:** fallow land, bare soil, water, built-up land, non-crop vegetation and unknown/insufficient evidence.

The initial target list is a scope, not a claim that satellite time series can already distinguish every listed crop reliably.

---

## 2. Proposed Technology Stack

| Component | Recommended technology | Purpose |
|---|---|---|
| Satellite data access | Google Earth Engine (GEE) | Retrieve and process Sentinel imagery |
| Optical satellite | Sentinel-2 Surface Reflectance Harmonized | Vegetation indices and spectral features |
| Radar satellite | Sentinel-1 GRD | Radar backscatter and change features |
| Additional radar | NISAR L-band/S-band products | Optional enhancement in a later phase |
| Data processing | Python, NumPy, Pandas | Feature extraction and time-series analysis |
| Geospatial processing | GeoPandas, Shapely, Rasterio | Field boundaries, geometries and raster operations |
| Rule engine | Python configuration-driven rules | Hierarchical crop classification |
| API | FastAPI | Expose classification functionality |
| Storage | PostgreSQL + PostGIS | Store boundaries, observations, features and predictions |
| Initial visualization | Leaflet or MapLibre | Display classified maps and time series |
| Development environment | Jupyter notebooks | Explore satellite observations and calibrate rules |

Use the following Earth Engine collections as the initial data sources:

- Sentinel-2: `COPERNICUS/S2_SR_HARMONIZED`
- Sentinel-1: `COPERNICUS/S1_GRD`

Official documentation: [Sentinel-2 Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) and [Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD).

NISAR should initially be an optional data adapter. Its data access and processing requirements differ from those of the standard Earth Engine Sentinel collections. Check available acquisitions and product coverage through [NASA NISAR Data](https://science.nasa.gov/mission/nisar/data/) before integrating it.

---

## 3. System Architecture

The application should follow this processing pipeline:

```text
User Request
    |
    v
Resolve Village Boundary / Field Polygons
    |
    v
Determine As-of Timestamp
    |
    v
Retrieve Satellite Observations
    |
    +----------------------+
    |                      |
    v                      v
Sentinel-2 Optical     Sentinel-1 Radar
    |                      |
    v                      v
Cloud Masking          Radar Filtering
    |                      |
    v                      v
Vegetation Indices     Backscatter Features
    |                      |
    +-----------+----------+
                |
                v
       Align Time Series
                |
                v
      Extract Temporal Features
                |
                v
       Data Quality Assessment
                |
                v
       Land-Cover Classification
                |
                v
   Seasonal vs Perennial Classification
                |
                v
        Crop-Specific Rules
                |
                v
   Prediction + Evidence + Uncertainty
                |
                v
       Store Results / Display Map
```

Each component should be independently testable. Satellite retrieval must not be tightly coupled to the crop classification rules.

---

## 4. Phase 1 — Input and Spatial Data Handling

### 4.1 Define the input contract

The first version should accept:

- `village_id`: official village identifier, when available.
- `village_boundary`: GeoJSON polygon or a resolvable boundary reference.
- `as_of_timestamp`: timestamp up to which observations may be used.
- `analysis_year`: optional crop-year selection.
- `field_ids`: optional list of field identifiers, when field boundaries are available.

Example request:

```json
{
  "village_id": "example_village_001",
  "village_boundary": null,
  "as_of_timestamp": "2026-10-12T00:00:00+05:30",
  "analysis_year": 2026
}
```

The village identifier is illustrative, not an actual government identifier.

If a boundary is supplied, validate its geometry and coordinate reference system. If only a village identifier is supplied, resolve it using an approved administrative boundary dataset.

### 4.2 Timestamp handling

Use the current timestamp by default, interpreted in the Asia/Kolkata timezone.

The timestamp controls which observations are eligible for the prediction. Convert dates consistently when querying satellite archives, and never use acquisitions later than the requested analysis date.

For reproducibility, preserve the requested timestamp even when the latest available satellite observation is several days old.

### 4.3 Field-level versus village-level processing

**Prefer field-level polygons over a single village-average time series.**

A village can contain soybean, cotton, orchards, fallow land and other land uses simultaneously. Averaging NDVI across the entire village would hide these differences.

The preferred hierarchy is:

1. Classify individual field polygons when reliable boundaries are available.
2. Otherwise, classify raster pixels or spatial units within the village.
3. Aggregate predictions to village-level crop-area estimates.

If the only available input is a village polygon, the application must not automatically return a single crop label for the whole village.

Deliverable: a validated boundary-processing module that returns a usable geometry for the satellite pipeline.

---

## 5. Phase 2 — Satellite Data Retrieval

### 5.1 Determine the analysis period

Retrieve historical data appropriate to the current or requested analysis date.

The initial defaults should be configurable:

- Seasonal crop analysis: current crop cycle plus sufficient preceding observations to establish the baseline.
- Perennial/orchard analysis: multiple crop years wherever available.
- Historical comparison: corresponding seasons from previous years.
- Minimum observation count and maximum allowable data gaps: configurable by sensor and analysis mode.

Do not assume that the current calendar month alone identifies the growing season.

### 5.2 Sentinel-2 processing

Use harmonized surface-reflectance data.

Processing steps:

1. Filter imagery by village geometry and permitted date range.
2. Mask clouds, cloud shadows and invalid pixels using the available scene-classification and quality information.
3. Apply correct reflectance scale factors.
4. Calculate the initial vegetation and land-surface indices.
5. Retain the acquisition date and valid-pixel information.
6. Build temporal composites.

The first version should use a regular 10-day composite cadence when observations support it. In heavily clouded periods, preserve gaps or use wider aggregation windows instead of manufacturing observations through aggressive interpolation.

### 5.3 Sentinel-1 processing

Filter Sentinel-1 GRD observations using a consistent acquisition configuration wherever possible.

Extract:

- VV backscatter.
- VH backscatter.
- VH/VV polarization contrast.
- Changes in VV and VH across observation dates.
- Observation counts and acquisition metadata.

Earth Engine's GRD collection contains calibrated, terrain-corrected backscatter represented in decibels. Be consistent when calculating polarization ratios: subtracting VH dB from VV dB is the dB equivalent of a linear-power ratio, but it must be named and interpreted consistently.

Avoid mixing incompatible acquisition configurations without documenting the effects.

### 5.4 NISAR integration

Do not make NISAR a dependency for the minimum viable product.

In a later phase:

1. Confirm acquisition coverage for the target villages.
2. Obtain suitable calibrated radar products through the official data-access systems.
3. Verify product geometry, polarization, units and temporal availability.
4. Build a separate processing adapter.
5. Evaluate whether NISAR features improve classification before using them in the production rules.

Deliverable: a sensor-independent observation table with aligned timestamps and documented units.

---

## 6. Phase 3 — Feature Engineering

### 6.1 Initial feature set

Start with a small, interpretable set of features.

| Feature | Source | Purpose |
|---|---|---|
| NDVI | Sentinel-2 | Vegetation greenness and growth |
| EVI | Sentinel-2 | Additional canopy/vegetation response |
| NDMI | Sentinel-2 | Canopy water-content-related changes |
| NDRE | Sentinel-2 | Red-edge vegetation response |
| BSI | Sentinel-2 | Bare-soil exposure and field preparation clues |
| MNDWI | Sentinel-2 | Water detection |
| VV | Sentinel-1 | Radar backscatter response |
| VH | Sentinel-1 | Cross-polarized radar response |
| VH/VV contrast | Sentinel-1 | Relative polarization response |
| Observation count | Both | Data quality and temporal coverage |

Important implementation detail: red-edge and SWIR indices use 20 m Sentinel-2 bands, while many visible/NIR bands are 10 m. Select and document a consistent analysis grid rather than treating every index as a 10 m measurement.

### 6.2 Vegetation-index formulas

Use standard formulas with explicit protection against zero denominators.

```text
NDVI = (NIR - Red) / (NIR + Red)

EVI = 2.5 * (NIR - Red)
      / (NIR + 6 * Red - 7.5 * Blue + 1)

NDMI = (NIR - SWIR1) / (NIR + SWIR1)

NDRE = (NIR_narrow - RedEdge) / (NIR_narrow + RedEdge)

BSI = (SWIR1 + Red - NIR - Blue)
      / (SWIR1 + Red + NIR + Blue)

MNDWI = (Green - SWIR1) / (Green + SWIR1)
```

For NDRE, use compatible narrow-NIR and red-edge bands, such as Sentinel-2 B8A and B5.

### 6.3 Temporal feature extraction

For each field or spatial unit, calculate:

- Minimum, maximum and median NDVI.
- NDVI amplitude: maximum minus minimum.
- Approximate green-up date.
- Maximum vegetation date.
- Decline period and rate.
- Duration of sustained greenness.
- Integrated vegetation-index value over the crop cycle.
- Number of substantial growth peaks.
- Temporal variation in NDMI and NDRE.
- Temporal changes in VV and VH.
- Number of valid observations and longest data gap.
- Similarity between the current and previous years' seasonal patterns.

These features will power the first rule engine.

Do not assign crop-specific numeric thresholds until they have been examined against verified local examples.

Deliverable: a feature extractor that converts satellite observations into a standardized temporal feature record.

---

## 7. Phase 4 — Build the Hierarchical Rule Engine

Implement the rule engine as distinct stages. A later stage must be allowed to return an inconclusive result rather than forcing a crop prediction.

### Stage 1: Observation quality assessment

Check whether the time series has:

- Enough valid observations.
- Acceptable temporal gaps.
- Adequate spatial coverage of the field.
- Comparable dates and valid index values.

Output:

```text
good_evidence
limited_evidence
insufficient_evidence
```

Do not interpret missing observations as low NDVI.

### Stage 2: Broad land-cover classification

Distinguish potential:

- Agricultural fields.
- Water.
- Built-up areas.
- Bare or fallow land.
- Persistent vegetation.
- Unknown land cover.

Use spectral and temporal evidence, supplemented by geographic masks where legitimate.

NDVI alone is insufficient for confidently separating natural vegetation, orchards and all other land-cover types.

### Stage 3: Seasonal versus perennial classification

Use a combination of:

- Seasonal vegetation amplitude.
- Crop-cycle timing.
- Persistence of green vegetation.
- Repeated annual trajectories.
- Radar changes.
- Existing orchard or field boundary data.

Possible outputs:

```text
seasonal_crop_candidate
perennial_orchard_candidate
other_perennial_vegetation
non_crop
unknown
```

Persistent greenness is evidence for a perennial candidate, not proof of an orchard.

### Stage 4: Cropping-season classification

For seasonal candidates, identify likely crop cycles:

```text
kharif
rabi
zaid_or_summer
multiple_or_mixed_cycles
unknown_season
```

Use locally appropriate agricultural calendars alongside the observed phenology. The decision should reflect the actual temporal pattern, not a month-based lookup alone.

Perennial crops should be assigned a separate cropping-system category rather than being forced into Kharif or Rabi.

### Stage 5: Crop-specific candidate generation

For each supported crop, define a configurable rule containing:

- Applicable season and geographic region.
- Expected timing of growth and maturity.
- Relevant temporal features.
- Supporting Sentinel-1 features.
- Conditions that contradict the candidate.
- Minimum evidence requirements.
- Rule version and supporting observations.

Example configuration structure:

```yaml
rules:
  soybean:
    crop_group: seasonal
    seasons:
      - kharif
    features:
      - ndvi_amplitude
      - greenup_date
      - peak_date
      - decline_rate
      - vh_temporal_change
    requires_local_calibration: true

  orange:
    crop_group: perennial_orchard
    seasons: []
    features:
      - multi_year_vegetation_persistence
      - seasonal_ndvi_variation
      - ndmi_variation
      - radar_temporal_features
    requires_local_calibration: true
```

This is a configuration example, not a scientifically validated soybean or orange rule.

### Stage 6: Candidate ranking and uncertainty

Score each candidate using configurable evidence weights.

For example:

```text
candidate_score =
    phenology_score
    + optical_index_score
    + radar_evidence_score
    + geographic_prior_score
    - contradiction_penalty
```

Only use geographic priors derived from documented local knowledge or data, and make them weak enough that they cannot override contradictory satellite observations.

Return a crop candidate, the supporting features and plausible alternatives. Until scores are calibrated using reference data, call them **rule scores**, not probabilities.

For ambiguous orchards, allow `orchard_unknown` instead of forcing orange or sapota.

Deliverable: a testable rule engine with versioned, configurable thresholds and evidence weights.

---

## 8. Phase 5 — Prediction Output

Each prediction should include:

```json
{
  "spatial_unit_id": "example_field_001",
  "as_of_timestamp": "2026-10-12T00:00:00+05:30",
  "crop_group": "seasonal",
  "season_candidate": "kharif",
  "crop_candidates": [
    {
      "crop": "soybean",
      "rule_score": null
    }
  ],
  "evidence_quality": "limited_evidence",
  "supporting_features": [],
  "contradicting_features": [],
  "alternative_candidates": [],
  "data_gaps": [],
  "rule_version": "0.1.0"
}
```

This is a schema example. `null` indicates that the numerical score has not been computed; the initial implementation should populate actual values only after the scoring rules are defined.

For each output, preserve enough detail to answer:

- Which satellite observations were used?
- What were the main NDVI and radar trends?
- Which rules fired?
- Why was a crop selected or rejected?
- What other classes remain plausible?
- Was enough evidence available to classify the field?

At village level, aggregate the spatial-unit predictions into crop-area estimates and class-wise uncertainty summaries. Do not equate a predicted crop percentage with an official acreage estimate without validating its accuracy.

---

## 9. Phase 6 — Validation and Rule Calibration

A rule-based model still requires ground truth. Satellite trends alone cannot establish that crop labels are correct.

### 9.1 Build a reference set

Collect a geographically diverse set of verified fields containing:

- Crop name.
- Field geometry.
- Season and year.
- Observation date.
- Verification source.
- Confidence in the reference label.

Use field surveys, agricultural researchers, government datasets and other verified local information where available.

Include orange orchards as a dedicated reference class for the Vidarbha pilot.

### 9.2 Validate each hierarchy level

Measure performance separately for:

1. Broad land-cover classification.
2. Seasonal versus perennial classification.
3. Kharif/Rabi/Zaid classification.
4. Individual crop classification.
5. Village-level crop-area estimates.

Report a confusion matrix, per-class precision, recall and F1-score. Also report how often the system abstains because the evidence is insufficient.

Split evaluation data by village or other meaningful geographic unit, rather than randomly splitting neighboring pixels between training/calibration and testing. This provides a more realistic estimate of generalization.

### 9.3 Compare feature sets

Run controlled comparisons:

- Optical time series only.
- Optical plus Sentinel-1 radar.
- Optical and radar plus multi-year persistence.
- All of the above plus NISAR, when suitable data are available.

Only retain added complexity if the evaluation shows a meaningful improvement.

Deliverable: a validation report documenting class-level performance, failure cases and the limitations of each crop rule.

---

## 10. Phase 7 — API and Application Integration

After the pipeline works in notebooks, expose it through FastAPI.

Suggested endpoints:

```text
POST /v1/classifications
GET  /v1/classifications/{job_id}
GET  /v1/fields/{field_id}/timeseries
GET  /v1/villages/{village_id}/crop-summary
GET  /v1/classifications/{job_id}/evidence
```

The classification endpoint should accept the input geometry and analysis timestamp, validate the request, create a processing job and return a job identifier. Longer satellite-processing tasks should not block the HTTP request.

The application should show:

- Predicted crop map.
- Crop-area distribution for the village.
- NDVI and NDMI time-series charts.
- Sentinel-1 temporal trends.
- Prediction evidence and alternative candidates.
- Data gaps and uncertainty warnings.

Keep the satellite acquisition layer separate from the API and user interface. This will make future changes in data providers easier.

---

## 11. Phase 8 — NISAR and Advanced Enhancements

Once the Sentinel-1/Sentinel-2 version has been validated:

1. Inspect NISAR product coverage and available historical observations.
2. Add compatible radar features to the observation schema.
3. Compare NISAR-enhanced rules with the existing baseline.
4. Determine which crops benefit from L-band or S-band observations.
5. Retain NISAR features only where they improve measured performance.

Future improvements can include:

- Anomaly detection for unusual growing seasons.
- More robust multi-year crop-cycle comparison.
- Active collection of labels for confusing classes.
- Automated threshold calibration.
- A hybrid rule-based and machine-learning classifier.
- Crop-area estimation with explicit uncertainty intervals.

---

## 12. Suggested Repository Structure

```text
maharashtra-crop-classifier/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── classification.py
│   │   ├── timeseries.py
│   │   └── village_summary.py
│   ├── data/
│   │   ├── boundaries.py
│   │   ├── sentinel2.py
│   │   ├── sentinel1.py
│   │   └── nisar.py
│   ├── preprocessing/
│   │   ├── cloud_masking.py
│   │   ├── radar_processing.py
│   │   └── compositing.py
│   ├── features/
│   │   ├── vegetation_indices.py
│   │   └── temporal_features.py
│   ├── classifier/
│   │   ├── landcover_rules.py
│   │   ├── season_rules.py
│   │   ├── crop_rules.py
│   │   └── scoring.py
│   ├── schemas/
│   └── storage/
├── configs/
│   ├── crop_rules.yaml
│   ├── crop_calendars.yaml
│   └── data_quality.yaml
├── notebooks/
│   ├── 01_satellite_exploration.ipynb
│   ├── 02_timeseries_analysis.ipynb
│   └── 03_rule_calibration.ipynb
├── tests/
│   ├── test_indices.py
│   ├── test_temporal_features.py
│   └── test_rules.py
├── docs/
│   ├── data_dictionary.md
│   └── validation_report.md
├── requirements.txt
└── README.md
```

---

## 13. Implementation Milestones

| Milestone | Work | Completion criterion |
|---|---|---|
| M1 | Boundary resolution and satellite retrieval | Retrieve and inspect observations for one pilot village |
| M2 | Optical/radar preprocessing | Generate valid, timestamped composites and quality metadata |
| M3 | Index and temporal features | Produce repeatable feature records and time-series charts |
| M4 | Hierarchical rule engine | Generate explainable land-cover, season and crop candidates |
| M5 | Ground-truth validation | Produce per-class evaluation and document failure cases |
| M6 | FastAPI and map integration | Return predictions and expose evidence/time-series endpoints |
| M7 | NISAR evaluation | Establish whether additional radar features improve results |

### Recommended development order

Begin with M1–M3 before writing extensive crop-specific rules. It is critical to understand the actual temporal patterns in your chosen Maharashtra pilot area before setting thresholds.

For the first working version, select a small number of villages with suitable reference data. Include both seasonal fields and orange orchards if the selected region supports both. Expand the crop list after the data pipeline and validation procedure are reliable.

## 14. Definition of Done for Version 0.1

Version 0.1 will be considered functional when it can:

- Accept a village boundary and an as-of timestamp.
- Retrieve historical Sentinel-2 and Sentinel-1 observations.
- Generate a quality-controlled temporal feature table.
- Separate broad land-cover classes and identify likely seasonal/perennial candidates.
- Rank supported crop candidates using explainable rules.
- Return `unknown` or insufficient evidence when appropriate.
- Produce a map, temporal charts and a structured prediction record.
- Run against verified reference fields and produce a validation report.

The goal of Version 0.1 is a reproducible, explainable pipeline—not a claim of reliable statewide identification of every crop.

### Reference documentation

- [Google Earth Engine: Sentinel-2 Surface Reflectance Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Google Earth Engine: Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- [NASA: NISAR data access and products](https://science.nasa.gov/mission/nisar/data/)
- [Google Earth Engine: Noncommercial usage tiers](https://developers.google.com/earth-engine/guides/noncommercial_tiers)

