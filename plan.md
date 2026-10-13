# Current Implementation Plan

## 1. Status summary

This project is no longer a blank prototype. The repository already contains the working foundation for a clustering-first crop classification system for Maharashtra.

The current architecture is:

- historical crop data audit and feature construction
- crop-regime clustering from district-level seasonal patterns
- agronomic label interpretation for clusters
- crop rule scoring for likely crop matches
- field/village aggregation into a service-ready output
- a service contract that can be exposed via a FastAPI endpoint

The main goal now is to connect this logic to real satellite inputs and a clean API layer without discarding the validated rule-based foundation.

---

## 2. What is already implemented

The repository currently includes the following working modules:

- data preparation and feature engineering in [src/feature_engineering/build_region_features.py](src/feature_engineering/build_region_features.py)
- cluster discovery in [src/clustering/cluster_regions.py](src/clustering/cluster_regions.py)
- cluster interpretation and agronomic labeling in [src/labeling/cluster_labeling.py](src/labeling/cluster_labeling.py)
- crop-rule scoring in [src/rules/rule_engine.py](src/rules/rule_engine.py)
- field-level candidate ranking in [src/rules/field_candidate_generator.py](src/rules/field_candidate_generator.py)
- village-level aggregation in [src/rules/village_classifier.py](src/rules/village_classifier.py)
- service response contract in [src/service/classification_service.py](src/service/classification_service.py)
- crop taxonomy in [src/config/crop_groups.yaml](src/config/crop_groups.yaml)
- rule definitions in [src/config/crop_rules.yaml](src/config/crop_rules.yaml)

The project is currently validated around a clustering-first, explainable classification model rather than a pure deep-learning or direct pixel classifier.

---

## 3. Current project logic

### 3.1 Historical feature pipeline

The project starts from district and village agriculture data, then builds a region profile based on:

- crop shares by season
- perennial crop share
- crop diversity and entropy
- top crop dominance
- seasonal balance
- district-level crop distribution

This produces the feature matrix used for regime discovery.

### 3.2 Clustering layer

The clustering step uses KMeans with StandardScaler and silhouette score selection to discover natural crop regimes. The output is a district-to-cluster assignment plus a cluster summary.

### 3.3 Labeling layer

Cluster summaries are interpreted with agronomic logic and crop-index references. This converts abstract clusters into meaningful labels such as seasonal paddy-heavy systems, mixed dryland systems, or perennial orchard patterns.

### 3.4 Rule-based classification

Once clusters are labeled, the system scores field or village profiles against crop rules. These rules include seasonal signal checks, crop-family logic, and weighting for important Maharashtra crops such as paddy, jowar, bajra, maize, wheat, gram, cotton, sugarcane, and orchard crops.

### 3.5 Output contract

The service layer packages the final output into a compact JSON response with:

- village_id
- dominant_crop
- dominant_share
- field_count
- summary
- top_candidates
- crop_mix

This contract is already suitable for a FastAPI endpoint and later frontend integration.

---

## 4. Real next phases

### Phase 0: foundation and data audit

Status: complete.

This included establishing the repo structure, cleaning the data-loading logic, fixing import paths, validating XLSX assumptions, and confirming the working environment under the agri Conda environment.

### Phase 1: feature engineering and clustering

Status: complete.

This stage created the region feature table and used clustering to discover crop regimes.

### Phase 2: labeling, rules, and interpretation

Status: complete.

This stage mapped clusters to agronomic crop systems and added a deterministic crop scoring engine.

### Phase 3: field and village classification

Status: complete.

The project now supports field-level candidate generation and village aggregate classification.

### Phase 4: service contract and API exposure

Status: next.

The target here is a FastAPI application that wraps the current service contract and exposes endpoints such as:

- health
- classify-field
- classify-village
- optional batch classification for multiple fields

### Phase 5: real Sentinel feature extraction

Status: next.

This is the true operational step: replace the current static profile logic with geospatial satellite extraction using Sentinel-1 and Sentinel-2.

The system should:

- accept a GeoJSON boundary or field polygon
- use Earth Engine or a local processing pipeline to fetch S1/S2 data
- calculate time-series vegetation and radar features
- extract NDVI, NDRE, EVI, SAVI, and related indices when needed
- derive seasonal metrics and crop signatures
- combine these values with historical crop priors and rule scores

### Phase 6: calibration and validation

Status: pending.

Use labeled field samples and known crop calendars to calibrate threshold values and tune the rule engine against actual observed patterns.

---

## 5. Data and environment requirements

### Required inputs

- Maharashtra historical crop data files
- village or field boundary GeoJSON
- known village IDs or district references
- crop calendar and season references
- sample labeled field data for validation

### Satellite access

For live operational work, the project needs a valid Google Earth Engine or equivalent service account setup. The credential file should remain local and outside Git.

Use the environment variable pattern:

- GOOGLE_APPLICATION_CREDENTIALS = local path to the service account JSON

Keep the actual JSON file out of version control and store it only in a secure local environment or secret store.

### Recommended data layers

- Sentinel-2 SR Harmonized imagery
- Sentinel-1 GRD backscatter
- cloud mask and pixel-quality filters
- 10- to 30-day aggregation windows depending on season and cloud state
- seasonal crop calendars for Maharashtra

---

## 6. Implementation priorities

### Immediate priority

1. Build the FastAPI app around the existing classification service contract.
2. Add input validation for field and village payloads.
3. Add basic API docs and health checks.
4. Keep the service contract stable so later satellite ingestion only changes the underlying feature calculations.

### Next technical priority

5. Add the real Sentinel-1 and Sentinel-2 extraction adapter.
6. Convert the extracted temporal signatures into the same profile structure expected by the rule engine.
7. Reuse the already-built cluster and rule logic instead of reinventing the pipeline.

---

## 7. Final project target

The end objective is a system that takes a field or village boundary, extracts the right satellite time-series features, combines them with historical agricultural patterns, and returns a crop classification with explainable reasons.

The architecture should remain modular:

- data layer
- feature extraction layer
- clustering layer
- labeling layer
- rule engine
- API layer
- validation layer

This is the correct direction for the current repository state and should remain the working plan going forward.

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

