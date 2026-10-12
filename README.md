# Maharashtra Crop Regime Clustering Project

This repository is being developed in phases. The current scope is Phase 0: project foundation and data audit for a Maharashtra crop-classification effort based on historical and village-level agricultural data.

## Current status

The project currently relies on the following source files in the workspace root:

- MH_Crop_History_1997_2019.xlsx
- MH_Crop_Training_Data.xlsx
- plan.md
- crops.md

These files provide the initial evidence base for historical crop patterns, village and district metadata, crop calendars, and phenology references.

## Phase 0 goals

- establish a clean project structure
- separate raw data from processed outputs
- create a reproducible analysis workflow
- define a clustering-first approach for crop regime discovery
- prepare the project for Phase 1: feature engineering

## Repository structure

- data/raw/: source files and untouched raw data
- data/processed/: cleaned and transformed datasets
- data/labels/: cluster labels and review outputs
- src/: Python modules for ingestion, feature engineering, clustering, labeling, and validation
- notebooks/: exploratory analysis notebooks
- tests/: automated checks for data and pipeline functions
- docs/: project documentation
- pipelines/: pipeline entry points

## Working principle

The project starts with clustering crop-regime patterns rather than forcing exact crop labels from limited historical data. The clusters will be interpreted and labeled by agronomic meaning before a rule-based crop classifier is built.

## Next phase

The immediate next step is Phase 1: build feature vectors from the current data and prepare them for clustering.
