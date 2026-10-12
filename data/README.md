# Data layout

This directory keeps the project data organized by stage.

## raw/

Contains the source Excel workbooks and any external raw datasets. These files should remain untouched for traceability.

## processed/

Contains cleaned tables, normalized features, and cluster-ready datasets.

## labels/

Contains cluster labels, manual review records, and label mappings.

## Phase 0 note

At this stage, the raw files remain in the project root because they were already supplied in the workspace. The purpose of this structure is to create a clean pipeline foundation and make future data ingests reproducible.
