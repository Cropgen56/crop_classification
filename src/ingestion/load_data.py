from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_excel_sheet(file_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """Load a single worksheet from an Excel file."""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def list_excel_sheets(file_path: str | Path) -> list[str]:
    """Return all worksheet names from the Excel workbook."""
    return pd.ExcelFile(file_path).sheet_names
