from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        ROOT / "data" / "raw" / "historical",
        ROOT / "data" / "raw" / "training",
        ROOT / "data" / "processed",
        ROOT / "data" / "labels",
        ROOT / "src" / "config",
        ROOT / "src" / "ingestion",
        ROOT / "src" / "feature_engineering",
        ROOT / "src" / "clustering",
        ROOT / "src" / "labeling",
        ROOT / "src" / "evaluation",
        ROOT / "src" / "utils",
        ROOT / "notebooks",
        ROOT / "tests",
        ROOT / "docs",
        ROOT / "pipelines",
    ]

    for directory in required:
        directory.mkdir(parents=True, exist_ok=True)

    print("Phase 0 setup complete.")
    print("Project root:", ROOT)
    print("Prepared directories:")
    for directory in required:
        print(" -", directory.relative_to(ROOT))


if __name__ == "__main__":
    main()
