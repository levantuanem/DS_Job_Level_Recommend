from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.preprocess import preprocess_dataset


def main() -> None:
    metadata = preprocess_dataset(
        PROJECT_ROOT / "data" / "raw" / "postings.csv",
        PROJECT_ROOT / "data" / "processed",
        include_posting_engagement=False,
    )
    print(
        f"Saved {metadata['rows']} rows with transformed shape "
        f"{tuple(metadata['transformed_shape'])}"
    )


if __name__ == "__main__":
    main()