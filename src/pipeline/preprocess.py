import json
from pathlib import Path
import joblib
import pandas as pd
from src.features.build_features import build_features
from src.features.preprocessing import build_preprocessor


def preprocess_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    include_posting_engagement: bool = False) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    features, target = build_features(
        data,
        include_posting_engagement=include_posting_engagement,
    )

    preprocessor = build_preprocessor(features)
    transformed_features = preprocessor.fit_transform(features)

    feature_matrix_path = output_dir / "features_matrix.joblib"
    target_path = output_dir / "target.csv"
    preprocessor_path = output_dir / "feature_preprocessor.joblib"
    metadata_path = output_dir / "feature_metadata.json"

    joblib.dump(transformed_features, feature_matrix_path)
    joblib.dump(preprocessor, preprocessor_path)
    target.to_csv(target_path, index=False)

    metadata = {
        "input_path": str(input_path),
        "rows": int(len(features)),
        "raw_feature_columns": list(features.columns),
        "target_column": target.name,
        "transformed_shape": list(transformed_features.shape),
        "include_posting_engagement": include_posting_engagement,
        "artifacts": {
            "features_matrix": str(feature_matrix_path),
            "target": str(target_path),
            "preprocessor": str(preprocessor_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata