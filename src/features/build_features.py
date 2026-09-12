from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.features.text_features import add_text_features
from src.features.skill_extraction import extract_skill_features
from src.features.temporal_features import add_temporal_features
from src.features.feature_selection import select_features

# =========================
# PATHS
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = (ROOT_DIR/ "data"/"processed"/"postings_clean.csv")
OUTPUT_PATH = (ROOT_DIR/"models"/"preprocessor.pkl")
SELECTOR_PATH = (ROOT_DIR/"models"/"feature_selector.pkl")

# =========================
# BUILD FEATURES
# =========================
def build_features(apply_feature_selection=False, k=1000):

    # =========================
    # LOAD DATA
    # =========================
    data = pd.read_csv(DATA_PATH, low_memory=False)
    print("Data loaded successfully!")
    print("Shape:", data.shape)

    # =========================
    # TARGET
    # =========================
    target = "formatted_experience_level"
    # Remove rows with missing target
    data = data.dropna(subset=[target]).copy()
    x = data.drop(columns=[target]).copy()
    y = data[target].astype(str)
    print("\nTarget distribution:")
    print(y.value_counts())

    # =========================
    # TEXT FEATURES
    # =========================
    x = add_text_features(x)

    # =========================
    # SKILL FEATURES
    # =========================
    x = extract_skill_features(x)

    # =========================
    # TEMPORAL FEATURES
    # =========================
    x = add_temporal_features(x)

    # =========================
    # REMOVE RAW / ID COLUMNS
    # =========================
    columns_to_drop = [
        "job_id",
        "company_id",
        "zip_code",
        "fips",
        "salary_id",

        "original_listed_time",
        "listed_time",
        "expiry",
        "closed_time",

        "job_posting_url",
        "application_url",

        "title",
        "description",
        "skills_desc",
    ]

    x = x.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in x.columns
        ]
    )

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )

    # =========================
    # FEATURE GROUPS
    # =========================
    numeric_features = [
        "max_salary",
        "med_salary",
        "min_salary",
        "normalized_salary",
        "applies",
        "views",

        # Text statistics
        "title_length",
        "description_length",
        "skills_length",
        "description_word_count",
        "skills_word_count",

        # Temporal features
        "posting_year",
        "posting_month",
        "posting_day",
        "posting_dayofweek",
        "posting_quarter",
        "posting_hour",
        "is_weekend",

        # Skill count
        "skill_count",
    ]

    categorical_features = [
        "company_name",
        "location",
        "pay_period",
        "formatted_work_type",
        "posting_domain",
        "application_type",
        "work_type",
        "currency",
        "compensation_type",
    ]

    binary_features = [
        "remote_allowed",
        "sponsored",
    ]

    # =========================
    # SKILL FEATURES
    # =========================
    skill_features = [
        column
        for column in x.columns
        if column.startswith("skill_")
        and column != "skill_count"
    ]

    numeric_features.extend(
        skill_features
    )

    # =========================
    # CHECK FEATURES
    # =========================
    all_structured_features = (
        numeric_features
        + categorical_features
        + binary_features
    )

    missing_features = [
        column
        for column in all_structured_features
        if column not in x.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if "combined_text" not in x.columns:
        raise ValueError(
            "Missing required text feature: combined_text"
        )

    # =========================
    # NUMERICAL PIPELINE
    # =========================
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # =========================
    # CATEGORICAL PIPELINE
    # =========================
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    # =========================
    # BINARY PIPELINE
    # =========================
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # =========================
    # TF-IDF
    # =========================
    tf_idf_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=10,
        max_df=0.95,
        max_features=20000
    )

    # =========================
    # COLUMN TRANSFORMER
    # =========================
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("binary", binary_pipeline, binary_features),
            ("text", tf_idf_vectorizer, "combined_text")
        ]
    )

    # =========================
    # FIT / TRANSFORM
    # =========================
    print("\nFitting preprocessor...")
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    print("Features before selection:", x_train.shape[1])

    # =========================
    # FEATURE SELECTION
    # =========================
    selector = None
    if apply_feature_selection:
        print(f"\nSelecting top {k} features...")

        x_train, x_test, selector = select_features(
            x_train,
            y_train,
            x_test,
            k=k
        )

        SELECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(selector, SELECTOR_PATH)
        print("Feature selector saved successfully!")

    # =========================
    # SAVE PREPROCESSOR
    # =========================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, OUTPUT_PATH)
    print("\nPreprocessor saved successfully!")
    print("X_train shape:", x_train.shape)
    print("X_test shape:", x_test.shape)

    # =========================
    # RETURN
    # =========================
    return (x_train, x_test, y_train, y_test)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    build_features(apply_feature_selection=False, k=1000)

