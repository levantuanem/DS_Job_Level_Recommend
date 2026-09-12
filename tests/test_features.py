import pandas as pd
from src.features.build_features import build_features
from src.features.numerical import create_numerical_features
from src.features.text_features import create_text_features
from src.features.skill_extraction import extract_skill_features
from src.features.categorical import prepare_categorical_features
from src.features.leakage import check_leakage
from src.features.preprocessing import build_preprocessor
from src.features.temporal import create_temporal_features


# ============================================================
# Test Data
# ============================================================


def create_test_dataframe():

    return pd.DataFrame(
        {
            "min_salary": [50000, 60000, 70000],
            "med_salary": [65000, 75000, 85000],
            "max_salary": [80000, 90000, 100000],
            "normalized_salary": [65000, 75000, 85000],

            "views": [100, 200, 500],
            "applies": [10, 20, 50],

            "work_type": [
                "Full-time",
                "Part-time",
                "Contract",
            ],

            "formatted_work_type": [
                "Full-time",
                "Part-time",
                "Contract",
            ],

            "remote_allowed": [
                True,
                False,
                True,
            ],

            "pay_period": [
                "YEARLY",
                "YEARLY",
                "HOURLY",
            ],

            "currency": [
                "USD",
                "USD",
                "USD",
            ],

            "location": [
                "New York",
                "Boston",
                "Chicago",
            ],

            "application_type": [
                "OffsiteApply",
                "ComplexOnsiteApply",
                "OffsiteApply",
            ],

            "title": [
                "Python Developer",
                "Data Analyst",
                "Machine Learning Engineer",
            ],

            "description": [
                "Python developer with SQL experience.",
                "Data analyst using Excel and SQL.",
                "Machine learning engineer using Python and AWS.",
            ],

            "skills_desc": [
                "Python, SQL",
                "Excel, SQL",
                "Python, AWS, Machine Learning",
            ],

            "formatted_experience_level": [
                "Entry level",
                "Associate",
                "Mid-Senior level",
            ],
        }
    )


# ============================================================
# Numerical Feature Tests
# ============================================================


def test_numerical_features():

    df = create_test_dataframe()

    result = create_numerical_features(df)

    assert "salary_range" in result.columns
    assert "log_salary" in result.columns
    assert "application_rate" in result.columns
    assert "view_to_apply_ratio" in result.columns


def test_salary_range():

    df = create_test_dataframe()

    result = create_numerical_features(df)

    assert result.loc[0, "salary_range"] == 30000


def test_application_rate():

    df = create_test_dataframe()

    result = create_numerical_features(df)

    assert result.loc[0, "application_rate"] == 0.1


# ============================================================
# Temporal Feature Tests
# ============================================================


def test_temporal_features():

    start = pd.Timestamp("2024-01-05", tz="UTC")
    df = pd.DataFrame(
        {
            "listed_time": [int(start.timestamp() * 1000)],
            "expiry": [int((start + pd.Timedelta(days=3)).timestamp() * 1000)],
        }
    )

    result = create_temporal_features(df)

    assert result.loc[0, "duration_days"] == 3
    assert result.loc[0, "post_month"] == 1
    assert result.loc[0, "post_dayofweek"] == 4
    assert result.loc[0, "is_weekend"] == 0


# ============================================================
# Preprocessing Tests
# ============================================================


def test_preprocessor_transforms_features():

    df = create_test_dataframe()
    X, _ = build_features(df)

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


# ============================================================
# Categorical Feature Tests
# ============================================================


def test_categorical_features():

    df = create_test_dataframe()

    columns = [
        "work_type",
        "formatted_work_type",
        "remote_allowed",
        "pay_period",
        "currency",
        "location",
        "application_type",
    ]

    result = prepare_categorical_features(
        df,
        columns,
    )

    for column in columns:
        assert column in result.columns


# ============================================================
# Text Feature Tests
# ============================================================


def test_text_features():

    df = create_test_dataframe()

    text_columns = [
        "title",
        "description",
        "skills_desc",
    ]

    result = create_text_features(
        df,
        text_columns,
    )

    expected_columns = [
        "combined_text",
        "text_length",
        "word_count",
        "sentence_count",
        "keyword_count",
        "technical_keyword_count",
        "management_keyword_count",
        "leadership_keyword_count",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_text_features_are_not_empty():

    df = create_test_dataframe()

    result = create_text_features(
        df,
        [
            "title",
            "description",
            "skills_desc",
        ],
    )

    assert (result["text_length"] > 0).all()
    assert (result["word_count"] > 0).all()


# ============================================================
# Skill Extraction Tests
# ============================================================


def test_skill_extraction():

    df = create_test_dataframe()

    result = extract_skill_features(
        df,
        [
            "title",
            "description",
            "skills_desc",
        ],
    )

    assert "has_python" in result.columns
    assert "has_sql" in result.columns
    assert "has_aws" in result.columns
    assert "has_machine_learning" in result.columns
    assert "skill_count" in result.columns


def test_python_skill():

    df = create_test_dataframe()

    result = extract_skill_features(
        df,
        [
            "title",
            "description",
            "skills_desc",
        ],
    )

    assert result.loc[0, "has_python"] == 1


def test_sql_skill():

    df = create_test_dataframe()

    result = extract_skill_features(
        df,
        [
            "title",
            "description",
            "skills_desc",
        ],
    )

    assert result.loc[0, "has_sql"] == 1


def test_skill_count():

    df = create_test_dataframe()

    result = extract_skill_features(
        df,
        [
            "title",
            "description",
            "skills_desc",
        ],
    )

    assert result.loc[0, "skill_count"] > 0


def test_skill_matching_uses_word_boundaries():

    df = pd.DataFrame(
        {
            "title": ["JavaScript Developer", "R Analyst", "MySQL Developer"],
            "description": ["JavaScript", "R programming", "MySQL"],
        }
    )

    result = extract_skill_features(df, ["title", "description"])

    assert result.loc[0, "has_javascript"] == 1
    assert result.loc[0, "has_java"] == 0
    assert result.loc[1, "has_r"] == 1
    assert result.loc[2, "has_sql"] == 0


# ============================================================
# Leakage Tests
# ============================================================


def test_leakage_detection():

    df = create_test_dataframe()

    df = create_numerical_features(df)

    report = check_leakage(
        df,
        "formatted_experience_level",
    )

    assert report["target_exists"] is True

    assert "views" in report["suspicious_features"]
    assert "applies" in report["suspicious_features"]


def test_build_features_can_remove_posting_engagement():

    df = create_test_dataframe()

    X, _ = build_features(
        df,
        include_posting_engagement=False,
    )

    assert "views" not in X.columns
    assert "applies" not in X.columns
    assert "application_rate" not in X.columns
    assert "view_to_apply_ratio" not in X.columns


# ============================================================
# Build Features Tests
# ============================================================


def test_build_features():

    df = create_test_dataframe()

    X, y = build_features(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_target_is_removed_from_X():

    df = create_test_dataframe()

    X, y = build_features(df)

    assert "formatted_experience_level" not in X.columns


def test_target_is_y():

    df = create_test_dataframe()

    X, y = build_features(df)

    assert y.name == "formatted_experience_level"


def test_X_y_have_same_number_of_rows():

    df = create_test_dataframe()

    X, y = build_features(df)

    assert len(X) == len(y)


def test_build_features_contains_engineered_features():

    df = create_test_dataframe()

    X, y = build_features(df)

    expected_columns = [
        "salary_range",
        "log_salary",
        "application_rate",
        "view_to_apply_ratio",
        "combined_text",
        "text_length",
        "word_count",
        "has_python",
        "has_sql",
        "skill_count",
    ]

    for column in expected_columns:
        assert column in X.columns