import pandas as pd

from src.features.text_features import add_text_features
from src.features.skill_extraction import extract_skill_features
from src.features.temporal_features import add_temporal_features


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

            "posting_domain": [
                "linkedin.com",
                "linkedin.com",
                "linkedin.com",
            ],

            "company_name": [
                "Company A",
                "Company B",
                "Company C",
            ],

            "compensation_type": [
                "BASE_SALARY",
                "BASE_SALARY",
                "BASE_SALARY",
            ],

            "sponsored": [
                0,
                1,
                0,
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

            "listed_time": [
                1704441600000,
                1704528000000,
                1704614400000,
            ],

            "formatted_experience_level": [
                "Entry level",
                "Associate",
                "Mid-Senior level",
            ],
        }
    )


# ============================================================
# Text Feature Tests
# ============================================================


def test_add_text_features():
    df = create_test_dataframe()

    result = add_text_features(df)

    expected_columns = [
        "title_length",
        "description_length",
        "skills_length",
        "description_word_count",
        "skills_word_count",
        "combined_text",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_text_features_are_not_empty():
    df = create_test_dataframe()

    result = add_text_features(df)

    assert (result["title_length"] > 0).all()
    assert (result["description_length"] > 0).all()
    assert (result["skills_length"] > 0).all()

    assert (result["description_word_count"] > 0).all()
    assert (result["skills_word_count"] > 0).all()


def test_combined_text_contains_original_text():
    df = create_test_dataframe()

    result = add_text_features(df)

    assert "Python Developer" in result.loc[0, "combined_text"]
    assert "Python developer with SQL experience." in result.loc[0, "combined_text"]
    assert "Python, SQL" in result.loc[0, "combined_text"]


def test_text_features_handle_missing_values():
    df = create_test_dataframe()

    df.loc[0, "title"] = None
    df.loc[1, "description"] = None
    df.loc[2, "skills_desc"] = None

    result = add_text_features(df)

    assert result["title"].isna().sum() == 0
    assert result["description"].isna().sum() == 0
    assert result["skills_desc"].isna().sum() == 0

    assert result["combined_text"].notna().all()


# ============================================================
# Skill Extraction Tests
# ============================================================


def test_skill_extraction():
    df = create_test_dataframe()

    result = extract_skill_features(df)

    assert "skill_python" in result.columns
    assert "skill_sql" in result.columns
    assert "skill_aws" in result.columns
    assert "skill_machine_learning" in result.columns
    assert "skill_count" in result.columns


def test_python_skill():
    df = create_test_dataframe()

    result = extract_skill_features(df)

    assert result.loc[0, "skill_python"] == 1
    assert result.loc[2, "skill_python"] == 1


def test_sql_skill():
    df = create_test_dataframe()

    result = extract_skill_features(df)

    assert result.loc[0, "skill_sql"] == 1
    assert result.loc[1, "skill_sql"] == 1


def test_aws_skill():
    df = create_test_dataframe()

    result = extract_skill_features(df)

    assert result.loc[0, "skill_aws"] == 0
    assert result.loc[2, "skill_aws"] == 1


def test_machine_learning_skill():
    df = create_test_dataframe()

    result = extract_skill_features(df)

    assert result.loc[2, "skill_machine_learning"] == 1


def test_skill_count():
    df = create_test_dataframe()

    result = extract_skill_features(df)

    assert result.loc[0, "skill_count"] > 0
    assert result.loc[1, "skill_count"] > 0
    assert result.loc[2, "skill_count"] > 0


def test_skill_extraction_does_not_modify_original_dataframe():
    df = create_test_dataframe()

    original_columns = df.columns.tolist()

    extract_skill_features(df)

    assert df.columns.tolist() == original_columns


# ============================================================
# Temporal Feature Tests
# ============================================================


def test_temporal_features():
    start = pd.Timestamp("2024-01-05", tz="UTC")

    df = pd.DataFrame(
        {
            "listed_time": [
                int(start.timestamp() * 1000)
            ]
        }
    )

    result = add_temporal_features(df)

    assert result.loc[0, "posting_year"] == 2024
    assert result.loc[0, "posting_month"] == 1
    assert result.loc[0, "posting_day"] == 5
    assert result.loc[0, "posting_dayofweek"] == 4
    assert result.loc[0, "posting_quarter"] == 1
    assert result.loc[0, "is_weekend"] == 0


def test_temporal_features_weekend():
    start = pd.Timestamp("2024-01-06", tz="UTC")

    df = pd.DataFrame(
        {
            "listed_time": [
                int(start.timestamp() * 1000)
            ]
        }
    )

    result = add_temporal_features(df)

    assert result.loc[0, "posting_dayofweek"] == 5
    assert result.loc[0, "is_weekend"] == 1


def test_temporal_features_without_listed_time():
    df = pd.DataFrame(
        {
            "title": ["Python Developer"]
        }
    )

    result = add_temporal_features(df)

    assert list(result.columns) == list(df.columns)


# ============================================================
# Integration Tests
# ============================================================


def test_text_and_skill_features_work_together():
    df = create_test_dataframe()

    df = add_text_features(df)
    df = extract_skill_features(df)

    assert "combined_text" in df.columns
    assert "skill_python" in df.columns
    assert "skill_sql" in df.columns
    assert "skill_count" in df.columns


def test_all_feature_modules_work_together():
    df = create_test_dataframe()

    df = add_text_features(df)
    df = extract_skill_features(df)
    df = add_temporal_features(df)

    assert "combined_text" in df.columns
    assert "skill_python" in df.columns
    assert "skill_sql" in df.columns
    assert "skill_count" in df.columns

    assert "posting_year" in df.columns
    assert "posting_month" in df.columns
    assert "posting_dayofweek" in df.columns
    assert "is_weekend" in df.columns