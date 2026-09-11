"""
tests/test_data.py
Unit tests và data validation tests cho branch feature/data.
Kiểm tra các hàm làm sạch (src/data/cleaning_functions.py) và tính toàn vẹn dữ liệu.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "src" / "data"))

from data.cleaning_functions import (
    report_missing,
    drop_missing_required,
    drop_duplicate_records,
    fix_salary_range,
    flag_zero_negative_salary,
    fix_salary_units,
    standardize_categorical,
    clean_text_column,
    detect_outliers_iqr,
    cast_dtype,
)


class TestCleaningFunctions:
    """Kiểm thử từng hàm làm sạch độc lập với mock data"""

    def test_report_missing(self):
        df = pd.DataFrame({
            "a": [1, np.nan, 3],
            "b": ["x", "y", "z"],
            "c": [np.nan, np.nan, np.nan],
        })
        rep = report_missing(df)
        assert len(rep) == 2  # 'a' and 'c'
        assert "c" in rep.index
        assert rep.loc["c", "pct_missing"] == 100.0
        assert rep.loc["a", "n_missing"] == 1

    def test_drop_duplicate_records_all(self):
        df = pd.DataFrame({
            "id": [1, 1, 2],
            "val": ["A", "A", "B"],
        })
        df_clean, report = drop_duplicate_records(df)
        assert len(df_clean) == 2
        assert report["n_removed"] == 1
        assert report["n_before"] == 3
        assert report["n_after"] == 2

    def test_drop_duplicate_records_subset(self):
        df = pd.DataFrame({
            "id": [1, 1, 2],
            "val": ["A", "different", "B"],
        })
        df_clean, report = drop_duplicate_records(df, subset=["id"])
        assert len(df_clean) == 2
        assert report["n_removed"] == 1
        assert df_clean["id"].tolist() == [1, 2]

    def test_drop_missing_required(self):
        df = pd.DataFrame({
            "id": [1, 2, np.nan, 4],
            "title": ["Data Scientist", None, "Analyst", "Engineer"],
            "desc": ["A", "B", "C", "D"],
        })
        df_clean, report = drop_missing_required(df, subset=["id", "title"])
        assert len(df_clean) == 2
        assert df_clean["id"].tolist() == [1.0, 4.0]
        assert report["n_removed"] == 2

    def test_fix_salary_range_swap(self):
        # Case: min > max due to swapped columns
        df = pd.DataFrame({
            "min_salary": [100.0, 50.0],
            "max_salary": [80.0, 70.0],
        })
        df_clean, report = fix_salary_range(df, "min_salary", "max_salary")
        assert report["n_affected"] == 1
        # Row 0 should be swapped
        assert df_clean.loc[0, "min_salary"] == 80.0
        assert df_clean.loc[0, "max_salary"] == 100.0
        # Row 1 remains unchanged
        assert df_clean.loc[1, "min_salary"] == 50.0
        assert df_clean.loc[1, "max_salary"] == 70.0

    def test_flag_zero_negative_salary(self):
        df = pd.DataFrame({
            "min_salary": [0.0, -10.0, 50000.0],
            "med_salary": [np.nan, 0.0, 60000.0],
            "max_salary": [100.0, 200.0, -5.0],
        })
        df_clean, report = flag_zero_negative_salary(
            df, cols=("min_salary", "med_salary", "max_salary")
        )
        assert pd.isna(df_clean.loc[0, "min_salary"])
        assert pd.isna(df_clean.loc[1, "min_salary"])
        assert df_clean.loc[2, "min_salary"] == 50000.0
        assert pd.isna(df_clean.loc[1, "med_salary"])
        assert pd.isna(df_clean.loc[2, "max_salary"])
        assert report["n_affected"] == 4

    def test_fix_salary_units_hourly_to_yearly(self):
        # Hourly but values are in thousands -> yearly salary wrongly tagged
        df = pd.DataFrame({
            "min_salary": [120000.0],
            "max_salary": [150000.0],
            "med_salary": [np.nan],
            "pay_period": ["HOURLY"],
            "normalized_salary": [120000.0 * 2080],  # Wrongly normalized to huge number
        })
        df_clean, report = fix_salary_units(df)
        assert df_clean.loc[0, "pay_period"] == "YEARLY"
        assert report["n_hourly_to_yearly"] == 1
        # Recalculated normalized salary should be the average (135,000)
        assert df_clean.loc[0, "normalized_salary"] == 135000.0

    def test_fix_salary_units_yearly_to_hourly(self):
        # Yearly but value is $25/hr -> hourly rate wrongly tagged as yearly
        df = pd.DataFrame({
            "min_salary": [20.0],
            "max_salary": [30.0],
            "med_salary": [np.nan],
            "pay_period": ["YEARLY"],
            "normalized_salary": [25.0],  # Wrongly normalized to $25/year
        })
        df_clean, report = fix_salary_units(df)
        assert df_clean.loc[0, "pay_period"] == "HOURLY"
        assert report["n_yearly_to_hourly"] == 1
        # Recalculated normalized salary should be 25 * 2080 = 52,000
        assert df_clean.loc[0, "normalized_salary"] == 25.0 * 2080.0

    def test_standardize_categorical(self):
        df = pd.DataFrame({
            "work_type": [" full_time ", "FULL_TIME", "part_time", None, "nan", ""]
        })
        df_clean, report = standardize_categorical(df, "work_type")
        assert df_clean.loc[0, "work_type"] == "FULL_TIME"
        assert df_clean.loc[2, "work_type"] == "PART_TIME"
        assert pd.isna(df_clean.loc[3, "work_type"])
        assert pd.isna(df_clean.loc[4, "work_type"])
        assert pd.isna(df_clean.loc[5, "work_type"])

    def test_clean_text_column(self):
        df = pd.DataFrame({
            "title": [
                "  Senior  Data   Scientist  ",
                "Machine Learning Engineer\n\t",
                "   ",
                None,
                "nan",
            ]
        })
        df_clean, report = clean_text_column(df, "title")
        assert df_clean.loc[0, "title"] == "Senior Data Scientist"
        assert df_clean.loc[1, "title"] == "Machine Learning Engineer"
        assert pd.isna(df_clean.loc[2, "title"])
        assert pd.isna(df_clean.loc[3, "title"])
        assert pd.isna(df_clean.loc[4, "title"])

    def test_detect_outliers_iqr(self):
        # Normal data [10, 11, 12, 13, 14] + extreme outlier 1000
        df = pd.DataFrame({"val": [10, 11, 12, 13, 14, 1000]})
        is_outlier = detect_outliers_iqr(df, "val", k=1.5)
        assert is_outlier.iloc[-1] == True
        assert is_outlier.iloc[0] == False

    def test_cast_dtype(self):
        df = pd.DataFrame({"num": ["1", "2", "not_a_num", np.nan]})
        df_clean, report = cast_dtype(df, "num", float)
        assert df_clean["num"].dtype == float
        assert df_clean.loc[0, "num"] == 1.0
        assert pd.isna(df_clean.loc[2, "num"])
        assert report["n_new_nan_from_coercion"] == 1


@pytest.fixture(scope="module")
def processed_dir():
    p = ROOT_DIR / "data" / "processed"
    if not p.exists() or not (p / "postings_clean.csv").exists():
        pytest.skip("data/processed/ datasets have not been generated yet.")
    return p


class TestProcessedDataIntegrity:
    """Kiểm tra tính toàn vẹn của dữ liệu sau khi chạy qua clean_data pipeline"""

    def test_postings_clean_integrity(self, processed_dir):
        postings = pd.read_csv(processed_dir / "postings_clean.csv", low_memory=False)
        assert len(postings) > 0

        # 1. job_id must be completely unique
        assert postings["job_id"].duplicated().sum() == 0

        # 2. No salary <= 0
        for col in ["min_salary", "med_salary", "max_salary", "normalized_salary"]:
            if col in postings.columns:
                assert (postings[col] <= 0).sum() == 0

        # 3. No min_salary > max_salary
        if "min_salary" in postings.columns and "max_salary" in postings.columns:
            assert (postings["min_salary"] > postings["max_salary"]).sum() == 0

        # 4. remote_allowed values must be valid (0 or 1, or NaN/1)
        if "remote_allowed" in postings.columns:
            assert postings["remote_allowed"].fillna(0).isin([0, 1, 0.0, 1.0]).all()

        # 5. Required fields must not be empty
        assert postings["title"].isnull().sum() == 0
        assert postings["description"].isnull().sum() == 0

    def test_companies_clean_integrity(self, processed_dir):
        companies = pd.read_csv(processed_dir / "companies_clean.csv")
        assert len(companies) > 0
        # company_id must be unique
        assert companies["company_id"].duplicated().sum() == 0
        # company_size must be between 1 and 7 (or NaN)
        valid_sizes = set([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        non_null_sizes = set(companies["company_size"].dropna().unique())
        assert non_null_sizes.issubset(valid_sizes)

    def test_salaries_clean_integrity(self, processed_dir):
        salaries = pd.read_csv(processed_dir / "salaries_clean.csv")
        assert len(salaries) > 0
        assert salaries["salary_id"].duplicated().sum() == 0
        for col in ["min_salary", "med_salary", "max_salary"]:
            assert (salaries[col] <= 0).sum() == 0

    def test_cleaning_log_exists(self, processed_dir):
        log_path = processed_dir / "cleaning_log.csv"
        assert log_path.exists()
        df_log = pd.read_csv(log_path)
        assert len(df_log) > 0
        assert "action" in df_log.columns
