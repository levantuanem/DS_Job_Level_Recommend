"""
clean_data.py
Pipeline làm sạch toàn bộ dataset LinkedIn Job Postings.
Đầu vào: data/raw/
Đầu ra: data/processed/
Branch: feature/data — Data Engineer / Data Analyst
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

import sys

# Đảm bảo đường dẫn import hoạt động cả khi gọi từ root hoặc từ bên trong src/data
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .cleaning_functions import (
        report_missing,
        drop_missing_required,
        drop_duplicate_records,
        fix_salary_range,
        flag_zero_negative_salary,
        fix_salary_units,
        standardize_categorical,
        clean_text_column,
    )
except (ImportError, ValueError):
    from cleaning_functions import (
        report_missing,
        drop_missing_required,
        drop_duplicate_records,
        fix_salary_range,
        flag_zero_negative_salary,
        fix_salary_units,
        standardize_categorical,
        clean_text_column,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def clean_companies(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch bảng companies.csv"""
    logs = []
    logger.info("Cleaning companies...")

    # 1. Drop duplicates
    df_clean, r = drop_duplicate_records(df, subset=["company_id"])
    logs.append({**r, "table": "companies"})

    # 2. Text cleaning
    for col in ["name", "description", "state", "city", "address"]:
        if col in df_clean.columns:
            df_clean, r = clean_text_column(df_clean, col)
            logs.append({**r, "table": "companies"})

    # 3. Country categorical standardization
    if "country" in df_clean.columns:
        df_clean, r = standardize_categorical(df_clean, "country")
        logs.append({**r, "table": "companies"})

    # company_size: Giữ nguyên (1-7 là mã hợp lệ, Genuine Extreme Value)
    return df_clean, logs


def clean_salaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch bảng salaries.csv"""
    logs = []
    logger.info("Cleaning salaries...")

    # 1. Drop duplicates
    df_clean, r = drop_duplicate_records(df, subset=["salary_id"])
    logs.append({**r, "table": "salaries"})

    # 2. Standardize categories first
    for col in ["pay_period", "currency", "compensation_type"]:
        if col in df_clean.columns:
            df_clean, r = standardize_categorical(df_clean, col)
            logs.append({**r, "table": "salaries"})

    # 3. Fix inverted range
    df_clean, r = fix_salary_range(df_clean, min_col="min_salary", max_col="max_salary")
    logs.append({**r, "table": "salaries"})

    # 4. Flag <= 0
    df_clean, r = flag_zero_negative_salary(df_clean, cols=("min_salary", "med_salary", "max_salary"))
    logs.append({**r, "table": "salaries"})

    # 5. Fix units
    df_clean, r = fix_salary_units(df_clean, min_col="min_salary", max_col="max_salary", med_col="med_salary", pay_period_col="pay_period")
    logs.append({**r, "table": "salaries"})

    return df_clean, logs


def clean_postings(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch bảng chính postings.csv"""
    logs = []
    logger.info("Cleaning postings...")

    # 1. Drop duplicates by job_id
    df_clean, r = drop_duplicate_records(df, subset=["job_id"])
    logs.append({**r, "table": "postings"})

    # 2. Drop missing required fields (job_id, title, description)
    df_clean, r = drop_missing_required(df_clean, subset=["job_id", "title", "description"])
    logs.append({**r, "table": "postings"})

    # 3. Standardize categorical columns
    for col in ["work_type", "formatted_work_type", "formatted_experience_level", "pay_period", "currency", "compensation_type"]:
        if col in df_clean.columns:
            df_clean, r = standardize_categorical(df_clean, col)
            logs.append({**r, "table": "postings"})

    # 4. Flag <= 0 salary
    salary_cols = ("min_salary", "med_salary", "max_salary", "normalized_salary")
    df_clean, r = flag_zero_negative_salary(df_clean, cols=salary_cols)
    logs.append({**r, "table": "postings"})

    # 5. Fix inverted salary range
    df_clean, r = fix_salary_range(df_clean, min_col="min_salary", max_col="max_salary")
    logs.append({**r, "table": "postings"})

    # 6. Fix salary unit mismatches & recalculate normalized_salary
    df_clean, r = fix_salary_units(
        df_clean,
        min_col="min_salary",
        max_col="max_salary",
        med_col="med_salary",
        pay_period_col="pay_period",
        norm_col="normalized_salary",
    )
    logs.append({**r, "table": "postings"})

    # 7. Clean text columns
    for col in ["title", "description", "skills_desc", "company_name", "location"]:
        if col in df_clean.columns:
            df_clean, r = clean_text_column(df_clean, col)
            logs.append({**r, "table": "postings"})

    # 8. remote_allowed: chuẩn hóa NaN thành 0, 1.0 thành 1
    if "remote_allowed" in df_clean.columns:
        df_clean["remote_allowed"] = df_clean["remote_allowed"].fillna(0).astype(int)
        logs.append({
            "action": "fill_remote_allowed",
            "table": "postings",
            "column": "remote_allowed",
            "reason": "Chuyển NaN thành 0 (không phải remote hoàn toàn), 1.0 thành 1.",
        })

    return df_clean, logs


def clean_bridge_table(df: pd.DataFrame, table_name: str, key_cols: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Làm sạch các bảng quan hệ / phụ"""
    logs = []
    logger.info(f"Cleaning {table_name}...")

    df_clean, r = drop_duplicate_records(df, subset=key_cols)
    logs.append({**r, "table": table_name})

    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean, r = clean_text_column(df_clean, col)
            logs.append({**r, "table": table_name})

    return df_clean, logs


def run_pipeline(raw_dir: Path, processed_dir: Path):
    """Chạy toàn bộ pipeline làm sạch và lưu trữ kết quả"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    all_logs = []
    before_counts = {}
    after_counts = {}

    # 1. Companies
    companies_path = raw_dir / "companies.csv"
    if companies_path.exists():
        companies = pd.read_csv(companies_path)
        before_counts["companies"] = len(companies)
        companies_clean, logs = clean_companies(companies)
        after_counts["companies"] = len(companies_clean)
        companies_clean.to_csv(processed_dir / "companies_clean.csv", index=False)
        all_logs.extend(logs)

    # 2. Company Industries
    ci_path = raw_dir / "company_industries.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        before_counts["company_industries"] = len(ci)
        ci_clean, logs = clean_bridge_table(ci, "company_industries", ["company_id", "industry"])
        after_counts["company_industries"] = len(ci_clean)
        ci_clean.to_csv(processed_dir / "company_industries_clean.csv", index=False)
        all_logs.extend(logs)

    # 3. Company Specialities
    cs_path = raw_dir / "company_specialities.csv"
    if cs_path.exists():
        cs = pd.read_csv(cs_path)
        before_counts["company_specialities"] = len(cs)
        cs_clean, logs = clean_bridge_table(cs, "company_specialities", ["company_id", "speciality"])
        after_counts["company_specialities"] = len(cs_clean)
        cs_clean.to_csv(processed_dir / "company_specialities_clean.csv", index=False)
        all_logs.extend(logs)

    # 4. Job Industries
    ji_path = raw_dir / "job_industries.csv"
    if ji_path.exists():
        ji = pd.read_csv(ji_path)
        before_counts["job_industries"] = len(ji)
        ji_clean, logs = clean_bridge_table(ji, "job_industries", ["job_id", "industry_id"])
        after_counts["job_industries"] = len(ji_clean)
        ji_clean.to_csv(processed_dir / "job_industries_clean.csv", index=False)
        all_logs.extend(logs)

    # 5. Job Skills
    js_path = raw_dir / "job_skills.csv"
    if js_path.exists():
        js = pd.read_csv(js_path)
        before_counts["job_skills"] = len(js)
        js_clean, logs = clean_bridge_table(js, "job_skills", ["job_id", "skill_abr"])
        after_counts["job_skills"] = len(js_clean)
        js_clean.to_csv(processed_dir / "job_skills_clean.csv", index=False)
        all_logs.extend(logs)

    # 6. Benefits
    bf_path = raw_dir / "benefits.csv"
    if bf_path.exists():
        bf = pd.read_csv(bf_path)
        before_counts["benefits"] = len(bf)
        bf_clean, logs = clean_bridge_table(bf, "benefits", ["job_id", "type"])
        after_counts["benefits"] = len(bf_clean)
        bf_clean.to_csv(processed_dir / "benefits_clean.csv", index=False)
        all_logs.extend(logs)

    # 7. Salaries
    salaries_path = raw_dir / "salaries.csv"
    if salaries_path.exists():
        salaries = pd.read_csv(salaries_path)
        before_counts["salaries"] = len(salaries)
        salaries_clean, logs = clean_salaries(salaries)
        after_counts["salaries"] = len(salaries_clean)
        salaries_clean.to_csv(processed_dir / "salaries_clean.csv", index=False)
        all_logs.extend(logs)

    # 8. Postings (bảng chính lớn nhất)
    postings_path = raw_dir / "postings.csv"
    if postings_path.exists():
        logger.info("Loading postings.csv...")
        postings = pd.read_csv(postings_path, low_memory=False)
        before_counts["postings"] = len(postings)
        postings_clean, logs = clean_postings(postings)
        after_counts["postings"] = len(postings_clean)
        logger.info("Saving postings_clean.csv...")
        postings_clean.to_csv(processed_dir / "postings_clean.csv", index=False)
        all_logs.extend(logs)

    # Lưu cleaning log
    cleaning_log_df = pd.DataFrame(all_logs)
    cleaning_log_df.to_csv(processed_dir / "cleaning_log.csv", index=False)
    logger.info(f"Saved cleaning log with {len(cleaning_log_df)} entries to {processed_dir / 'cleaning_log.csv'}")

    # In và lưu bảng tóm tắt
    summary = pd.DataFrame({
        "Table": list(before_counts.keys()),
        "Records Before": list(before_counts.values()),
        "Records After": [after_counts.get(k, 0) for k in before_counts.keys()],
    })
    summary["Records Removed"] = summary["Records Before"] - summary["Records After"]
    summary.to_csv(processed_dir / "cleaning_summary.csv", index=False)
    logger.info(f"Saved cleaning summary to {processed_dir / 'cleaning_summary.csv'}")

    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY: BEFORE vs AFTER CLEANING")
    print("=" * 50)
    print(summary.to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw LinkedIn job posting datasets")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data directory")
    args = parser.parse_args()

    run_pipeline(Path(args.raw_dir), Path(args.processed_dir))
