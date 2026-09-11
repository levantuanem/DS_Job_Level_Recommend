"""
cleaning_functions.py
Các hàm làm sạch dữ liệu tái sử dụng cho dataset LinkedIn Job Postings.
Branch: feature/data — Data Engineer / Data Analyst

Nguyên tắc:
- Không tự động xóa outlier chỉ vì nó là giá trị cực trị.
- Luôn phân biệt Data Error (sai do lỗi nhập liệu / hệ thống) và
  Genuine Extreme Value (giá trị cực trị nhưng hợp lý, vd lương chuyên gia rất cao).
- Mỗi hàm trả về (df_cleaned, report_dict) để phục vụ Data Quality Report và Logging.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về bảng thống kê missing value: số lượng + tỷ lệ % theo từng cột có khuyết thiếu.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["n_missing", "pct_missing"])
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    rep = pd.DataFrame({"n_missing": miss, "pct_missing": pct})
    return rep[rep["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def drop_missing_required(
    df: pd.DataFrame, subset: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loại bỏ các bản ghi thiếu thông tin tại các trường bắt buộc (required fields).
    """
    before = len(df)
    existing_cols = [c for c in subset if c in df.columns]
    if not existing_cols:
        return df, {
            "action": "drop_missing_required",
            "columns": subset,
            "n_before": before,
            "n_after": before,
            "n_removed": 0,
            "reason": "Không tìm thấy cột bắt buộc trong DataFrame",
        }

    df_clean = df.dropna(subset=existing_cols).copy()
    after = len(df_clean)
    report = {
        "action": "drop_missing_required",
        "columns": existing_cols,
        "n_before": before,
        "n_after": after,
        "n_removed": before - after,
        "reason": f"Các trường bắt buộc ({', '.join(existing_cols)}) không được để trống.",
    }
    return df_clean, report


def drop_duplicate_records(
    df: pd.DataFrame, subset: Optional[Union[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loại bỏ dòng trùng lặp hoàn toàn (hoặc trùng theo subset khóa chính).
    """
    before = len(df)
    df_clean = df.drop_duplicates(subset=subset).copy()
    after = len(df_clean)
    report = {
        "action": "drop_duplicates",
        "subset": subset,
        "n_before": before,
        "n_after": after,
        "n_removed": before - after,
    }
    return df_clean, report


def fix_salary_range(
    df: pd.DataFrame,
    min_col: str = "min_salary",
    max_col: str = "max_salary",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Xử lý các dòng có min_salary > max_salary (Data Error do lỗi nhập liệu đảo cột).
    Cách xử lý: Hoán đổi lại hai giá trị thay vì xóa record.
    """
    df = df.copy()
    if min_col not in df.columns or max_col not in df.columns:
        return df, {"action": "fix_salary_range_swap", "n_affected": 0, "reason": "Columns not found"}

    mask = df[min_col].notna() & df[max_col].notna() & (df[min_col] > df[max_col])
    n_affected = int(mask.sum())
    if n_affected > 0:
        df.loc[mask, [min_col, max_col]] = df.loc[mask, [max_col, min_col]].values

    report = {
        "action": "fix_salary_range_swap",
        "columns": [min_col, max_col],
        "n_affected": n_affected,
        "reason": "min_salary > max_salary là lỗi nhập liệu (đảo cột), xử lý bằng cách hoán đổi giá trị.",
    }
    return df, report


def flag_zero_negative_salary(
    df: pd.DataFrame,
    cols: Tuple[str, ...] = ("min_salary", "med_salary", "max_salary", "normalized_salary"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Đánh dấu (không xóa dòng) các giá trị lương <= 0 thành NaN.
    Lương <= 0 là giá trị không hợp lệ (Data Error), nhưng ta giữ lại record để
    bảo toàn các thông tin khác (title, description, skills).
    """
    df = df.copy()
    total_affected = 0
    affected_per_col = {}

    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].notna() & (df[col] <= 0)
        n = int(mask.sum())
        total_affected += n
        affected_per_col[col] = n
        if n > 0:
            df.loc[mask, col] = np.nan

    report = {
        "action": "flag_zero_negative_salary_as_nan",
        "columns": list(cols),
        "n_affected": total_affected,
        "details": affected_per_col,
        "reason": "Lương <= 0 là giá trị không hợp lệ (Data Error), gán NaN để không làm sai lệch phân phối và mô hình.",
    }
    return df, report


def fix_salary_units(
    df: pd.DataFrame,
    min_col: str = "min_salary",
    max_col: str = "max_salary",
    med_col: str = "med_salary",
    pay_period_col: str = "pay_period",
    norm_col: str = "normalized_salary",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Sửa lỗi lệch đơn vị thời gian trả lương (Salary Unit / Pay Period Mismatch):
    - Dòng có pay_period = 'HOURLY' nhưng lương >= 1000: Đây là nhập lương năm nhưng gắn nhãn Hourly.
      -> Đổi pay_period thành 'YEARLY', tính lại normalized_salary.
    - Dòng có pay_period = 'YEARLY' nhưng lương <= 150: Đây là nhập lương giờ nhưng gắn nhãn Yearly.
      -> Đổi pay_period thành 'HOURLY', tính lại normalized_salary (x 2080 giờ chuẩn).
    """
    df = df.copy()
    if pay_period_col not in df.columns:
        return df, {"action": "fix_salary_units", "n_affected": 0, "reason": "pay_period column missing"}

    has_norm = norm_col in df.columns

    # Chuẩn hóa tạm thời để so sánh chuẩn xác
    period_s = df[pay_period_col].astype(str).str.strip().str.upper()

    # 1. Trường hợp HOURLY nhưng lương >= 1000
    hourly_mask = (period_s == "HOURLY") & (
        (df[min_col] >= 1000)
        | (df[max_col] >= 1000)
        | (df[med_col] >= 1000)
    )
    n_hourly_fixed = int(hourly_mask.sum())
    if n_hourly_fixed > 0:
        df.loc[hourly_mask, pay_period_col] = "YEARLY"
        if has_norm:
            # Recalculate normalized salary for YEARLY
            # Base yearly = med_salary if present else average of min & max
            base = df.loc[hourly_mask, med_col]
            avg_min_max = (df.loc[hourly_mask, min_col].fillna(df.loc[hourly_mask, max_col]) +
                           df.loc[hourly_mask, max_col].fillna(df.loc[hourly_mask, min_col])) / 2.0
            base = base.fillna(avg_min_max)
            df.loc[hourly_mask, norm_col] = base

    # 2. Trường hợp YEARLY nhưng lương <= 150 (chỉ áp dụng khi lương > 0)
    yearly_mask = (period_s == "YEARLY") & (
        ((df[max_col] > 0) & (df[max_col] <= 150))
        | ((df[med_col] > 0) & (df[med_col] <= 150))
        | ((df[min_col] > 0) & (df[min_col] <= 150))
    )
    n_yearly_fixed = int(yearly_mask.sum())
    if n_yearly_fixed > 0:
        df.loc[yearly_mask, pay_period_col] = "HOURLY"
        if has_norm:
            base = df.loc[yearly_mask, med_col]
            avg_min_max = (df.loc[yearly_mask, min_col].fillna(df.loc[yearly_mask, max_col]) +
                           df.loc[yearly_mask, max_col].fillna(df.loc[yearly_mask, min_col])) / 2.0
            base = base.fillna(avg_min_max)
            df.loc[yearly_mask, norm_col] = base * 2080.0

    report = {
        "action": "fix_salary_units",
        "n_hourly_to_yearly": n_hourly_fixed,
        "n_yearly_to_hourly": n_yearly_fixed,
        "n_total_affected": n_hourly_fixed + n_yearly_fixed,
        "reason": "Lệch đơn vị thời gian (nhập lương năm vào ô Hourly hoặc ngược lại). Sửa lại pay_period và recalculate normalized_salary.",
    }
    return df, report


def standardize_categorical(
    df: pd.DataFrame, col: str, mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Chuẩn hóa category: strip khoảng trắng thừa, đồng bộ chữ hoa/thường,
    và áp dụng mapping tùy chỉnh nếu có.
    """
    df = df.copy()
    if col not in df.columns:
        return df, {"action": "standardize_categorical", "column": col, "n_affected": 0}

    before_unique = int(df[col].nunique(dropna=True))
    s = df[col].astype(str).str.strip().str.upper()
    s = s.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    if mapping:
        s = s.replace(mapping)
    df[col] = s
    after_unique = int(df[col].nunique(dropna=True))

    report = {
        "action": "standardize_categorical",
        "column": col,
        "n_unique_before": before_unique,
        "n_unique_after": after_unique,
    }
    return df, report


def clean_text_column(
    df: pd.DataFrame, col: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Làm sạch text bị lỗi: loại bỏ ký tự xuống dòng/tab thừa, chuẩn hóa nhiều khoảng
    trắng liên tiếp thành một khoảng trắng, strip đầu cuối, chuyển chuỗi rỗng thành NaN.
    Không rút gọn hay can thiệp nội dung ngữ nghĩa nghiệp vụ.
    """
    df = df.copy()
    if col not in df.columns:
        return df, {"action": "clean_text_column", "column": col, "n_empty_or_whitespace_found": 0}

    # Đếm số dòng có khoảng trắng thừa đầu cuối hoặc rỗng
    str_series = df[col].dropna().astype(str)
    n_space_issues = int(((str_series != str_series.str.strip()) | (str_series.str.strip() == "")).sum())

    cleaned = (
        df[col]
        .dropna()
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    df[col] = cleaned

    report = {
        "action": "clean_text_column",
        "column": col,
        "n_empty_or_whitespace_found": n_space_issues,
    }
    return df, report


def detect_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    """
    Phát hiện outlier bằng phương pháp IQR — CHỈ để đánh dấu / báo cáo, KHÔNG tự động xóa.
    Trả về boolean Series đánh dấu các dòng nằm ngoài [Q1 - k*IQR, Q3 + k*IQR].
    Cần xem xét thủ công từng trường hợp: Data Error hay Genuine Extreme Value.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)

    series = pd.to_numeric(df[col], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def cast_dtype(
    df: pd.DataFrame, col: str, target_dtype: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ép kiểu dữ liệu cho một cột, ghi log số lượng giá trị lỗi (coerce -> NaN).
    """
    df = df.copy()
    if col not in df.columns:
        return df, {"action": "cast_dtype", "column": col, "n_new_nan_from_coercion": 0}

    before_na = int(df[col].isna().sum())
    if target_dtype in (int, float, "float64", "int64"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    elif target_dtype == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        df[col] = df[col].astype(target_dtype, errors="ignore")

    after_na = int(df[col].isna().sum())
    report = {
        "action": "cast_dtype",
        "column": col,
        "target_dtype": str(target_dtype),
        "n_new_nan_from_coercion": after_na - before_na,
    }
    return df, report
