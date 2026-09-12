import pandas as pd

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # =========================
    # PARSE POSTING TIME
    # =========================
    if "listed_time" not in df.columns:
        return df
    posting_time = pd.to_datetime(
        df["listed_time"],
        unit="ms",
        errors="coerce"
    )

    # =========================
    # CALENDAR FEATURES
    # =========================
    df["posting_year"] = posting_time.dt.year
    df["posting_month"] = posting_time.dt.month
    df["posting_day"] = posting_time.dt.day
    df["posting_dayofweek"] = posting_time.dt.dayofweek
    df["posting_quarter"] = posting_time.dt.quarter

    # =========================
    # TIME OF DAY
    # =========================
    df["posting_hour"] = posting_time.dt.hour

    # =========================
    # WEEKEND FEATURE
    # =========================
    df["is_weekend"] = (posting_time.dt.dayofweek >= 5).astype(int)

    return df