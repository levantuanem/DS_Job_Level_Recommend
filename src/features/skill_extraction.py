import re
import pandas as pd

# =========================
# SKILL VOCABULARY
# =========================
SKILL_VOCABULARY = [
    # Programming
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "r",
    "sql",

    # Data Science / ML
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data analysis",
    "natural language processing",
    "computer vision",

    # ML frameworks
    "pytorch",
    "tensorflow",
    "keras",
    "scikit-learn",

    # Data
    "pandas",
    "numpy",
    "spark",
    "hadoop",

    # Cloud / DevOps
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",

    # Databases
    "mysql",
    "postgresql",
    "mongodb",
    "oracle",

    # Visualization / BI
    "tableau",
    "power bi",
    "excel",

    # Software / Web
    "git",
    "github",
    "linux",
    "html",
    "css",
    "react",
    "node.js"
]

# =========================
# TEXT PREPARATION
# =========================
def _combine_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
        + " "
        + df["skills_desc"].fillna("").astype(str)
    ).str.lower()

# =========================
# SKILL EXTRACTION
# =========================

def extract_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = _combine_text(df)
    skill_columns = []
    for skill in SKILL_VOCABULARY:
        column_name = (
            "skill_"
            + re.sub(r"[^a-zA-Z0-9]+", "_", skill)
            .strip("_")
            .lower()
        )

        # Avoid duplicate feature names
        if column_name in skill_columns:
            continue

        df[column_name] = (
            text.str.contains(
                re.escape(skill),
                regex=True,
                na=False,
            )
            .astype(int)
        )

        skill_columns.append(column_name)

    # Total number of detected skills
    df["skill_count"] = df[skill_columns].sum(axis=1)

    return df