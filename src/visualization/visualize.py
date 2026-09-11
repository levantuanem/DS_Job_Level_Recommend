"""
Module: src.visualization.visualize
Chứa các hàm trực quan hóa dữ liệu và EDA chuẩn hóa cho dự án DS Job Recommend.
Người thực hiện: Người 2 (Nhánh: feature/visualization)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_numerical_distribution(df: pd.DataFrame, col: str, bins: int = 60):
    """Vẽ phân phối (Histogram + KDE nếu có biến thiên) và Boxplot cho một cột số."""
    if col not in df.columns:
        print(f"Cột {col} không tồn tại trong DataFrame.")
        return

    data = df[col].dropna()
    if len(data) == 0:
        print(f"Cột {col}: không có dữ liệu để vẽ.")
        return

    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((data < lower) | (data > upper)).sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram + KDE
    axes[0].hist(
        data,
        bins=bins,
        color="steelblue",
        edgecolor="white",
        alpha=0.7,
        density=True,
    )
    # Chỉ vẽ KDE khi dữ liệu có độ biến thiên (tránh lỗi ma trận kỳ dị)
    if data.nunique() > 1 and data.std() > 0:
        data.plot.kde(ax=axes[0], color="crimson", linewidth=2)

    axes[0].axvline(
        data.mean(),
        color="orange",
        linestyle="--",
        label=f"Mean = {data.mean():,.0f}",
    )
    axes[0].axvline(
        data.median(),
        color="green",
        linestyle="--",
        label=f"Median = {data.median():,.0f}",
    )
    axes[0].set_title(f"Phân phối — {col}", fontsize=13, fontweight="bold")
    axes[0].legend()

    skew = data.skew() if len(data) > 2 else 0.0
    txt = (
        "Lệch phải"
        if skew > 0.5
        else ("Lệch trái" if skew < -0.5 else "Đối xứng")
    )
    axes[0].text(
        0.97,
        0.95,
        f"Skew = {skew:.2f} ({txt})",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="lightyellow", alpha=0.9, boxstyle="round"),
    )

    # Boxplot
    axes[1].boxplot(
        data,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker=".", color="gray", alpha=0.3, markersize=3),
    )
    axes[1].set_title(f"Boxplot — {col}", fontsize=13, fontweight="bold")
    pct_out = (n_out / len(data) * 100) if len(data) > 0 else 0.0
    axes[1].text(
        0.97,
        0.85,
        f"Q1 = {Q1:,.0f}\nQ3 = {Q3:,.0f}\nIQR = {IQR:,.0f}\nOutlier = {n_out:,} ({pct_out:.1f}%)",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="lightyellow", alpha=0.9, boxstyle="round"),
    )

    plt.suptitle(
        f"Phân tích đơn biến: {col} (n={len(data):,})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_categorical_distribution(
    df: pd.DataFrame, col: str, top_n: int = 15
):
    """Vẽ biểu đồ số lượng và tỷ lệ % cho cột phân loại (xử lý an toàn nhãn NaN)."""
    if col not in df.columns:
        print(f"Cột {col} không tồn tại trong DataFrame.")
        return

    vc = df[col].value_counts(dropna=False).head(top_n)
    if len(df) == 0 or len(vc) == 0:
        print(f"Cột {col}: không có dữ liệu để vẽ.")
        return

    pct = vc / len(df) * 100
    x_labels = [str(x) if pd.notna(x) else "(Trống)" for x in vc.index]
    colors = sns.color_palette("Set2", len(vc))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Số lượng
    bars = axes[0].bar(x_labels, vc.values, color=colors, edgecolor="white")
    for bar, p in zip(bars, pct.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vc.values.max() * 0.01,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_title(f"Số lượng — {col}", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=40)

    # Tỷ lệ %
    bars2 = axes[1].bar(x_labels, pct.values, color=colors, edgecolor="white")
    for bar, p in zip(bars2, pct.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pct.values.max() * 0.01,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylabel("Tỷ lệ (%)")
    axes[1].set_title(
        f"Tỷ lệ phần trăm — {col}", fontsize=12, fontweight="bold"
    )
    axes[1].tick_params(axis="x", rotation=40)

    plt.suptitle(
        f"Phân tích đơn biến: {col}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame, num_cols: list, method: str = "pearson"
):
    """Vẽ Heatmap ma trận tương quan nửa dưới cho danh sách cột số."""
    valid_cols = [c for c in num_cols if c in df.columns]
    if len(valid_cols) < 2:
        print("Cần ít nhất 2 cột số hợp lệ để tính tương quan.")
        return

    corr = df[valid_cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title(
        f"Ma trận tương quan {method.title()} giữa các biến số",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_feature_vs_target(
    df: pd.DataFrame, num_col: str, target_col: str
):
    """Vẽ Boxplot và Violin plot so sánh phân phối cột số theo biến mục tiêu."""
    if num_col not in df.columns or target_col not in df.columns:
        print(f"Cột {num_col} hoặc {target_col} không tồn tại.")
        return

    plot_df = df[[target_col, num_col]].dropna()
    if len(plot_df) == 0:
        return
    order = (
        plot_df.groupby(target_col)[num_col]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.boxplot(
        data=plot_df,
        x=target_col,
        y=num_col,
        order=order,
        palette="Set2",
        hue=target_col,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title(
        f"{num_col} theo {target_col} — Boxplot",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].tick_params(axis="x", rotation=30)

    sns.violinplot(
        data=plot_df,
        x=target_col,
        y=num_col,
        order=order,
        palette="Set2",
        hue=target_col,
        legend=False,
        ax=axes[1],
        inner="quartile",
        cut=0,
    )
    axes[1].set_title(
        f"{num_col} theo {target_col} — Violin",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle(
        f"Quan hệ: {num_col} vs {target_col}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_class_imbalance(df: pd.DataFrame, target_col: str):
    """Vẽ biểu đồ cột và tròn thể hiện mất cân bằng nhãn mục tiêu."""
    if target_col not in df.columns:
        print(f"Cột {target_col} không tồn tại.")
        return

    counts = df[target_col].value_counts(dropna=False)
    if len(counts) == 0:
        return
    pcts = df[target_col].value_counts(normalize=True, dropna=False) * 100
    labels = [
        str(x) if pd.notna(x) else "(Trống / NaN)" for x in counts.index
    ]
    colors = sns.color_palette("Set2", len(counts))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cột
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white")
    for bar, cnt, pct in zip(bars, counts.values, pcts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.values.max() * 0.01,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_title(
        f"Phân phối số lượng — {target_col}", fontsize=13, fontweight="bold"
    )
    axes[0].tick_params(axis="x", rotation=30)

    # Tròn
    axes[1].pie(
        counts.values,
        labels=[f"{l}\n({p:.1f}%)" for l, p in zip(labels, pcts.values)],
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title(
        f"Tỷ lệ phần trăm — {target_col}", fontsize=13, fontweight="bold"
    )

    plt.suptitle(
        f"Phân tích mất cân bằng lớp — {target_col}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_outlier_log_transform(df: pd.DataFrame, col: str):
    """So sánh phân phối gốc và sau khi áp dụng Log(1+x)."""
    if col not in df.columns:
        print(f"Cột {col} không tồn tại.")
        return

    data = df[col].dropna()
    if len(data) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Boxplot
    axes[0].boxplot(
        data,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[0].set_title(
        f"Boxplot gốc: {col}", fontsize=11, fontweight="bold"
    )

    # Gốc
    axes[1].hist(
        data,
        bins=60,
        color="steelblue",
        alpha=0.7,
        edgecolor="white",
        density=True,
    )
    skew_orig = data.skew() if len(data) > 2 else 0.0
    axes[1].set_title(
        f"Phân phối gốc (Skew={skew_orig:.2f})",
        fontsize=11,
        fontweight="bold",
    )

    # Log
    pos_data = data[data > 0]
    if len(pos_data) > 0:
        log_d = np.log1p(pos_data)
        axes[2].hist(
            log_d,
            bins=60,
            color="seagreen",
            alpha=0.7,
            edgecolor="white",
            density=True,
        )
        skew_log = log_d.skew() if len(log_d) > 2 else 0.0
        axes[2].set_title(
            f"Sau Log(1+x) (Skew={skew_log:.2f})",
            fontsize=11,
            fontweight="bold",
        )
    else:
        axes[2].set_title(f"Không có dữ liệu > 0 để tính log", fontsize=11)

    axes[2].set_xlabel("log(1 + giá trị)")

    plt.suptitle(
        f"Ngoại lệ & Biến đổi Log — {col}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()