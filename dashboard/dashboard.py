import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
)

sns.set_style("whitegrid")
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


@st.cache_data
def load_data() -> pd.DataFrame:
    """Memuat dan memproses dataset utama dashboard."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "main_data.csv"
    df = pd.read_csv(data_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["is_late"] = df["is_late"].astype(bool)
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung metrik RFM dan segmentasi Manual Grouping/Binning tanpa ML."""
    if df.empty:
        return pd.DataFrame()

    snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_unique_id", as_index=False)
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("total_item_value", "sum"),
        )
    )
    rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days

    # Gunakan rank(method="first") agar qcut tetap stabil pada subset filter kecil
    if len(rfm) >= 4:
        rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    else:
        rfm["r_score"] = 2
        rfm["m_score"] = 2

    rfm["f_score"] = np.where(rfm["frequency"] >= 3, 4, np.where(rfm["frequency"] == 2, 3, 1))

    def assign_segment(row):
        if row["frequency"] >= 2 and row["r_score"] >= 3 and row["m_score"] >= 3:
            return "Champions"
        elif row["frequency"] >= 2:
            return "Loyal Repeaters"
        elif row["r_score"] >= 3 and row["m_score"] >= 3:
            return "Potential High-Value"
        elif row["r_score"] <= 2 and row["m_score"] >= 3:
            return "At Risk High-Value"
        elif row["r_score"] >= 3:
            return "Recent Customers"
        else:
            return "Hibernating"

    rfm["customer_segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


all_df = load_data()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🛒 Filter Analisis")
    st.caption("Sesuaikan rentang waktu, wilayah, dan kategori produk.")

    min_date = all_df["order_purchase_timestamp"].min().date()
    max_date = all_df["order_purchase_timestamp"].max().date()

    date_range = st.date_input(
        "Rentang Tanggal Pembelian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    states = sorted(all_df["customer_state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        "Negara Bagian (State)",
        options=states,
        default=[],
        placeholder="Semua Negara Bagian",
    )

    categories = sorted(all_df["product_category_name_english"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "Kategori Produk",
        options=categories,
        default=[],
        placeholder="Semua Kategori Produk",
    )

# Filter dataframe
filtered_df = all_df[
    (all_df["order_purchase_timestamp"].dt.date >= start_date) &
    (all_df["order_purchase_timestamp"].dt.date <= end_date)
].copy()

if selected_states:
    filtered_df = filtered_df[filtered_df["customer_state"].isin(selected_states)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["product_category_name_english"].isin(selected_categories)]

# ==================== HEADER & KPI ====================
st.title("📊 E-Commerce Public Dataset Analytics Dashboard")
st.markdown(
    "Dashboard interaktif untuk memantau **tren penjualan**, **kepuasan pelanggan berdasarkan ketepatan pengiriman**, "
    "**segmentasi pelanggan (RFM)**, dan **sebaran geografis**."
)

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan kombinasi filter yang dipilih. Silakan ubah filter di sidebar.")
    st.stop()

order_summary_df = filtered_df.groupby("order_id", as_index=False).agg(
    is_late=("is_late", "first"),
    review_score=("review_score", "first"),
)

total_orders = filtered_df["order_id"].nunique()
total_revenue = filtered_df["price"].sum()
avg_review = order_summary_df["review_score"].mean()
on_time_rate = (1 - order_summary_df["is_late"].mean()) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pesanan Unik", f"{total_orders:,}")
col2.metric("Total Revenue (Produk)", f"R$ {total_revenue:,.0f}")
col3.metric("Rata-rata Skor Ulasan", f"{avg_review:.2f} / 5.00")
col4.metric("Ketepatan Pengiriman", f"{on_time_rate:.1f}%")

st.divider()

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Penjualan & Kategori",
    "🚚 Pengiriman & Ulasan",
    "👥 RFM & Segmentasi",
    "🗺️ Analisis Geografis",
])

# --- TAB 1: PENJUALAN & KATEGORI ---
with tab1:
    st.subheader("Tren Pesanan Bulanan & Kategori Produk Teratas")

    filtered_df["year_month"] = filtered_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly_df = (
        filtered_df.groupby("year_month", as_index=False)
        .agg(total_orders=("order_id", "nunique"), total_revenue=("price", "sum"))
        .sort_values("year_month")
    )

    cat_df = (
        filtered_df.groupby("product_category_name_english", as_index=False)
        .agg(total_orders=("order_id", "nunique"), total_revenue=("price", "sum"))
        .sort_values("total_revenue", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        x = range(len(monthly_df))
        ax1.bar(x, monthly_df["total_orders"], color="#B0C4DE", width=0.65)
        if not monthly_df.empty:
            peak_pos = int(monthly_df["total_orders"].values.argmax())
            ax1.patches[peak_pos].set_color("#1F4E78")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(monthly_df["year_month"], rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Jumlah Pesanan", fontweight="bold")
        ax1.set_title("Tren Pesanan & Revenue Bulanan", fontweight="bold")
        ax1.spines["top"].set_visible(False)

        ax2 = ax1.twinx()
        ax2.plot(list(x), monthly_df["total_revenue"] / 1e3, color="#D95F02", marker="o", linewidth=2)
        ax2.set_ylabel("Revenue (Ribu BRL)", fontweight="bold", color="#D95F02")
        ax2.spines["top"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        top10_cat = cat_df.head(10).sort_values("total_revenue", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors_cat = ["#B0C4DE"] * max(0, len(top10_cat) - 1) + ["#1F4E78"]
        ax.barh(top10_cat["product_category_name_english"], top10_cat["total_revenue"] / 1e3, color=colors_cat[:len(top10_cat)])
        ax.set_xlabel("Total Revenue (Ribu BRL)", fontweight="bold")
        ax.set_title("Top 10 Kategori Produk berdasarkan Revenue", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not top10_cat.empty:
            ax.set_xlim(0, top10_cat["total_revenue"].max() / 1e3 * 1.18)
            for i, val in enumerate(top10_cat["total_revenue"] / 1e3):
                ax.text(val + (top10_cat["total_revenue"].max() / 1e3 * 0.02), i, f"R$ {val:,.0f}k", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --- TAB 2: PENGIRIMAN & ULASAN ---
with tab2:
    st.subheader("Pengaruh Ketepatan Pengiriman terhadap Skor Ulasan")

    deliv_clean = order_summary_df.dropna(subset=["review_score"]).copy()
    deliv_clean["delivery_status"] = np.where(
        deliv_clean["is_late"], "Terlambat (Late)", "Tepat Waktu / Lebih Awal"
    )
    deliv_clean["is_negative"] = deliv_clean["review_score"] <= 2

    deliv_summary = (
        deliv_clean.groupby("delivery_status", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            avg_score=("review_score", "mean"),
            neg_pct=("is_negative", lambda s: s.mean() * 100),
        )
        .sort_values("avg_score", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors_d = ["#2E8B57" if "Tepat" in s else "#C0392B" for s in deliv_summary["delivery_status"]]
        bars = ax.bar(deliv_summary["delivery_status"], deliv_summary["avg_score"], color=colors_d, width=0.5)
        ax.set_ylim(0, 5.3)
        ax.set_ylabel("Rata-rata Review Score (1–5)", fontweight="bold")
        ax.set_title("Rata-rata Skor Ulasan: Tepat Waktu vs Terlambat", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, (_, row) in zip(bars, deliv_summary.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{row['avg_score']:.2f}\n({row['neg_pct']:.1f}% skor 1–2)",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        dist_df = (
            deliv_clean.assign(score_int=deliv_clean["review_score"].round().astype(int))
            .groupby(["delivery_status", "score_int"])["order_id"]
            .count()
            .unstack(fill_value=0)
        )
        for s in range(1, 6):
            if s not in dist_df.columns:
                dist_df[s] = 0
        dist_df = dist_df[[1, 2, 3, 4, 5]]
        dist_pct = dist_df.div(dist_df.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x_scores = np.arange(1, 6)
        w = 0.35
        if "Tepat Waktu / Lebih Awal" in dist_pct.index:
            ax.bar(x_scores - w/2, dist_pct.loc["Tepat Waktu / Lebih Awal"], width=w, color="#2E8B57", label="Tepat Waktu / Lebih Awal")
        if "Terlambat (Late)" in dist_pct.index:
            ax.bar(x_scores + w/2, dist_pct.loc["Terlambat (Late)"], width=w, color="#C0392B", label="Terlambat (Late)")
        ax.set_xticks(x_scores)
        ax.set_xticklabels([f"Skor {s}" for s in x_scores])
        ax.set_ylabel("Persentase Pesanan (%)", fontweight="bold")
        ax.set_title("Distribusi Skor Ulasan Berdasarkan Ketepatan Pengiriman", fontweight="bold")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --- TAB 3: RFM & SEGMENTASI ---
with tab3:
    st.subheader("RFM Analysis & Segmentasi Pelanggan (Manual Grouping)")

    rfm_df = compute_rfm(filtered_df)
    seg_summary = (
        rfm_df.groupby("customer_segment", as_index=False)
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
    )
    seg_summary["customer_pct"] = seg_summary["customer_count"] / seg_summary["customer_count"].sum() * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Rata-rata Recency (Hari)", f"{rfm_df['recency'].mean():.1f}")
    m2.metric("Rata-rata Frequency (Order)", f"{rfm_df['frequency'].mean():.2f}")
    m3.metric("Rata-rata Monetary (BRL)", f"R$ {rfm_df['monetary'].mean():,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        seg_plot = seg_summary.sort_values("customer_count", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors_seg = ["#B0C4DE"] * max(0, len(seg_plot) - 1) + ["#1F4E78"]
        ax.barh(seg_plot["customer_segment"], seg_plot["customer_count"], color=colors_seg[:len(seg_plot)])
        ax.set_xlabel("Jumlah Pelanggan Unik", fontweight="bold")
        ax.set_title("Distribusi Segmen Pelanggan", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, seg_plot["customer_count"].max() * 1.24)
        for i, (_, row) in enumerate(seg_plot.iterrows()):
            ax.text(row["customer_count"] + seg_plot["customer_count"].max() * 0.02, i,
                    f"{row['customer_count']:,} ({row['customer_pct']:.1f}%)", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        seg_mon = seg_summary.sort_values("avg_monetary", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        mon_colors = [
            "#1F4E78" if s in ["Champions", "At Risk High-Value", "Potential High-Value"] else "#B0C4DE"
            for s in seg_mon["customer_segment"]
        ]
        ax.barh(seg_mon["customer_segment"], seg_mon["avg_monetary"], color=mon_colors)
        ax.set_xlabel("Rata-rata Monetary (BRL)", fontweight="bold")
        ax.set_title("Rata-rata Nilai Belanja per Segmen", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, seg_mon["avg_monetary"].max() * 1.2)
        for i, val in enumerate(seg_mon["avg_monetary"]):
            ax.text(val + seg_mon["avg_monetary"].max() * 0.02, i, f"R$ {val:,.1f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# --- TAB 4: GEOSPATIAL ---
with tab4:
    st.subheader("Geospatial Analysis: Sebaran Pelanggan & Revenue per Negara Bagian")

    state_summary = (
        filtered_df.groupby("customer_state", as_index=False)
        .agg(
            total_customers=("customer_unique_id", "nunique"),
            total_orders=("order_id", "nunique"),
            total_revenue=("total_item_value", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    state_summary["share_pct"] = state_summary["total_revenue"] / state_summary["total_revenue"].sum() * 100

    c1, c2 = st.columns(2)
    with c1:
        geo_sample = (
            filtered_df[["customer_unique_id", "geolocation_lat", "geolocation_lng"]]
            .dropna()
            .drop_duplicates(subset=["customer_unique_id"])
        )
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.scatter(geo_sample["geolocation_lng"], geo_sample["geolocation_lat"], s=2.5, alpha=0.25, color="#1F4E78")
        ax.set_xlabel("Longitude", fontweight="bold")
        ax.set_ylabel("Latitude", fontweight="bold")
        ax.set_title("Peta Sebaran Geografis Pelanggan di Brasil", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        top10_states = state_summary.head(10).sort_values("total_revenue", ascending=True)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        st_colors = ["#B0C4DE"] * max(0, len(top10_states) - 1) + ["#1F4E78"]
        ax.barh(top10_states["customer_state"], top10_states["total_revenue"] / 1e6, color=st_colors[:len(top10_states)])
        ax.set_xlabel("Total Revenue (Juta BRL)", fontweight="bold")
        ax.set_title("Top 10 Negara Bagian berdasarkan Revenue", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not top10_states.empty:
            ax.set_xlim(0, top10_states["total_revenue"].max() / 1e6 * 1.24)
            for i, (_, row) in enumerate(top10_states.iterrows()):
                ax.text(
                    row["total_revenue"] / 1e6 + (top10_states["total_revenue"].max() / 1e6 * 0.02),
                    i,
                    f"R$ {row['total_revenue']/1e6:.2f}M ({row['share_pct']:.1f}%)",
                    va="center", fontsize=8
                )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.caption("© Submission Proyek Analisis Data dengan Python — E-Commerce Public Dataset")

