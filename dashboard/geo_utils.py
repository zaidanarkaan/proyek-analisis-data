"""Peta kepadatan pelanggan dengan batas Brasil yang tersedia secara lokal."""

import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter


def draw_customer_density_map(ax, df) -> int:
    """Gambar satu lokasi kode pos terakhir yang valid per pelanggan unik.

    Warna menunjukkan jumlah pelanggan per sel heksagonal, bukan jumlah item.
    Nilai balik adalah jumlah pelanggan yang memiliki koordinat untuk dipetakan.
    """
    boundary_path = Path(__file__).resolve().with_name("brazil_boundary.geojson")
    with boundary_path.open(encoding="utf-8") as boundary_file:
        boundary = json.load(boundary_file)

    outline = np.asarray(boundary["geometry"]["coordinates"][0])
    ax.set_facecolor("#EAF2F8")
    ax.add_patch(Polygon(outline, facecolor="#F4F6F8", edgecolor="none", zorder=1))

    geo_customers = (
        df.dropna(subset=["customer_unique_id", "geolocation_lat", "geolocation_lng"])
        .sort_values("order_purchase_timestamp")
        .drop_duplicates(subset=["customer_unique_id"], keep="last")
    )
    if geo_customers.empty:
        ax.text(0.5, 0.5, "Tidak ada koordinat pelanggan untuk filter ini.",
                transform=ax.transAxes, ha="center", va="center")
    else:
        density = ax.hexbin(
            geo_customers["geolocation_lng"],
            geo_customers["geolocation_lat"],
            gridsize=40,
            extent=(-75, -34, -35, 6),
            mincnt=1,
            cmap="YlOrRd",
            norm=LogNorm(vmin=1),
            linewidths=0,
            zorder=2,
        )
        max_count = max(2, int(density.get_array().max()))
        density.set_clim(1, max_count)
        colorbar = ax.figure.colorbar(density, ax=ax, fraction=0.046, pad=0.04)
        ticks = sorted({1, max_count, *[10 ** n for n in range(1, int(np.log10(max_count)) + 1)]})
        colorbar.set_ticks(ticks)
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        colorbar.ax.minorticks_off()
        colorbar.set_label("Pelanggan unik per sel (skala log)", fontsize=9)

    ax.plot(outline[:, 0], outline[:, 1], color="#667085", linewidth=0.9, zorder=3)
    ax.text(-57, -7, "BRASIL", color="#667085", fontsize=10, fontweight="bold", zorder=4)
    ax.text(-38, -27, "Samudra\nAtlantik", color="#667085", fontsize=9,
            ha="center", style="italic", zorder=4)
    ax.set_xlim(-75, -33)
    ax.set_ylim(-35, 7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", fontweight="bold")
    ax.set_ylabel("Latitude", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):.0f}°W"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{abs(value):.0f}°{'S' if value < 0 else 'N'}" if value else "0°"
    ))
    ax.grid(color="#D0D5DD", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return len(geo_customers)
