"""
models/resource_allocator.py
Translates ward risk scores into actionable BBMP resource deployment recommendations.
Outputs deployment_plan.csv and deployment_summary.png.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Resource allocation rules ──────────────────────────────────────────────────
ALLOCATION_RULES = {
    "CRITICAL": {
        "pump_trucks":       3,
        "sandbag_pallets":   50,
        "inspection_teams":  2,
        "sms_alerts":        5000,
        "deadline":          "Before June 1",
        "primary_action":    "Emergency drain desilt + pump pre-positioning",
        "secondary_action":  "24hr water level monitoring at 3 low points",
        "colour":            "#C00000",
    },
    "HIGH": {
        "pump_trucks":       2,
        "sandbag_pallets":   30,
        "inspection_teams":  1,
        "sms_alerts":        3000,
        "deadline":          "Before June 15",
        "primary_action":    "Priority drain inspection + standby pump allocation",
        "secondary_action":  "Identify 2 emergency water discharge points",
        "colour":            "#ED7D31",
    },
    "MODERATE": {
        "pump_trucks":       0,
        "sandbag_pallets":   10,
        "inspection_teams":  1,
        "sms_alerts":        1000,
        "deadline":          "Before June 30",
        "primary_action":    "Scheduled SWD drain inspection",
        "secondary_action":  "Clear inlet grates + check pump stations",
        "colour":            "#FFC000",
    },
    "LOW": {
        "pump_trucks":       0,
        "sandbag_pallets":   0,
        "inspection_teams":  0,
        "sms_alerts":        0,
        "deadline":          "Standard monsoon protocol",
        "primary_action":    "Routine pre-monsoon checklist",
        "secondary_action":  "No urgent action required",
        "colour":            "#70AD47",
    },
}

BLUE_DARK  = "#1A3A5C"
GREY_BG    = "#F5F7FA"


def allocate_resources(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add resource allocation columns to ward dataframe.
    Returns sorted deployment plan.
    """
    df = ward_df.copy().sort_values("risk_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    for col in ["pump_trucks","sandbag_pallets","inspection_teams","sms_alerts",
                "deadline","primary_action","secondary_action"]:
        df[col] = df["risk_label"].map(lambda lbl: ALLOCATION_RULES[lbl][col])

    df["priority_rank"] = df.index

    # Citywide totals
    print(f"\n[allocator] === BBMP Pre-Monsoon 2026 Deployment Plan ===")
    print(f"  Total pump trucks:       {df['pump_trucks'].sum()}")
    print(f"  Total sandbag pallets:   {df['sandbag_pallets'].sum()}")
    print(f"  Total inspection teams:  {df['inspection_teams'].sum()}")
    print(f"  Total SMS alerts:        {df['sms_alerts'].sum():,}")
    print(f"  CRITICAL wards:          {(df['risk_label']=='CRITICAL').sum()}")
    print(f"  HIGH wards:              {(df['risk_label']=='HIGH').sum()}")

    return df


def save_deployment_csv(df: pd.DataFrame, out_path: str) -> None:
    cols = ["priority_rank","ward_name","risk_score","risk_label",
            "pump_trucks","sandbag_pallets","inspection_teams","sms_alerts",
            "deadline","primary_action","secondary_action"]
    df.reset_index()[cols].to_csv(out_path, index=False)
    print(f"[allocator] CSV saved → {out_path}")


def save_deployment_visual(df: pd.DataFrame, out_path: str) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(GREY_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # pump trucks
    ax2 = fig.add_subplot(gs[0, 1])   # inspection teams
    ax3 = fig.add_subplot(gs[1, :])   # deployment table

    tiers  = ["CRITICAL","HIGH","MODERATE","LOW"]
    colors = [ALLOCATION_RULES[t]["colour"] for t in tiers]
    counts = [int((df["risk_label"] == t).sum()) for t in tiers]
    pumps  = [ALLOCATION_RULES[t]["pump_trucks"] * c for t, c in zip(tiers, counts)]
    teams  = [ALLOCATION_RULES[t]["inspection_teams"] * c for t, c in zip(tiers, counts)]

    # Pump trucks bar chart
    ax1.bar(tiers, pumps, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax1.set_title("Pump Trucks Required\nby Risk Tier", fontsize=11,
                  fontweight="bold", color=BLUE_DARK)
    ax1.set_ylabel("Number of Pump Trucks", fontsize=10, color=BLUE_DARK)
    ax1.set_facecolor("white")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate(pumps):
        ax1.text(i, v + 0.3, str(v), ha="center", fontsize=10,
                 fontweight="bold", color=BLUE_DARK)

    # Inspection teams
    ax2.bar(tiers, teams, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax2.set_title("Inspection Teams Required\nby Risk Tier", fontsize=11,
                  fontweight="bold", color=BLUE_DARK)
    ax2.set_ylabel("Number of Teams", fontsize=10, color=BLUE_DARK)
    ax2.set_facecolor("white")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate(teams):
        ax2.text(i, v + 0.1, str(v), ha="center", fontsize=10,
                 fontweight="bold", color=BLUE_DARK)

    # Deployment table — top 15 wards
    ax3.axis("off")
    ax3.set_facecolor(GREY_BG)
    top15 = df.head(15).reset_index()

    col_labels = ["Rank","Ward","Score","Tier","Pumps","Teams","Deadline","Primary Action"]
    col_keys   = ["rank","ward_name","risk_score","risk_label","pump_trucks",
                  "inspection_teams","deadline","primary_action"]
    col_w      = [0.04, 0.16, 0.05, 0.08, 0.05, 0.05, 0.12, 0.40]
    xs = [0.01]
    for cw in col_w[:-1]:
        xs.append(xs[-1] + cw)

    # Header
    for cl, x, cw in zip(col_labels, xs, col_w):
        ax3.text(x + cw/2, 0.97, cl, ha="center", va="top",
                 transform=ax3.transAxes, fontsize=8.5, fontweight="bold",
                 color="white",
                 bbox=dict(boxstyle="square,pad=0.3", fc=BLUE_DARK, ec="none"))

    for ri, row in top15.iterrows():
        y = 0.97 - 0.063 * (ri + 1)
        tier_col = ALLOCATION_RULES.get(row["risk_label"], {}).get("colour", "#888")
        bg = tier_col + "22"   # 13% alpha hex
        ax3.add_patch(mpatches.FancyBboxPatch(
            (0.005, y - 0.03), 0.99, 0.057,
            boxstyle="square,pad=0", facecolor=bg, edgecolor="#DDDDDD",
            linewidth=0.4, transform=ax3.transAxes, zorder=1))

        for key, x, cw in zip(col_keys, xs, col_w):
            val = str(row[key]) if key in row else ""
            col = BLUE_DARK
            fw  = "normal"
            if key == "risk_label":
                col = ALLOCATION_RULES.get(val, {}).get("colour", "#888")
                fw  = "bold"
            ax3.text(x + cw/2, y, val, ha="center", va="center",
                     transform=ax3.transAxes, fontsize=8, color=col, fontweight=fw)

    ax3.set_title("Top-15 Priority Wards — Resource Deployment Plan",
                  fontsize=12, fontweight="bold", color=BLUE_DARK, pad=8)

    # Summary banner
    total_p = df["pump_trucks"].sum()
    total_t = df["inspection_teams"].sum()
    total_s = df["sms_alerts"].sum()
    fig.text(0.5, 0.01,
             f"Citywide totals: {total_p} pump trucks  ·  {total_t} inspection teams  ·  "
             f"{total_s:,} SMS alerts  ·  NDMA Urban Flood Guidelines 2010",
             ha="center", fontsize=9, color="grey", style="italic")

    fig.suptitle("BBMP Pre-Monsoon 2026 — Resource Deployment Plan\n"
                 "HydraGIS NDMA Risk-Driven Allocation",
                 fontsize=14, fontweight="bold", color=BLUE_DARK)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=GREY_BG)
    plt.close()
    print(f"[allocator] Visual saved → {out_path}")


if __name__ == "__main__":
    from pipeline.ward_pipeline import build_ward_scores
    import os

    df  = build_ward_scores()
    dep = allocate_resources(df)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../data")
    os.makedirs(out_dir, exist_ok=True)

    save_deployment_csv(dep, os.path.join(out_dir, "deployment_plan.csv"))
    save_deployment_visual(dep, os.path.join(out_dir, "deployment_summary.png"))
