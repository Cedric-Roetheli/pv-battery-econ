from __future__ import annotations
import os, io, base64, json, math
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt          
from matplotlib.ticker import MaxNLocator, FuncFormatter


def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_chf(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x:,.2f} CHF".replace(",", "'")


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x*100:.2f} %"


def _fmt_kwh(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x:,.0f} kWh".replace(",", "'")


def _fmt_chf_axis(x, pos) -> str:
    # Achsenticks: 12'345 (ohne CHF, ganzzahlig)
    return f"{x:,.0f}".replace(",", "'")


def _get_econ_numbers(results: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    econ = cfg.get("economics") or cfg.get("econ") or {}
    dr = econ.get("discount_rate_pct", 4.0)
    return {
        "discount_rate_pct": float(dr),
        "capex_net_chf": results.get("capex_net_chf"),
        "capex_total_chf": results.get("capex_total_chf"),
        "subsidy_upfront_chf_applied": results.get("subsidy_upfront_chf_applied"),
        "npv_chf": results.get("npv_chf"),
        "irr": results.get("irr"),
        "discounted_payback_years": results.get("discounted_payback_years"),
        "lcos_chf_per_kWh": results.get("lcos_chf_per_kWh"),
        "annual_net_benefit_chf_year1": results.get("annual_net_benefit_chf_year1"),
        "gross_benefit_chf_year1": results.get("gross_benefit_chf_year1"),
        "import_savings_chf_year1": results.get("import_savings_chf_year1"),
        "lost_feed_in_revenue_chf_year1": results.get("lost_feed_in_revenue_chf_year1"),
    }


def _discounted_cum_cf(cashflows: List[float], rate_pct: float) -> List[float]:
    r = rate_pct / 100.0
    cum = 0.0
    out = []
    for t, cf in enumerate(cashflows):
        cum += cf / ((1.0 + r) ** t)
        out.append(cum)
    return out


# -----------------------------
#   Payback-Grafiken
# -----------------------------
def _plot_grouped_payback(cf_year: List[float],
                          title: str,
                          subtitle: Optional[str],
                          payback_years: Optional[float]) -> str:
    """
    Gruppierte Balken je Jahr:
      - Balken 1: Jahres-Cashflow
      - Balken 2: Kumulierte Cashflows
    Vertikale Linie bei Payback (falls vorhanden).
    """
    years = np.arange(len(cf_year))
    cum = np.cumsum(cf_year)

    # Größer gemacht
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)

    width = 0.38
    bars_year = ax.bar(years - width/2, cf_year, width, label="Jahres-Cashflow")
    bars_cum  = ax.bar(years + width/2, cum,    width, label="Kumuliert",
                       alpha=0.75, edgecolor="black", linewidth=0.8, linestyle="--")

    # Jahr 0 hervorheben
    try:
        bars_year[0].set_hatch("//")
        bars_cum[0].set_hatch("//")
    except Exception:
        pass

    # y-Limits mit Headroom
    y_min = min(0.0, np.min(cf_year + cum.tolist()))
    y_max = max(0.0, np.max(cf_year + cum.tolist()))
    span = (y_max - y_min) or 1.0
    ax.set_ylim(y_min - 0.08*span, y_max + 0.08*span)

    # Payback-Linie & Label
    if payback_years is not None:
        ax.axvline(payback_years, color="gray", linestyle="--", linewidth=1.5)
        ax.text(
            payback_years + 0.1,
            ax.get_ylim()[0] + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            f"PB = {payback_years:.1f} a",
            rotation=90, va="bottom", ha="left", color="gray"
        )

    # Stil
    ax.set_title(title, pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#666666")

    ax.set_xlabel("Jahr")
    ax.set_ylabel("CHF")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_chf_axis))
    ax.axhline(0, color="#333333", linewidth=1.0)

    # Dezente vertikale Linien je Jahr
    for x in years:
        ax.axvline(x, color="#E6E6E6", linewidth=0.6, zorder=0)

    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _b64_png(fig)


def _fig_cashflows_and_payback(results: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """
    Diskontierte Grafik (DPB): nutzt 'cashflows_disc' (falls vorhanden), sonst 'cashflows'.
    """
    econ = _get_econ_numbers(results, cfg)
    cfs_disc: List[float] = list(results.get("cashflows_disc") or results.get("cashflows") or [])
    if not cfs_disc:
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Keine Cashflows verfügbar", ha="center", va="center")
        plt.axis("off")
        return _b64_png(fig)

    dpb = results.get("discounted_payback_years")
    subtitle = " | ".join([
        f"NPV: {_fmt_chf(econ.get('npv_chf'))}",
        f"IRR: {_fmt_pct(econ.get('irr'))}",
        f"DPB: {dpb:.1f} a" if dpb is not None else "DPB: –",
    ])
    return _plot_grouped_payback(
        cf_year=cfs_disc,
        title="Payback (diskontiert): Jahres- vs. kumulierte Cashflows",
        subtitle=subtitle,
        payback_years=dpb
    )


def _fig_cashflows_and_payback_simple(results: Dict[str, Any]) -> Optional[str]:
    """
    Undiskontierte Grafik (SPB): benötigt 'cashflows_undisc' und 'simple_payback_years'.
    """
    cfs_undisc: List[float] = list(results.get("cashflows_undisc") or [])
    if not cfs_undisc:
        return None

    spb = results.get("simple_payback_years")
    subtitle = f"SPB: {spb:.1f} a" if spb is not None else "Kein Payback innerhalb der Laufzeit"
    return _plot_grouped_payback(
        cf_year=cfs_undisc,
        title="Payback (ohne Diskontierung): Jahres- vs. kumulierte Cashflows",
        subtitle=subtitle,
        payback_years=spb
    )


# -----------------------------
#   Weitere Charts (unverändert)
# -----------------------------
def _fig_energy_bars(results: Dict[str, Any]) -> Optional[str]:
    base = results.get("timeseries_baseline") or {}
    sc0 = base.get("self_consumption_no_storage_kWh")
    imp0 = base.get("import_no_storage_kWh")
    feed0 = base.get("feed_in_no_storage_kWh")

    sc1 = results.get("self_consumption_with_storage_kWh")
    imp1 = results.get("import_with_storage_kWh")
    feed1 = results.get("feed_in_with_storage_kWh")

    if any(x is None for x in (sc0, imp0, feed0, sc1, imp1, feed1)):
        return None

    # Feste Farben pro Kategorie (gleich in beiden Säulen)
    COLORS = {
        "Import": "#1f77b4",         # blau
        "Eigenverbrauch": "#ff7f0e", # orange
        "Einspeisung": "#2ca02c",    # grün
    }

    # Daten pro Kategorie (ohne / mit Speicher)
    categories = ["Import", "Eigenverbrauch", "Einspeisung"]
    data_ohne = [imp0, sc0, feed0]
    data_mit  = [imp1, sc1, feed1]

    fig = plt.figure(figsize=(8, 5))
    x_ohne, x_mit = 0, 1
    width = 0.6

    bottom_ohne = 0.0
    bottom_mit  = 0.0

    for i, cat in enumerate(categories):
        v0 = data_ohne[i]
        v1 = data_mit[i]
        color = COLORS[cat]

        # Label nur einmal (an der linken Säule), rechte Säule ohne Legendeneintrag
        plt.bar([x_ohne], [v0], width, bottom=[bottom_ohne], color=color, label=cat)
        plt.bar([x_mit],  [v1], width, bottom=[bottom_mit],  color=color)

        bottom_ohne += v0
        bottom_mit  += v1

    plt.xticks([x_ohne, x_mit], ["Ohne Speicher", "Mit Speicher"])
    plt.ylabel("Energie [kWh/Jahr]")
    plt.title("Energiebilanz: Import / Eigenverbrauch / Einspeisung")
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)

    return _b64_png(fig)


def _fig_paid_vs_free(results: Dict[str, Any]) -> Optional[str]:
    cap = results.get("remuneration_cap_details") or {}
    p0 = cap.get("paid_feed_in_no_storage_kWh")
    f0 = cap.get("free_feed_in_no_storage_kWh")
    p1 = cap.get("paid_feed_in_with_storage_kWh")
    f1 = cap.get("free_feed_in_with_storage_kWh")
    if any(x is None for x in (p0, f0, p1, f1)):
        return None

    fig = plt.figure(figsize=(8, 5))
    idx = [0, 1]
    width = 0.35
    plt.bar([i - width/2 for i in idx], [p0, p1], width, label="Paid")
    plt.bar([i + width/2 for i in idx], [f0, f1], width, label="Free (Kappe)")
    plt.xticks(idx, ["Ohne", "Mit"])
    plt.ylabel("Einspeisung [kWh/Jahr]")
    plt.title("Bezahlte vs. freie Einspeisung (Cap-Logik)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    return _b64_png(fig)

def _fig_battery_soc(results: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    path = results.get("timeseries_csv_path")
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # SoC in % rekonstruieren (bevorzugt aus Energy-Endbestand, sonst aus SoC-Fraktion)
    cap = (cfg.get("storage") or {}).get("simulate", {}).get("capacity_kwh")
    soc_pct = None
    if "battery_energy_end_kWh" in df.columns and cap:
        soc_pct = (df["battery_energy_end_kWh"] / float(cap)) * 100.0
    elif "battery_soc_ending" in df.columns:
        soc_pct = df["battery_soc_ending"] * 100.0
    if soc_pct is None:
        return None

    ts = df["timestamp"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, soc_pct, linewidth=1.0)
    ax.set_title("Batterie State of Charge (SoC) über Zeit")
    ax.set_ylabel("SoC [%]")
    ax.set_xlabel("Zeit")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.set_ylim(0, 100)
    return _b64_png(fig)


# -----------------------------
#   HTML-Report
# -----------------------------
def generate_html_report(results: Dict[str, Any], cfg: Dict[str, Any], output_path: str) -> None:
    """Erzeugt ein eigenständiges HTML mit eingebetteten PNGs (base64)."""
    ts = cfg.get("timeseries") or cfg.get("ts") or {}
    st = cfg.get("storage") or {}
    prices = cfg.get("energy_prices") or cfg.get("prices") or {}
    econ = cfg.get("economics") or cfg.get("econ") or {}
    rem  = cfg.get("remuneration") or {}

    cons = ts.get("consumption_csv", "–")
    pv = ts.get("pv_csv", "–")

    # Figuren
    cashflow_disc_png   = _fig_cashflows_and_payback(results, cfg)
    cashflow_simple_png = _fig_cashflows_and_payback_simple(results)
    energy_png = _fig_energy_bars(results)
    soc_png    = _fig_battery_soc(results, cfg)

    # --- Werte für die zwei neuen Karten ---
    base = results.get("timeseries_baseline") or {}
    L    = base.get("load_total_kWh")
    P    = base.get("pv_total_kWh")

    sc0  = base.get("self_consumption_no_storage_kWh")
    imp0 = base.get("import_no_storage_kWh")
    feed0= base.get("feed_in_no_storage_kWh")

    sc1  = results.get("self_consumption_with_storage_kWh")
    imp1 = results.get("import_with_storage_kWh")
    feed1= results.get("feed_in_with_storage_kWh")

    evq0 = (sc0 / P) if (P and sc0 is not None and P > 0) else None
    evq1 = results.get("evq_with_storage")
    if evq1 is None and sc1 is not None and P:
        evq1 = sc1 / P


    # Infos & KPIs
    econ_nums = _get_econ_numbers(results, cfg)

    # Kleinere Helfer
    def bn(x: str) -> str:
        return os.path.basename(x) if isinstance(x, str) else str(x)

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # HTML
    html = f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>PV-Batterie Auswertung</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #111; }}
  h1 {{ font-size: 28px; margin-bottom: 0; }}
  .muted {{ color:#666; font-size: 13px; }}
  h2 {{ margin-top: 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  /* Neu: 2 breite Spalten für eigene Payback-Zeile */
  .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); }}
  .card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.03); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #f0f0f0; }}
  .kpi {{ font-weight: 600; }}
  img {{ max-width: 100%; height: auto; border:1px solid #eee; border-radius: 8px; }}
  code, pre {{ background: #f8f8f8; border:1px solid #eee; border-radius:6px; padding: 8px; display:block; overflow-x:auto; }}
</style>
</head>
<body>
  <header>
    <h1>PV-Batterie Auswertung</h1>
    <div class="muted">Generiert: {gen_time}</div>
  </header>

  <!-- Eingaben / Preise / Ökonomie / Speicher -->
  <section class="grid">
    <div class="card">
      <h2>Eingaben</h2>
      <table>
        <tr><th>Lastgang</th><td>{bn(cons)}</td></tr>
        <tr><th>PV-Produktion</th><td>{bn(pv)}</td></tr>
        <tr><th>Zeitschritt</th><td>{ts.get("step_seconds", "–")} s</td></tr>
        <tr><th>Start</th><td>{ts.get("start_datetime", "–")}</td></tr>
        <tr><th>Timezone</th><td>{ts.get("tz", "–")}</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Preise</h2>
      <table>
        <tr><th>Import</th><td>{prices.get("import_chf_per_kWh", prices.get("import_price_chf_per_kWh", "–"))} CHF/kWh</td></tr>
        <tr><th>Netz</th><td>{prices.get("grid_chf_per_kWh", prices.get("grid_price_chf_per_kWh", "–"))} CHF/kWh</td></tr>
        <tr><th>Einspeisung</th><td>{prices.get("feed_in_chf_per_kWh", prices.get("feed_in_price_chf_per_kWh", "–"))} CHF/kWh</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Ökonomie</h2>
      <table>
        <tr><th>CAPEX Gesamt</th><td>{_fmt_chf(econ_nums["capex_total_chf"])}</td></tr>
        <tr><th>Subvention</th><td>{_fmt_chf(econ_nums["subsidy_upfront_chf_applied"])}</td></tr>
        <tr><th>CAPEX Netto</th><td>{_fmt_chf(econ_nums["capex_net_chf"])}</td></tr>
        <tr><th>Diskontsatz</th><td>{econ_nums["discount_rate_pct"]:.2f} %</td></tr>
        <tr><th>NPV</th><td>{_fmt_chf(econ_nums["npv_chf"])}</td></tr>
        <tr><th>IRR</th><td>{_fmt_pct(econ_nums["irr"])}</td></tr>
        <tr><th>Discounted Payback</th><td>{econ_nums["discounted_payback_years"] if econ_nums["discounted_payback_years"] is not None else "–"}</td></tr>
        <tr><th>LCOS</th><td>{_fmt_chf(econ_nums["lcos_chf_per_kWh"])} / kWh</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Speicher-Setup</h2>
      <table>
        <tr><th>Modus</th><td>{(st.get("mode") or "simulate")}</td></tr>
        <tr><th>Kapazität</th><td>{((st.get("simulate") or {}).get("capacity_kwh", "–"))} kWh</td></tr>
        <tr><th>Leistung (±)</th><td>{((st.get("simulate") or {}).get("p_charge_kw", "–"))} / {((st.get("simulate") or {}).get("p_discharge_kw", "–"))} kW</td></tr>
        <tr><th>η Round-trip</th><td>{((st.get("simulate") or {}).get("roundtrip_eff", "–"))}</td></tr>
        <tr><th>Cap (vergütet)</th><td>{(rem.get("paid_feed_in_cap_kW", "–"))} kW</td></tr>
      </table>
    </div>
  </section>

  <!-- Jahr 1 & Eigenverbrauch (eigene Zeile) -->
  <section class="grid">
    <div class="card">
      <h2>Jahr 1: Nutzen</h2>
      <table>
        <tr><th>Import-Ersparnis</th><td class="kpi">{_fmt_chf(results.get("import_savings_chf_year1"))}</td></tr>
        <tr><th>Verlust Einspeisevergütung</th><td class="kpi">{_fmt_chf(results.get("lost_feed_in_revenue_chf_year1"))}</td></tr>
        <tr><th>Brutto-Nutzen</th><td class="kpi">{_fmt_chf(results.get("gross_benefit_chf_year1"))}</td></tr>
        <tr><th>Netto-Nutzen</th><td class="kpi">{_fmt_chf(results.get("annual_net_benefit_chf_year1"))}</td></tr>
      </table>
      <p class="muted">„Year1“ basiert auf deinen 15-Min-Profilen. Folgejahre werden in den Kennzahlen aus Year1 abgeleitet (ggf. mit Decay).</p>
    </div>

    <div class="card">
      <h2>Eigenverbrauch</h2>
      <table>
        <tr>
          <th>Eigenverbrauchsquote (mit Speicher)</th>
          <td class="kpi">
            {_fmt_pct(results.get("evq_with_storage")) if results.get("evq_with_storage") is not None else "–"}
          </td>
        </tr>
        <tr>
          <th>Eigenverbrauchssteigerung vs. ohne</th>
          <td class="kpi">
            {_fmt_pct(results.get("evq_improvement_rel")) if results.get("evq_improvement_rel") is not None else "–"}
          </td>
        </tr>
      </table>
    </div>
      <div class="card">
    <h2>ohne Batterie</h2>
    <table>
      <tr><th>Gesamtverbrauch</th><td class="kpi">{_fmt_kwh(L)}</td></tr>
      <tr><th>PV-Produktion</th><td class="kpi">{_fmt_kwh(P)}</td></tr>
      <tr><th>Eigenverbrauch (ohne Speicher)</th><td class="kpi">{_fmt_kwh(sc0)}</td></tr>
      <tr><th>Eigenverbrauchsquote (ohne)</th><td class="kpi">{_fmt_pct(evq0)}</td></tr>
      <tr><th>Rückspeisung (ohne)</th><td class="kpi">{_fmt_kwh(feed0)}</td></tr>
      <tr><th>Netzbezug (ohne)</th><td class="kpi">{_fmt_kwh(imp0)}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>mit Batterie</h2>
    <table>
      <tr><th>Gesamtverbrauch</th><td class="kpi">{_fmt_kwh(L)}</td></tr>
      <tr><th>PV-Produktion</th><td class="kpi">{_fmt_kwh(P)}</td></tr>
      <tr><th>Eigenverbrauch (mit Speicher)</th><td class="kpi">{_fmt_kwh(sc1)}</td></tr>
      <tr><th>Eigenverbrauchsquote (mit)</th><td class="kpi">{_fmt_pct(evq1)}</td></tr>
      <tr><th>Rückspeisung (mit)</th><td class="kpi">{_fmt_kwh(feed1)}</td></tr>
      <tr><th>Netzbezug (mit)</th><td class="kpi">{_fmt_kwh(imp1)}</td></tr>
    </table>
  </div>
  </section>

  <!-- Paybacks: eigene Zeile mit 2 breiten Spalten -->
  <section class="grid grid-2">
    <div class="card">
      <h2>Payback (diskontiert)</h2>
      <img src="data:image/png;base64,{cashflow_disc_png}" alt="Discounted Payback">
    </div>

    <div class="card">
      <h2>Payback (ohne Diskontierung)</h2>
      {"<img src='data:image/png;base64," + cashflow_simple_png + "'>" if cashflow_simple_png else "<p class='muted'>Nicht verfügbar (benötigt cashflows_undisc & simple_payback_years).</p>"}
    </div>
  </section>

  <!-- Energiebilanz & SoC -->
  <section class="grid">
    <div class="card">
      <h2>Energiebilanz (ohne vs. mit Speicher)</h2>
      {"<img src='data:image/png;base64," + energy_png + "'>" if energy_png else "<p class='muted'>Nicht verfügbar.</p>"}
    </div>

  <div class="card">
    <h2>Batterie-SoC</h2>
    {"<img src='data:image/png;base64," + soc_png + "'>" if soc_png else "<p class='muted'>SoC-Daten nicht verfügbar.</p>"}
  </div>

  </section>

  <section>
    <h2>Konfiguration (YAML)</h2>
    <p class="muted">Auszug der wichtigsten Annahmen (komplette Datei im Projekt).</p>
    <pre><code>{json.dumps({
      "timeseries": cfg.get("timeseries") or cfg.get("ts"),
      "storage": cfg.get("storage"),
      "energy_prices": prices,
      "economics": econ,
      "remuneration": rem
    }, indent=2, ensure_ascii=False)}</code></pre>
  </section>

</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
