from __future__ import annotations
import json
import math
from typing import Optional, Tuple, Dict, Any, List
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

from pv_battery_econ.timeseries import (
    read_kwh_csv_auto,
    baseline_from_profiles,
)
from pv_battery_econ.economics.dispatch import BatteryParams, simulate_self_consumption
from pv_battery_econ.economics.cashflow import (
    Prices,
    AnnualEnergy,
    StorageEffect,
    EconParams,
    evaluate,
)
from pv_battery_econ.reporting.html_report import generate_html_report

app = typer.Typer(add_completion=False, help="Timeseries runner for pv-battery-econ (with remunerated export cap)")


# ------------------------ Generic helpers ------------------------

def _pick(dct: Dict[str, Any], aliases: List[str], *, required: bool = False, default=None, ctx: str = ""):
    for k in aliases:
        if k in dct:
            return dct[k]
    if required:
        raise KeyError(f"Missing required key in {ctx}: one of {aliases}")
    return default


def _extract_ts_and_storage(cfg: dict):
    ts = cfg.get("timeseries") or cfg.get("ts") or {}
    st = cfg.get("storage") or {}

    timeseries = {
        "consumption_csv": _pick(ts, ["consumption_csv", "cons_csv", "load_csv"], required=True, ctx="timeseries"),
        "pv_csv":          _pick(ts, ["pv_csv", "pv_production_csv", "gen_csv"], required=True, ctx="timeseries"),
        "step_seconds":    int(_pick(ts, ["step_seconds", "step", "dt_s"], default=900, ctx="timeseries")),
        "start_datetime":  _pick(ts, ["start_datetime", "start", "anchor"], default="2024-01-01T00:00:00", ctx="timeseries"),
        "tz":              _pick(ts, ["tz", "timezone"], default=None, ctx="timeseries"),
    }

    mode = (st.get("mode") or "simulate").strip().lower()
    sim = st.get("simulate") or {}
    man = st.get("manual") or {}

    storage = {
        "mode": mode,
        "simulate": {
            "capacity_kwh":   float(_pick(sim, ["capacity_kwh", "capacity"], default=20.0, ctx="storage.simulate")),
            "p_charge_kw":    float(_pick(sim, ["p_charge_kw", "p_chg_kw"], default=10.0, ctx="storage.simulate")),
            "p_discharge_kw": float(_pick(sim, ["p_discharge_kw", "p_dis_kw"], default=10.0, ctx="storage.simulate")),
            "roundtrip_eff":  float(_pick(sim, ["roundtrip_eff", "eta_rt"], default=0.92, ctx="storage.simulate")),
            "soc_min":        float(_pick(sim, ["soc_min"], default=0.05, ctx="storage.simulate")),
            "soc_max":        float(_pick(sim, ["soc_max"], default=0.95, ctx="storage.simulate")),
            "soc0":           float(_pick(sim, ["soc0"], default=0.50, ctx="storage.simulate")),
        },
        "manual": {
            "delta_ev_abs_kwh": man.get("delta_ev_abs_kwh"),
            "delta_ev_rel": man.get("delta_ev_rel"),
            "evq_with_storage": man.get("evq_with_storage"),
        }
    }
    return timeseries, storage


def _build_prices_and_econ_from_cfg(cfg: dict):
    prices_raw = cfg.get("energy_prices") or cfg.get("prices") or {}
    econ_raw   = cfg.get("economics")    or cfg.get("econ")   or {}
    se_raw     = cfg.get("storage_effect") or {}

    import_price = float(_pick(
        prices_raw,
        ["import_price_chf_per_kWh","import_price_chf_per_kwh","import_chf_per_kWh","import_chf_per_kwh","import"],
        required=True, ctx="energy_prices"
    ))
    grid_price = float(_pick(
        prices_raw,
        ["grid_price_chf_per_kWh","grid_price_chf_per_kwh","grid_chf_per_kWh","grid_chf_per_kwh","grid"],
        required=True, ctx="energy_prices"
    ))
    feed_in_price = float(_pick(
        prices_raw,
        ["feed_in_price_chf_per_kWh","feed_in_price_chf_per_kwh","feed_in_chf_per_kWh","feed_in_chf_per_kwh","feed_in","feedin_chf_per_kWh"],
        required=True, ctx="energy_prices"
    ))

    capex_battery = float(_pick(
        econ_raw,
        ["capex_battery_chf","battery_capex_chf","capex_battery"],
        required=True, ctx="economics"
    ))
    capex_install = float(_pick(
        econ_raw,
        ["capex_installation_chf","installation_capex_chf","install_capex_chf","capex_installation"],
        default=0.0, ctx="economics"
    ))
    opex_year = float(_pick(
        econ_raw,
        ["opex_chf_per_year","opex_annual_chf","opex_year_chf","opex"],
        default=0.0, ctx="economics"
    ))
    lifetime = int(_pick(
        econ_raw,
        ["lifetime_years","years","lifetime"],
        default=20, ctx="economics"
    ))
    dr_pct = float(_pick(
        econ_raw,
        ["discount_rate_pct","discount_pct","r","wacc_pct"],
        default=4.0, ctx="economics"
    ))
    subsidy = float(_pick(
        econ_raw,
        ["subsidy_upfront_chf","subsidy","grant_chf"],
        default=0.0, ctx="economics"
    ))

    decay_pct = float(_pick(
        se_raw,
        ["improvement_decay_pct_per_year","performance_decay_pct_per_year","decay_pct_per_year"],
        default=0.0, ctx="storage_effect"
    ))

    # Adapt to actual dataclass field names
    prices_params = set(signature(Prices).parameters)
    econ_params   = set(signature(EconParams).parameters)

    if {"import_price_chf_per_kWh","grid_price_chf_per_kWh","feed_in_price_chf_per_kWh"} <= prices_params:
        p = Prices(
            import_price_chf_per_kWh=import_price,
            grid_price_chf_per_kWh=grid_price,
            feed_in_price_chf_per_kWh=feed_in_price,
        )
    elif {"import_chf_per_kWh","grid_chf_per_kWh","feed_in_chf_per_kWh"} <= prices_params:
        p = Prices(
            import_chf_per_kWh=import_price,
            grid_chf_per_kWh=grid_price,
            feed_in_chf_per_kWh=feed_in_price,
        )
    else:
        raise TypeError(f"Unsupported Prices signature: {prices_params}")

    econ_kwargs = {}
    if "capex_battery_chf" in econ_params:
        econ_kwargs["capex_battery_chf"] = capex_battery
    elif "battery_capex_chf" in econ_params:
        econ_kwargs["battery_capex_chf"] = capex_battery
    else:
        raise TypeError(f"Unsupported EconParams capex field: {econ_params}")

    if "capex_installation_chf" in econ_params:
        econ_kwargs["capex_installation_chf"] = capex_install
    elif "installation_capex_chf" in econ_params:
        econ_kwargs["installation_capex_chf"] = capex_install

    if "opex_chf_per_year" in econ_params:
        econ_kwargs["opex_chf_per_year"] = opex_year
    elif "opex_annual_chf" in econ_params:
        econ_kwargs["opex_annual_chf"] = opex_year

    if "lifetime_years" in econ_params:
        econ_kwargs["lifetime_years"] = lifetime
    if "discount_rate_pct" in econ_params:
        econ_kwargs["discount_rate_pct"] = dr_pct
    if "subsidy_upfront_chf" in econ_params:
        econ_kwargs["subsidy_upfront_chf"] = subsidy

    e = EconParams(**econ_kwargs)
    return p, e, decay_pct


def _build_annual_energy(base: dict):
    params = set(signature(AnnualEnergy).parameters)
    if {"P", "L", "SC0"} <= params:
        return AnnualEnergy(
            P=base["pv_total_kWh"],
            L=base["load_total_kWh"],
            SC0=base["self_consumption_no_storage_kWh"],
        )
    elif {"pv_gen_kWh", "load_kWh", "self_consumption_no_storage_kWh"} <= params:
        return AnnualEnergy(
            pv_gen_kWh=base["pv_total_kWh"],
            load_kWh=base["load_total_kWh"],
            self_consumption_no_storage_kWh=base["self_consumption_no_storage_kWh"],
        )
    else:
        raise TypeError(f"Unsupported AnnualEnergy signature: {params}")


def _build_storage_effect(*, delta_ev_abs=None, delta_ev_rel=None, evq=None, improvement_decay=0.0):
    params = set(signature(StorageEffect).parameters)
    kwargs = {}
    if delta_ev_abs is not None:
        if "delta_ev_abs_kWh" in params:
            kwargs["delta_ev_abs_kWh"] = float(delta_ev_abs)
        elif "delta_ev_kWh" in params:
            kwargs["delta_ev_kWh"] = float(delta_ev_abs)
    if (delta_ev_rel is not None) and ("delta_ev_rel" in params):
        kwargs["delta_ev_rel"] = float(delta_ev_rel)
    if (evq is not None) and ("evq_with_storage" in params):
        kwargs["evq_with_storage"] = float(evq)
    if "improvement_decay_pct_per_year" in params:
        kwargs["improvement_decay_pct_per_year"] = float(improvement_decay)
    elif "performance_decay_pct_per_year" in params:
        kwargs["performance_decay_pct_per_year"] = float(improvement_decay)
    return StorageEffect(**kwargs)


# ------------------------ Cap helpers ------------------------

def _split_paid_free_feed_in(surplus_kwh: pd.Series, paid_cap_kWh_per_interval: float) -> Tuple[float, float]:
    """Summiere bezahlte vs. freie Einspeisemengen über das Jahr."""
    v = surplus_kwh.to_numpy(dtype=float)
    paid = np.minimum(v, paid_cap_kWh_per_interval).sum()
    free = np.maximum(v - paid_cap_kWh_per_interval, 0.0).sum()
    return float(paid), float(free)


def _split_series_paid_free(feed_series: pd.Series, paid_cap_kWh_per_interval: float) -> Tuple[pd.Series, pd.Series]:
    """Per-Intervall-Aufteilung in bezahlte vs. unbezahlte Einspeisung (kWh/Intervall)."""
    v = feed_series.to_numpy(dtype=float)
    paid = np.minimum(v, paid_cap_kWh_per_interval)
    free = np.maximum(v - paid_cap_kWh_per_interval, 0.0)
    return pd.Series(paid, index=feed_series.index), pd.Series(free, index=feed_series.index)


def _supports_return_series(func) -> bool:
    try:
        return "return_series" in signature(func).parameters
    except Exception:
        return False


def _simulate_with_series(cons: pd.Series, pv: pd.Series, step_hours: float, params: BatteryParams) -> Dict[str, Any]:
    """
    Lokale Greedy-Simulation mit Serien-Output (Fallback, falls dispatch.simulate_self_consumption
    kein return_series unterstützt). Energetik identisch zur dortigen Logik.
    Gibt Import/Feed-Serien zurück; SoC wird später rekonstruiert.
    """
    cons_v = cons.values.astype(float)
    pv_v = pv.values.astype(float)

    eff = float(params.roundtrip_eff)
    eta = math.sqrt(eff) if eff > 0 else 0.0
    cap = float(params.capacity_kWh)
    soc = float(params.soc0) * cap
    soc_min = float(params.soc_min) * cap
    soc_max = float(params.soc_max) * cap
    e_ch_max = float(params.p_charge_kW) * step_hours
    e_dis_max = float(params.p_discharge_kW) * step_hours

    feed = np.zeros_like(cons_v, dtype=float)
    imp  = np.zeros_like(cons_v, dtype=float)

    for i, (c, p) in enumerate(zip(cons_v, pv_v)):
        surplus = p - c
        if surplus > 0:
            room = soc_max - soc
            e_store_dc = max(0.0, min(room, e_ch_max * eta))
            e_req_ac = e_store_dc / eta if eta > 0 else 0.0
            e_ac = min(surplus, e_req_ac)
            e_dc = e_ac * eta
            soc += e_dc
            feed[i] = max(0.0, surplus - e_ac)
            imp[i] = 0.0
        else:
            deficit = -surplus
            avail_dc = max(0.0, min(soc - soc_min, e_dis_max))
            e_out_ac = avail_dc * eta
            used_ac = min(deficit, e_out_ac)
            used_dc = used_ac / eta if eta > 0 else 0.0
            soc -= used_dc
            imp[i] = max(0.0, deficit - used_ac)
            feed[i] = 0.0

    sc1 = cons_v.sum() - imp.sum()
    return {
        "import_with_storage_kWh": float(imp.sum()),
        "feed_in_with_storage_kWh": float(feed.sum()),
        "self_consumption_with_storage_kWh": float(sc1),
        "feed_series_kWh": feed.tolist(),
        "import_series_kWh": imp.tolist(),
    }


# ------------------------ Finance helpers (for scaling) ------------------------

def _npv(rate: float, cashflows: List[float]) -> float:
    return float(sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows)))


def _irr(cashflows: List[float], guess_low: float = -0.9, guess_high: float = 1.0, tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
    """Bisection IRR (robust, keine externen Pakete). Returns None wenn kein Vorzeichenwechsel."""
    f_low = _npv(guess_low, cashflows)
    f_high = _npv(guess_high, cashflows)
    if f_low * f_high > 0:
        return None  # kein root in [low, high]
    low, high = guess_low, guess_high
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = _npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return mid


def _discounted_payback(cashflows: List[float], rate: float) -> Optional[float]:
    acc = 0.0
    for t, cf in enumerate(cashflows):
        acc += cf / ((1.0 + rate) ** t)
        if acc >= 0:
            prev_acc = acc - cf / ((1.0 + rate) ** t)
            if cf == 0:
                return float(t)
            frac = (0 - prev_acc) / (cf / ((1.0 + rate) ** t))
            return float(t - 1 + frac)
    return None


# ------------------------ CLI ------------------------

@app.command()
def run_ts(
    config: str = typer.Option(..., help="Pfad zur YAML (Timeseries + Preise + Ökonomie)"),
    mode: Optional[str] = typer.Option(None, help="simulate | manual (überschreibt YAML)"),
    capacity_kwh: Optional[float] = typer.Option(None),
    p_charge_kw: Optional[float] = typer.Option(None),
    p_discharge_kw: Optional[float] = typer.Option(None),
    roundtrip_eff: Optional[float] = typer.Option(None),
    delta_ev_abs_kwh: Optional[float] = typer.Option(None),
    delta_ev_rel: Optional[float] = typer.Option(None),
    evq_with_storage: Optional[float] = typer.Option(None),
    report_html: Optional[str] = typer.Option(None, help="Pfad für HTML-Report (optional). Wenn nicht gesetzt, wird kein Report erzeugt."),
    series_csv: Optional[str] = typer.Option(None, help="Pfad für Timeseries-CSV (Intervallwerte)"),
    export_feed_split: bool = typer.Option(  # ← NEU
        True, "--export-feed-split/--no-export-feed-split",
        help="Paid/Free-Einspeisungsspalten in CSV exportieren (nur wenn Cap gesetzt ist)."
    ),
):
    # YAML laden
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    report_cfg = cfg.get("report") or {}
    report_path = report_html or report_cfg.get("output_html")
    report_enabled = bool(report_cfg.get("enabled", True)) if report_path else False
    series_csv_path = series_csv or report_cfg.get("timeseries_csv")

    # Timeseries + Storage aus YAML
    ts_cfg, st_cfg = _extract_ts_and_storage(cfg)

    consumption_csv = ts_cfg["consumption_csv"]
    pv_csv          = ts_cfg["pv_csv"]
    step_seconds    = ts_cfg["step_seconds"]
    start_datetime  = ts_cfg["start_datetime"]
    tz              = ts_cfg["tz"]

    # Optional: CLI-Overrides anwenden
    if mode is not None:
        st_cfg["mode"] = mode.strip().lower()
    if capacity_kwh is not None:   st_cfg["simulate"]["capacity_kwh"] = capacity_kwh
    if p_charge_kw is not None:    st_cfg["simulate"]["p_charge_kw"] = p_charge_kw
    if p_discharge_kw is not None: st_cfg["simulate"]["p_discharge_kw"] = p_discharge_kw
    if roundtrip_eff is not None:  st_cfg["simulate"]["roundtrip_eff"] = roundtrip_eff
    if delta_ev_abs_kwh is not None: st_cfg["manual"]["delta_ev_abs_kwh"] = delta_ev_abs_kwh
    if delta_ev_rel is not None:     st_cfg["manual"]["delta_ev_rel"] = delta_ev_rel
    if evq_with_storage is not None: st_cfg["manual"]["evq_with_storage"] = evq_with_storage

    # Preise & Ökonomie
    prices, econ, decay_from_cfg = _build_prices_and_econ_from_cfg(cfg)

    # Vergütungs-Kappe
    rem_raw = cfg.get("remuneration") or {}
    cap_kW = rem_raw.get("paid_feed_in_cap_kW", None)
    cap_kWh = (cap_kW * (step_seconds / 3600.0)) if cap_kW is not None else None

    # CSVs einlesen
    cons = read_kwh_csv_auto(
        consumption_csv,
        step_seconds=step_seconds,
        start_datetime=start_datetime,
        tz=tz,
    )
    pv = read_kwh_csv_auto(
        pv_csv,
        step_seconds=step_seconds,
        start_datetime=start_datetime,
        tz=tz,
    )

    # Baseline ohne Speicher
    base = baseline_from_profiles(cons, pv)
    surplus0 = (pv - cons).clip(lower=0)
    if cap_kWh is not None:
        paid0_kWh, free0_kWh = _split_paid_free_feed_in(surplus0, cap_kWh)
    else:
        paid0_kWh, free0_kWh = float(surplus0.sum()), 0.0
    import0_kWh = float((cons - pv).clip(lower=0).sum())

    # AnnualEnergy
    energy = _build_annual_energy(base)
    improvement_decay = decay_from_cfg

    # StorageEffect + ggf. Simulation
    mode_l = st_cfg["mode"]
    sim_extras: Dict[str, Any] = {"mode": mode_l}

    # Für Export bereitzuhalten:
    feed1_series: Optional[pd.Series] = None
    imp1_series: Optional[pd.Series] = None
    soc_series: Optional[pd.Series] = None
    ch_kWh_series: Optional[pd.Series] = None
    de_kWh_series: Optional[pd.Series] = None

    if mode_l == "manual":
        if sum(x is not None for x in (st_cfg["manual"]["delta_ev_abs_kwh"], st_cfg["manual"]["delta_ev_rel"], st_cfg["manual"]["evq_with_storage"])) != 1:
            raise typer.BadParameter("manual mode: genau eine der Optionen setzen: delta_ev_abs_kwh | delta_ev_rel | evq_with_storage")
        effect = _build_storage_effect(
            delta_ev_abs=st_cfg["manual"]["delta_ev_abs_kwh"],
            delta_ev_rel=st_cfg["manual"]["delta_ev_rel"],
            evq=st_cfg["manual"]["evq_with_storage"],
            improvement_decay=improvement_decay,
        )
        import1_kWh = None
        paid1_kWh = None
        free1_kWh = None
    else:
        step_hours = step_seconds / 3600.0
        params = BatteryParams(
            capacity_kWh=st_cfg["simulate"]["capacity_kwh"],
            p_charge_kW=st_cfg["simulate"]["p_charge_kw"],
            p_discharge_kW=st_cfg["simulate"]["p_discharge_kw"],
            roundtrip_eff=st_cfg["simulate"]["roundtrip_eff"],
            soc_min=st_cfg["simulate"]["soc_min"],
            soc_max=st_cfg["simulate"]["soc_max"],
            soc0=st_cfg["simulate"]["soc0"],
        )

        # Serien sicherstellen
        if _supports_return_series(simulate_self_consumption):
            sim = simulate_self_consumption(cons, pv, step_hours=step_hours, params=params, return_series=True)  # type: ignore
            feed1_series = pd.Series(sim["feed_series_kWh"], index=cons.index).astype("float64")
            imp1_series  = pd.Series(sim["import_series_kWh"], index=cons.index).astype("float64")

            if "soc_series" in sim:
                soc_series = pd.Series(sim["soc_series"], index=cons.index)
            if "charge_kWh_series" in sim:
                ch_kWh_series = pd.Series(sim["charge_kWh_series"], index=cons.index)
            if "discharge_kWh_series" in sim:
                de_kWh_series = pd.Series(sim["discharge_kWh_series"], index=cons.index)
        else:
            sim = _simulate_with_series(cons, pv, step_hours=step_hours, params=params)
            feed1_series = pd.Series(sim["feed_series_kWh"], index=cons.index).astype("float64")
            imp1_series  = pd.Series(sim["import_series_kWh"], index=cons.index).astype("float64")
            # soc/charge/discharge werden ggf. später rekonstruiert

        SC1 = float(sim["self_consumption_with_storage_kWh"])
        delta_ev_abs = SC1 - base["self_consumption_no_storage_kWh"]

        effect = _build_storage_effect(delta_ev_abs=delta_ev_abs, improvement_decay=improvement_decay)
        sim_extras.update({
            "import_with_storage_kWh": float(imp1_series.sum()),
            "feed_in_with_storage_kWh": float(feed1_series.sum()),
            "self_consumption_with_storage_kWh": float(SC1),
            "delta_ev_abs_kWh_from_sim": float(delta_ev_abs),
        })

        import1_kWh = float(imp1_series.sum())
        if cap_kWh is not None:
            paid1_kWh, free1_kWh = _split_paid_free_feed_in(feed1_series, cap_kWh)
        else:
            paid1_kWh, free1_kWh = float(feed1_series.sum()), 0.0

    # Ökonomie
    res = evaluate(energy=energy, prices=prices, effect=effect, econ=econ)
    try:
        P = float(base.get("pv_total_kWh", 0.0)) or 0.0
        SC0 = float(base.get("self_consumption_no_storage_kWh", 0.0)) or 0.0
        evq0 = (SC0 / P) if P > 0 else None

        sc1_candidates = [
            (sim_extras.get("self_consumption_with_storage_kWh") if isinstance(sim_extras, dict) else None),
            res.get("SC1_kWh"),
        ]
        SC1_val = next((x for x in sc1_candidates if x is not None), None)
        evq1 = (float(SC1_val) / P) if (SC1_val is not None and P > 0) else None

        if evq1 is not None:
            res["evq_with_storage"] = float(evq1)
        if evq0 is not None and evq0 > 0 and evq1 is not None:
            res["evq_improvement_rel"] = float((evq1 / evq0) - 1.0)
    except Exception:
        pass

    # Cap-Details (Jahr 1)
    out_cap: Dict[str, Any] = {
        "paid_feed_in_cap_kW": cap_kW,
        "paid_feed_in_no_storage_kWh": paid0_kWh,
        "free_feed_in_no_storage_kWh": free0_kWh,
    }
    if mode_l == "simulate" and cap_kWh is not None:
        import_price = getattr(prices, "import_price_chf_per_kWh", getattr(prices, "import_chf_per_kWh"))
        feed_price   = getattr(prices, "feed_in_price_chf_per_kWh", getattr(prices, "feed_in_chf_per_kWh"))

        out_cap.update({
            "paid_feed_in_with_storage_kWh": paid1_kWh,
            "free_feed_in_with_storage_kWh": free1_kWh,
        })

        import_savings_cap = (import0_kWh - import1_kWh) * import_price  # type: ignore
        lost_feed_rev_cap  = (paid0_kWh - paid1_kWh) * feed_price
        gross_benefit_cap  = import_savings_cap - lost_feed_rev_cap

        opex_year = getattr(econ, "opex_chf_per_year", getattr(econ, "opex_annual_chf", 0.0))
        annual_net_benefit_cap = gross_benefit_cap - opex_year

        out_cap.update({
            "import_savings_chf_year1_cap": import_savings_cap,
            "lost_feed_in_revenue_chf_year1_cap": lost_feed_rev_cap,
            "gross_benefit_chf_year1_cap": gross_benefit_cap,
            "annual_net_benefit_chf_year1_cap": annual_net_benefit_cap,
        })

        if bool((cfg.get("remuneration") or {}).get("apply_to_economics", True)):
            try:
                uncap_net1 = float(res.get("annual_net_benefit_chf_year1", 0.0))
                scale = (annual_net_benefit_cap / uncap_net1) if abs(uncap_net1) > 1e-12 else 1.0
            except Exception:
                scale = 1.0

            cashflows = list(res.get("cashflows", []))
            if cashflows:
                cf_scaled = [cashflows[0]] + [cf * scale for cf in cashflows[1:]]
                dr = float(getattr(econ, "discount_rate_pct", 4.0)) / 100.0
                def _npv(rate: float, cashflows: List[float]) -> float:
                    return float(sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows)))
                def _irr(cashflows: List[float], guess_low: float = -0.9, guess_high: float = 1.0, tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
                    f_low = _npv(guess_low, cashflows)
                    f_high = _npv(guess_high, cashflows)
                    if f_low * f_high > 0: return None
                    low, high = guess_low, guess_high
                    for _ in range(max_iter):
                        mid = (low + high) / 2.0
                        f_mid = _npv(mid, cashflows)
                        if abs(f_mid) < tol: return mid
                        if f_low * f_mid <= 0:
                            high, f_high = mid, f_mid
                        else:
                            low, f_low = mid, f_mid
                    return mid
                def _discounted_payback(cashflows: List[float], rate: float) -> Optional[float]:
                    acc = 0.0
                    for t, cf in enumerate(cashflows):
                        acc += cf / ((1.0 + rate) ** t)
                        if acc >= 0:
                            prev_acc = acc - cf / ((1.0 + rate) ** t)
                            if cf == 0: return float(t)
                            frac = (0 - prev_acc) / (cf / ((1.0 + rate) ** t))
                            return float(t - 1 + frac)
                    return None

                npv_new = _npv(dr, cf_scaled)
                irr_new = _irr(cf_scaled)
                dpb_new = _discounted_payback(cf_scaled, dr)
                res["cashflows"] = cf_scaled
                res["npv_chf"] = npv_new
                res["irr"] = irr_new
                res["discounted_payback_years"] = dpb_new

            res["import_savings_chf_year1"] = import_savings_cap
            res["lost_feed_in_revenue_chf_year1"] = lost_feed_rev_cap
            res["gross_benefit_chf_year1"] = gross_benefit_cap
            res["annual_net_benefit_chf_year1"] = annual_net_benefit_cap

    # --------- Timeseries-CSV schreiben (simulate-Mode) ---------
    if mode_l == "simulate" and series_csv_path:
        try:
            Path(series_csv_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        df_cols: Dict[str, Any] = {
            "consumption_kWh":           cons.values,
            "pv_generation_kWh":         pv.values,
        }

        if imp1_series is not None and feed1_series is not None:
            # Netzflüsse & Eigenverbrauch (mit Speicher)
            df_cols.update({
                "import_with_storage_kWh":   imp1_series.values,
                "feed_in_with_storage_kWh":  feed1_series.values,
                "self_consumption_kWh":      (cons - imp1_series).clip(lower=0).values,
            })

            # Paid/Free je Intervall (falls Cap definiert)
            if cap_kWh is not None and export_feed_split:  # ← export_feed_split hinzu
                paid_series, free_series = _split_series_paid_free(feed1_series, cap_kWh)
                df_cols["feed_in_paid_kWh"] = paid_series.astype("float64").values
                df_cols["feed_in_free_kWh"] = free_series.astype("float64").values

            interval_hours = step_seconds / 3600.0
            eta_rt = float(st_cfg["simulate"]["roundtrip_eff"])
            eta = eta_rt ** 0.5

            # Netto-DC-Energie je Intervall (konsistent mit SoC)
            net_dc_kWh: Optional[pd.Series] = None
            if ch_kWh_series is not None and de_kWh_series is not None:
                net_dc_kWh = (ch_kWh_series - (de_kWh_series / eta)).astype("float64")
            elif soc_series is not None:
                net_dc_kWh = soc_series.diff().fillna(0.0) * float(st_cfg["simulate"]["capacity_kwh"])
            else:
                # letzte Option: aus AC-Saldo ableiten und zu SoC integrieren
                net_ac_kWh = (pv - cons + imp1_series - feed1_series)
                cap = float(st_cfg["simulate"]["capacity_kwh"])
                s = float(st_cfg["simulate"]["soc0"])
                smin = float(st_cfg["simulate"]["soc_min"])
                smax = float(st_cfg["simulate"]["soc_max"])
                soc_vals = []
                for dE in net_ac_kWh.values:
                    s = min(max(s + (dE / cap), smin), smax)
                    soc_vals.append(s)
                soc_series = pd.Series(soc_vals, index=cons.index)
                net_dc_kWh = soc_series.diff().fillna(0.0) * cap  # konsistent mit rekonstruierter SoC

            # SoC vorhanden? Falls nein, jetzt aus net_dc_kWh rekonstruieren
            if soc_series is None and net_dc_kWh is not None:
                cap = float(st_cfg["simulate"]["capacity_kwh"])
                s = float(st_cfg["simulate"]["soc0"])
                smin = float(st_cfg["simulate"]["soc_min"])
                smax = float(st_cfg["simulate"]["soc_max"])
                soc_vals = []
                for dE in net_dc_kWh.values:
                    s = min(max(s + (dE / cap), smin), smax)
                    soc_vals.append(s)
                soc_series = pd.Series(soc_vals, index=cons.index)

            # Spalten setzen
            if net_dc_kWh is not None:
                df_cols["battery_net_energy_kWh"] = net_dc_kWh.values
                df_cols["battery_net_power_kW"]   = (net_dc_kWh / interval_hours).values

            if soc_series is not None:
                df_cols["battery_soc_ending"] = soc_series.values

                    # >>> ADD: Energie-Bestände aus SoC berechnen
                cap_batt = float(st_cfg["simulate"]["capacity_kwh"])
                energy_end = soc_series * cap_batt
                # Anfangsbestand für die 1. Zeile = soc0 * C
                energy_begin = energy_end.shift(1)
                energy_begin.iloc[0] = float(st_cfg["simulate"]["soc0"]) * cap_batt

                df_cols["battery_energy_begin_kWh"] = energy_begin.values
                df_cols["battery_energy_end_kWh"]   = energy_end.values

            if ch_kWh_series is not None and de_kWh_series is not None:
                df_cols["battery_charge_kWh"]    = ch_kWh_series.values     # DC
                df_cols["battery_discharge_kWh"] = de_kWh_series.values     # AC
                df_cols["battery_charge_kW"]     = (ch_kWh_series / interval_hours).values
                df_cols["battery_discharge_kW"]  = (de_kWh_series / interval_hours).values

        # DataFrame bauen + Typen/Clipping/Export
        df = pd.DataFrame(df_cols, index=cons.index)
        df.index.name = "timestamp"

        # Alle numerischen Energiespalten als float
        float_cols = [c for c in df.columns if c.endswith("_kWh") or c.endswith("_kW") or c == "battery_soc_ending"]
        df[float_cols] = df[float_cols].astype("float64")

        # Nur nicht-negative Größen clippen (Netto-Energie/Leistung NICHT clippen!)
        nonneg = [
            "consumption_kWh", "pv_generation_kWh",
            "import_with_storage_kWh", "feed_in_with_storage_kWh",
            "self_consumption_kWh",
            "battery_charge_kWh", "battery_discharge_kWh",
            "feed_in_paid_kWh", "feed_in_free_kWh",
        ]
        existing_nonneg = [c for c in nonneg if c in df.columns]
        if existing_nonneg:
            df[existing_nonneg] = df[existing_nonneg].clip(lower=0)

        # Saubere Ausgabe ohne unnötige Nullen
        df.to_csv(series_csv_path, float_format="%.12g")
        print(f"Wrote timeseries CSV → {series_csv_path}")

    # Finales Output
    out = {
        **res,
        "timeseries_baseline": base,
        **sim_extras,
        "remuneration_cap_details": out_cap,
        "config_used": {"config_path": config, "timeseries": ts_cfg, "storage": st_cfg},
        "timeseries_csv_path": str(series_csv_path) if series_csv_path else None,
    }

    # Optional: HTML-Report erzeugen
    if report_enabled and report_path:
        try:
            generate_html_report(out, cfg, report_path)
            out["report_html_path"] = report_path  # im JSON mit angeben
        except Exception as e:
            out["report_html_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    app()


if __name__ == "__main__":
    main()
