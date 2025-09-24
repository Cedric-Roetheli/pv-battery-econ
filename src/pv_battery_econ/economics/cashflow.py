from dataclasses import dataclass
from typing import Optional, List
import logging
logging.basicConfig(level=logging.INFO)


@dataclass
class Prices:
    import_chf_per_kWh: float
    grid_chf_per_kWh: float = 0.0
    feed_in_chf_per_kWh: float = 0.0

@dataclass
class AnnualEnergy:
    pv_gen_kWh: float
    load_kWh: float
    self_consumption_no_storage_kWh: float

@dataclass
class StorageEffect:
    evq_with_storage: Optional[float] = None
    delta_ev_abs_kWh: Optional[float] = None
    delta_ev_rel: Optional[float] = None
    improvement_decay_pct_per_year: float = 0.0  # 0.01 = 1%/a


@dataclass
class EconParams:
    capex_battery_chf: float
    capex_installation_chf: float = 0.0
    opex_annual_chf: float = 0.0
    lifetime_years: int = 10
    discount_rate_pct: float = 5.0
    subsidy_upfront_chf: float = 0.0   # NEU

def _npv(rate: float, cashflows: List[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def _irr(cashflows: List[float]) -> Optional[float]:
    # einfache Bisektion
    low, high = -0.9, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        npv = _npv(mid, cashflows)
        if abs(npv) < 1e-6:
            return mid
        if npv > 0:
            low = mid
        else:
            high = mid
    return None

def _discounted_payback(rate: float, cashflows: List[float]):
    cum = 0.0
    for t, cf in enumerate(cashflows):
        cum += cf / ((1 + rate) ** t)
        if cum >= 0:
            return float(t)
    return None

def _simple_payback_years(cashflows_undisc):
    """Linear interpolierter Simple Payback (in Jahren); None, wenn nicht erreicht."""
    cum = 0.0
    prev = 0.0
    for year, cf in enumerate(cashflows_undisc):
        cum += cf
        if cum >= 0.0:
            if year == 0:
                return 0.0
            c0, c1 = prev, cum
            if c1 == c0:
                return float(year)
            frac = (0.0 - c0) / (c1 - c0)
            return float(year - 1) + max(0.0, min(1.0, frac))
        prev = cum
    return None


def _pv_series(values: List[float], rate: float, start_t: int = 1) -> float:
    return sum(v / ((1 + rate) ** t) for t, v in enumerate(values, start_t))


def evaluate(energy: AnnualEnergy, prices: Prices, effect: StorageEffect, econ: EconParams):
    P, L, SC0 = energy.pv_gen_kWh, energy.load_kWh, energy.self_consumption_no_storage_kWh

    # 1) Ziel-Delta bestimmen (Eigenverbrauchssteigerung)
    if effect.delta_ev_abs_kWh is not None:
        delta_target = max(0.0, effect.delta_ev_abs_kWh)
    elif effect.delta_ev_rel is not None:
        delta_target = max(0.0, SC0 * effect.delta_ev_rel)
    elif effect.evq_with_storage is not None:
        # Falls jemand doch EVQ setzt, berechnen wir das entsprechende Delta
        sc1_target = min(L, effect.evq_with_storage * P)
        delta_target = max(0.0, sc1_target - SC0)
    else:
        raise ValueError("Gib delta_ev_abs_kWh ODER delta_ev_rel ODER evq_with_storage an.")

    # 2) Physikalisch/energetisch mögliche Obergrenze für Delta:
    #    - nicht mehr als PV-Überschuss (was bisher eingespeist wurde)
    #    - nicht mehr als Eigenverbrauchs-Lücke (was noch zu Last fehlen würde)
    delta_max_pv = max(0.0, P - SC0)   # bisherige Einspeisung
    delta_max_load = max(0.0, L - SC0) # unversorgter Lastteil
    delta_cap = min(delta_max_pv, delta_max_load)

    # 3) Gekapptes Delta verwenden
    delta = min(delta_target, delta_cap)

    # Warnung ausgeben, falls Kappung aktiv
    if delta < delta_target:
        logging.warning(
            f"ΔEV wurde gekappt: Eingabe {delta_target:.1f} kWh, "
            f"zulässig max. {delta_cap:.1f} kWh (limitiert durch PV-Überschuss {delta_max_pv:.1f} "
            f"und Last-Lücke {delta_max_load:.1f})."
        )

    # 4) Resultierender Eigenverbrauch mit Speicher
    SC1 = SC0 + delta

    # 5) Import/Einspeisung ändern sich genau um Delta:
    I0 = L - SC0
    I1 = L - SC1
    d_import = max(0.0, I0 - I1)   # = delta
    d_feed   = (P - SC1) - (P - SC0)  # = -delta

    # Komponenten
    import_savings_chf = (prices.import_chf_per_kWh + prices.grid_chf_per_kWh) * d_import
    lost_feed_in_revenue_chf = prices.feed_in_chf_per_kWh * d_import  # entgangene Vergütung (Kosten)

    # RICHTIG: entgangene Vergütung abziehen
    energy_benefit_chf = import_savings_chf - lost_feed_in_revenue_chf
    net_year = energy_benefit_chf - econ.opex_annual_chf

    total_capex = econ.capex_battery_chf + econ.capex_installation_chf
    subsidy = max(0.0, econ.subsidy_upfront_chf)
    applied_subsidy = min(subsidy, total_capex)  # Zuschuss kann Invest nicht übersteigen
    net_capex = total_capex - applied_subsidy
    capex = -net_capex  # Jahr-0-CF (negativ)

    # Optional: kurze Warnung, falls gekappt
    import logging
    if subsidy > applied_subsidy:
        logging.warning(
            f"Subsidy gekappt: beantragt {subsidy:.2f} CHF, anrechenbar {applied_subsidy:.2f} CHF (<= CAPEX {total_capex:.2f} CHF)."
        )


    r = econ.discount_rate_pct / 100.0
    decay = effect.improvement_decay_pct_per_year

    # Jahr 0 + degradierende Jahresnutzen
    cfs = [capex]
    for t in range(1, econ.lifetime_years + 1):
        cf_t = net_year * ((1 - decay) ** (t - 1))
        cfs.append(cf_t)

    # --- NEU: Undiskontierte Cashflows + Simple Payback ---
    # Annahme: 'cfs' ist deine diskontierte Serie (inkl. Jahr 0 = -net_capex).
    cashflows_disc = list(cfs)  # Alias für Klarheit / spätere Ausgabe

    # Diskontierung rückrechnen: CF_undisc[t] = CF_disc[t] * (1+r)^t
    r = float(getattr(econ, "discount_rate", 0.0) or 0.0)
    if r != 0.0:
        cashflows_undisc = [cf * ((1.0 + r) ** t) for t, cf in enumerate(cashflows_disc)]
    else:
        cashflows_undisc = list(cashflows_disc)

    spb = _simple_payback_years(cashflows_undisc)

    # KPIs
    npv = _npv(r, cfs)
    irr = _irr(cfs)
    dpb = _discounted_payback(r, cfs)

    # LCOS (vereinfachte Sicht): PV(Kosten) / PV(verschobene kWh)
    if d_import <= 0:
        lcos = float("inf")
    else:
        pv_costs = -capex + _pv_series([econ.opex_annual_chf] * econ.lifetime_years, r, 1)
        pv_shifted = _pv_series([d_import] * econ.lifetime_years, r, 1)
        lcos = pv_costs / pv_shifted if pv_shifted > 0 else float("inf")

    import_savings_chf = (prices.import_chf_per_kWh + prices.grid_chf_per_kWh) * d_import
    lost_feed_in_revenue_chf = prices.feed_in_chf_per_kWh * d_import  # entgangene Vergütung
    gross_benefit_chf = import_savings_chf + lost_feed_in_revenue_chf

    return {
        "SC1_kWh": SC1,
        "delta_import_kWh": d_import,

        # Jahr 1
        "import_savings_chf_year1": import_savings_chf,
        "lost_feed_in_revenue_chf_year1": lost_feed_in_revenue_chf,  # positiv ausgewiesen, semantisch ein Abzug
        "gross_benefit_chf_year1": energy_benefit_chf,               # jetzt: Ersparnis MINUS entgangene Vergütung
        "annual_net_benefit_chf_year1": net_year,
        
        "capex_total_chf": total_capex,
        "subsidy_upfront_chf_applied": applied_subsidy,
        "capex_net_chf": net_capex,

        "npv_chf": npv,
        "irr": irr,
        "discounted_payback_years": dpb,
        "lcos_chf_per_kWh": lcos,
        "cashflows": cfs,

        "cashflows_disc": cashflows_disc,           # explizit benannt (identisch zu 'cashflows')
        "cashflows_undisc": cashflows_undisc,       # NEU für Simple Payback
        "simple_payback_years": spb,                # NEU: SPB

        # … evtl. weitere KPIs
    }



