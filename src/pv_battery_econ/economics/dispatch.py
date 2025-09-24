from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

@dataclass
class BatteryParams:
    capacity_kWh: float
    p_charge_kW: float
    p_discharge_kW: float
    roundtrip_eff: float = 0.92
    soc_min: float = 0.05
    soc_max: float = 0.95
    soc0: float = 0.5

def simulate_self_consumption(
    cons_kwh,                     # sequence of kWh per interval (load)
    pv_kwh,                       # sequence of kWh per interval (PV gen)
    *, 
    params: BatteryParams,
    step_hours: Optional[float] = None,
    return_series: bool = False,
) -> Dict[str, Any]:
    """
    Greedy-Dispatch zur Eigenverbrauchsmaximierung mit harten Leistungskappen.
    - p_charge_kW / p_discharge_kW werden in kWh/Intervall via step_hours begrenzt.
    - Wirkungsgradbehandlung symmetrisch über eta = sqrt(roundtrip_eff).
      * 'charge_kWh_series' = im Akku gespeicherte Energie (DC) je Intervall.
      * 'discharge_kWh_series' = ans Haus abgegebene Energie (AC) je Intervall.
    - 'import_series_kWh' und 'feed_series_kWh' sind AC-Netzflüsse.
    """
    v_cons = [float(x) for x in cons_kwh]
    v_pv   = [float(x) for x in pv_kwh]
    n = len(v_cons)
    assert n == len(v_pv), "cons_kwh und pv_kwh müssen gleich lang sein"

    cap   = float(params.capacity_kWh)
    soc   = float(params.soc0) * cap                # kWh (DC) im Speicher
    soc_min_kWh = float(params.soc_min) * cap
    soc_max_kWh = float(params.soc_max) * cap

    # Leistungskappen → kWh pro Intervall
    if step_hours is None or step_hours <= 0:
        e_ch_max = float("inf")
        e_dis_max = float("inf")
    else:
        e_ch_max  = float(params.p_charge_kW)    * step_hours  # AC-Energie, die für Laden genutzt werden darf
        e_dis_max = float(params.p_discharge_kW) * step_hours  # DC-Energie, die pro Intervall entnommen werden darf

    # Effizienzaufteilung (symmetrisch)
    eff = max(1e-9, float(params.roundtrip_eff))
    eta = math.sqrt(eff)  # DC↔AC Umrechnung pro Halbweg

    import_series  = [0.0] * n
    feed_series    = [0.0] * n
    soc_series     = [0.0] * n
    charge_dc_ser  = [0.0] * n
    discharge_ac_ser = [0.0] * n

    for t in range(n):
        c = v_cons[t]  # AC last
        p = v_pv[t]    # AC pv
        surplus = p - c

        if surplus >= 0:
            # Laden: wir dürfen höchstens e_ch_max AC zum Laden abzweigen
            # dieser AC-Anteil speichert DC-Energie von e_dc = e_ac * eta
            room_dc = max(0.0, soc_max_kWh - soc)
            # durch Leistungskappe begrenzte AC-Ladeenergie
            e_ac_cap = e_ch_max if math.isfinite(e_ch_max) else surplus
            e_ac = min(surplus, e_ac_cap)
            e_dc = e_ac * eta
            # zusätzlich durch freien Speicher begrenzen
            if e_dc > room_dc:
                e_dc = room_dc
                e_ac = e_dc / eta
            # State update
            soc += e_dc
            charge_dc_ser[t] = e_dc
            # Restüberschuss einspeisen
            feed_series[t] = max(0.0, surplus - e_ac)
            import_series[t] = 0.0

        else:
            # Entladen: Bedarf decken, aber durch SoC, Leistung und Effizienz begrenzt
            deficit = -surplus
            # verfügbare DC-Menge im Akku (SoC-Min beachten)
            avail_dc = max(0.0, soc - soc_min_kWh)
            # Leistungskappe (DC-Seite) begrenzt zusätzlich
            e_dc_cap = e_dis_max if math.isfinite(e_dis_max) else avail_dc
            e_dc_use = min(avail_dc, e_dc_cap)
            # was davon als AC am Haus ankommt:
            e_ac_max = e_dc_use * eta
            used_ac = min(deficit, e_ac_max)
            used_dc = used_ac / eta if eta > 0 else 0.0
            # State update
            soc -= used_dc
            discharge_ac_ser[t] = used_ac
            # verbleibender Bedarf → Netzimport
            import_series[t] = max(0.0, deficit - used_ac)
            feed_series[t] = 0.0

        # SoC im Intervallende
        soc = min(max(soc, soc_min_kWh), soc_max_kWh)
        soc_series[t] = soc / cap  # 0..1

    # Jahreswerte
    imp_sum  = float(sum(import_series))
    feed_sum = float(sum(feed_series))
    sc1 = float(sum(v_cons) - imp_sum)

    res = {
        "import_with_storage_kWh": imp_sum,
        "feed_in_with_storage_kWh": feed_sum,
        "self_consumption_with_storage_kWh": sc1,
        "import_series_kWh": import_series,
        "feed_series_kWh":   feed_series,
    }
    if return_series:
        res.update({
            "soc_series": soc_series,                      # 0..1 (Ende jedes Intervalls)
            "charge_kWh_series": charge_dc_ser,           # im Akku gespeicherte Energie (DC)
            "discharge_kWh_series": discharge_ac_ser,     # ans Haus gelieferte Energie (AC)
        })
    return res
