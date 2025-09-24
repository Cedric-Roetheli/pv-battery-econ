from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def _clean_energy_strings_to_float(s: pd.Series) -> pd.Series:
    """
    Robustly parse numeric strings with either '.' or ',' as decimal separator.
    - Handles tokens like '1,23', '1.23', and '1.234,56' (thousands + decimal).
    - Drops whitespace; leaves None/NaN where parsing is impossible.
    """
    s = s.astype(str).str.strip()

    def _one(x: str) -> Optional[float]:
        if x == "" or x.lower() in {"nan", "none"}:
            return None
        # if both ',' and '.' exist and comma is the decimal (like '1.234,56')
        if ("," in x) and ("." in x):
            # Heuristic: if comma is after last dot -> comma is decimal, dots are thousands
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                # else assume dot is decimal, remove commas as thousands
                x = x.replace(",", "")
        else:
            # only comma -> treat as decimal
            if "," in x:
                x = x.replace(",", ".")
            # only dot -> already decimal
        try:
            return float(x)
        except Exception:
            return None

    return s.apply(_one).astype(float)


def read_seconds_kwh_csv(
    path: str,
    *,
    time_col=0,
    value_col=1,
    step_seconds: int = 900,
    encoding: str = "utf-8",
) -> pd.Series:
    """
    Read a 2-column CSV where:
      - first column = time since start in seconds (0, 900, 1800, ...)
      - second column = energy PER INTERVAL in kWh
    The reader is robust to extra header rows like '#Zeit [s]' inside the data.
    Returns a Series indexed by integer seconds (0, step, 2*step, ...).
    """
    # sep=None lets pandas sniff ; , \t etc.
    df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)

    # Try to coerce the time column; drop any non-numeric rows (like '#Zeit [s]')
    sec_try = pd.to_numeric(df.iloc[:, time_col], errors="coerce")
    df = df.loc[sec_try.notna()].copy()
    seconds = sec_try.loc[df.index].astype(int)

    # Energy column: robust string cleanup (supports , and .)
    energy_raw = df.iloc[:, value_col]
    energy = _clean_energy_strings_to_float(energy_raw)
    # Drop trailing notes/rows that still failed
    df = df.loc[energy.notna()]
    seconds = seconds.loc[df.index]
    energy = energy.loc[df.index].astype(float)

    # Basic validation: regular step
    diffs = seconds.diff().dropna()
    if not diffs.empty and not (diffs == step_seconds).all():
        uniq = pd.unique(diffs.astype(int))[:5]
        raise ValueError(
            f"{path}: Non-uniform step detected. Expected {step_seconds}s, got diffs like {list(uniq)}"
        )

    return pd.Series(energy.values, index=seconds.values, name="kWh")


def seconds_to_datetime_index(
    n_points: int,
    *,
    start_datetime: str = "2024-01-01T00:00:00",
    step_seconds: int = 900,
    tz: Optional[str] = None,
) -> pd.DatetimeIndex:
    """
    Build a calendar index from a start timestamp and a fixed step.
    We keep exactly n_points slots (DST safe) by using a fixed-frequency range.
    """
    freq = f"{step_seconds}s"
    return pd.date_range(start=start_datetime, periods=int(n_points), freq=freq, tz=tz)


def attach_calendar(
    series_seconds: pd.Series,
    *,
    start_datetime: str = "2024-01-01T00:00:00",
    step_seconds: int = 900,
    tz: Optional[str] = None,
) -> pd.Series:
    idx = seconds_to_datetime_index(
        len(series_seconds), start_datetime=start_datetime, step_seconds=step_seconds, tz=tz
    )
    return pd.Series(series_seconds.values, index=idx, name=series_seconds.name)


def baseline_from_profiles(cons_kwh: pd.Series, pv_kwh: pd.Series) -> dict:
    """
    Compute baseline annual integrals (no battery) from kWh-per-interval series.
    """
    if len(cons_kwh) != len(pv_kwh):
        raise ValueError(
            f"Profiles length mismatch: consumption={len(cons_kwh)} vs pv={len(pv_kwh)}"
        )
    cons = cons_kwh.astype(float)
    pv = pv_kwh.astype(float)
    sc0 = np.minimum(cons, pv).sum()
    feed0 = (pv - cons).clip(lower=0).sum()
    import0 = (cons - pv).clip(lower=0).sum()
    return {
        "pv_total_kWh": float(pv.sum()),
        "load_total_kWh": float(cons.sum()),
        "self_consumption_no_storage_kWh": float(sc0),
        "feed_in_no_storage_kWh": float(feed0),
        "import_no_storage_kWh": float(import0),
    }

def split_paid_free_feed_in(surplus_kwh: pd.Series, paid_cap_kWh_per_interval: float) -> tuple[float, float]:
    """
    surplus_kwh: Serie der Nettoeinspeisung pro Intervall (kWh >= 0)
    paid_cap_kWh_per_interval: z. B. 7.5 bei 30 kW und 15 Minuten
    Rückgabe: (paid_kWh, free_kWh) — summiert über das Jahr.
    """
    v = surplus_kwh.to_numpy(dtype=float)
    paid = np.minimum(v, paid_cap_kWh_per_interval).sum()
    free = np.maximum(v - paid_cap_kWh_per_interval, 0.0).sum()
    return float(paid), float(free)

# --- NEU: flexibler CSV-Reader (Sekunden ODER Datetime) ---
from typing import Optional
import numpy as np
import pandas as pd

def read_kwh_csv_auto(
    path: str,
    step_seconds: int,
    start_datetime: Optional[str] = None,
    tz: Optional[str] = None,
    time_col: int = 0,
    value_col: int = 1,
    encoding: str = "utf-8",
) -> pd.Series:
    """
    Liest CSV mit 2 Spalten:
      - Variante A: [Sekunden, kWh/Intervall]
      - Variante B: [Datetime, kWh/Intervall] (z.B. '2024-01-01 00:00:00')
    Rückgabe: Series mit DateTimeIndex (ggf. tz-aware), Werte in kWh/Intervall.
    Validiert, dass die Schrittweite ~ step_seconds ist (±5% Toleranz).
    """
    # robustes Einlesen (Komma/Semikolon etc.)
    df = pd.read_csv(path, sep=None, engine="python", encoding=encoding)
    if df.shape[1] < max(time_col, value_col) + 1:
        raise ValueError(f"{path}: erwartet mind. {max(time_col, value_col)+1} Spalten, gefunden {df.shape[1]}")

    time_raw = df.iloc[:, time_col]
    val_raw  = df.iloc[:, value_col]

    # Werte in kWh tolerant parsen (Komma-/Punkt-Dezimaltrennzeichen)
    vals = pd.to_numeric(
        val_raw.astype(str)
               .str.replace("\u00a0", "", regex=False)  # non-breaking space
               .str.replace(" ", "", regex=False)
               .str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # Zeilen mit NaN-Werten verwerfen
    mask_ok = vals.notna()
    time_raw = time_raw[mask_ok]
    vals     = vals[mask_ok]

    # 1) Versuch: Sekunden in Zeitspalte
    sec_try = pd.to_numeric(
        time_raw.astype(str).str.replace(" ", "", regex=False),
        errors="coerce"
    )
    use_seconds = sec_try.notna().sum() >= int(0.95 * len(time_raw)) and (sec_try.dropna().min() >= 0)

    if use_seconds:
        # Sekunden -> DateTimeIndex via start_datetime
        if not start_datetime:
            raise ValueError(
                f"{path}: Zeitspalte sieht nach Sekunden aus, aber 'start_datetime' fehlt. "
                f"Setze 'timeseries.start_datetime' in der YAML."
            )
        # FUTURE WARNING fix: .ffill() statt fillna(method="ffill")
        sec = sec_try.ffill().astype(int)

        base = pd.to_datetime(start_datetime)

        if tz:
            # *** DST-sicherer Weg ***
            # Startzeit als tz-aware (lokale Zeitzone) setzen und dann Timedeltas addieren.
            # Pandas überspringt dabei die nicht-existente Stunde (Spring forward) korrekt.
            base = pd.Timestamp(base)
            if base.tzinfo is None:
                base = base.tz_localize(tz)
            else:
                base = base.tz_convert(tz)
            idx = base + pd.to_timedelta(sec, unit="s")
            idx = pd.DatetimeIndex(idx)
        else:
            # keine TZ: naive Zeiten ok
            idx = pd.DatetimeIndex(pd.Timestamp(base) + pd.to_timedelta(sec, unit="s"))

    else:
        # 2) Versuch: Datetime parsen
        dt = pd.to_datetime(time_raw, errors="coerce", infer_datetime_format=True)
        if dt.isna().all():
            raise ValueError(
                f"{path}: Zeitspalte konnte weder als Sekunden noch als Datum/Uhrzeit geparst werden."
            )
        idx = pd.DatetimeIndex(dt)
        if tz:
            # *** DST-robust lokalisieren ***
            # - 'nonexistent=shift_forward' schiebt 02:00→03:00 bei Frühjahrsumstellung
            # - 'ambiguous=infer' löst die doppelte Stunde im Herbst
            if idx.tz is None:
                idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            else:
                idx = idx.tz_convert(tz)

    s = pd.Series(vals.values, index=idx, name="kWh").sort_index()

    # Duplikate auf gleichem Timestamp summieren (z.B. doppelte Zeilen)
    if s.index.has_duplicates:
        s = s.groupby(level=0).sum()

    # Schrittweite validieren (±5% Toleranz, um Randfälle/DST zuzulassen)
    if len(s) >= 3:
        # Delta in Sekunden aus DatetimeIndex (ns -> s)
        diffs_sec = np.diff(s.index.view("i8")) / 1e9
        med = float(np.median(diffs_sec))
        rel = abs(med - float(step_seconds)) / float(step_seconds)
        if rel > 0.05:
            raise ValueError(
                f"{path}: erkannte Schrittweite ~{med:.0f}s passt nicht zu step_seconds={step_seconds}. "
                "Bitte YAML anpassen oder Daten vorab auf die gewünschte Auflösung bringen."
            )

    return s

