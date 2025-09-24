# pv-battery-econ

PV-Batterie Wirtschaftlichkeits-Tool (Zeitscheiben-Simulation + Report). Erzeugt einen kompakten HTML-Report und eine Timeseries-CSV auf Basis von Last- und PV-Profilen.

## Quickstart (Windows PowerShell)

```powershell
# ins Projekt wechseln
Set-Location "C:\\Users\\<du>\\OneDrive - Eniwa AG\\Dokumente\\code\\pv-battery-econ"

# frische venv (empfohlen nach Ordner-Umbenennung)
py -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Installation (liest pyproject.toml)
python -m pip install --upgrade pip
pip install -e .

# optional: Reports-Ordner anlegen
New-Item -ItemType Directory -Path reports -Force | Out-Null

# Run (Timeseries → HTML + CSV, ohne Paid/Free-Spalten)
python -m pv_battery_econ.cli.run_ts `
  --config "configs/site_timeseries.yaml" `
  --report-html "reports/2025-09-24_(10kwh)_Philippe.html" `
  --series-csv "reports/2025-09-24_(10kwh)_Philippe_timeseries.csv" `
  --no-export-feed-split
```

**Check:** `pvbat hello` sollte *„pv-battery-econ CLI läuft.“* ausgeben.

## Quickstart (macOS / Linux)

```bash
cd /pfad/zu/pv-battery-econ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

mkdir -p reports
python -m pv_battery_econ.cli.run_ts   --config "configs/site_timeseries.yaml"   --report-html "reports/2025-09-24_(10kwh)_Philippe.html"   --series-csv "reports/2025-09-24_(10kwh)_Philippe_timeseries.csv"   --no-export-feed-split
```

## Eingaben

- **Configs** (`configs/*.yaml`)
  - `site_timeseries.yaml`: Pfade zu **consumption_csv** und **pv_csv**, Zeitraster (`step_seconds`), Startzeitpunkt, Zeitzone.
  - `storage.simulate`: `capacity_kwh`, `p_charge_kw`, `p_discharge_kw`, `roundtrip_eff` (z. B. 0.92).

## Ausgaben

- **HTML-Report** (`reports/*.html`)
  Enthält u. a.:
  - Cashflows & Payback (diskontiert & einfach)
  - Energiebilanz (Import / Eigenverbrauch / Einspeisung)
  - **Batterie-SoC über Zeit** (ersetzt frühere „Bezahlte vs. freie Einspeisung“-Grafik)
  - Kennzahlenkarten inkl. „ohne Batterie“ und „mit Batterie“

- **Timeseries-CSV** (`reports/*_timeseries.csv`) mit u. a.:
  - `consumption_kWh`, `pv_generation_kWh`, `import_with_storage_kWh`, `feed_in_with_storage_kWh`, `self_consumption_kWh`
  - Batterie: `battery_charge_kWh` (DC, ≥0), `battery_discharge_kWh` (AC, ≥0), `battery_net_energy_kWh` (DC, ±), `battery_net_power_kW`, `battery_energy_begin_kWh`, `battery_energy_end_kWh`
  - **Optional:** `feed_in_paid_kWh`, `feed_in_free_kWh` — werden **unterdrückt**, wenn `--no-export-feed-split` gesetzt ist oder keine Cap konfiguriert ist.

## CLI

```text
python -m pv_battery_econ.cli.run_ts --config <yaml> [--report-html <html>] [--series-csv <csv>] [--no-export-feed-split]
```

Wichtige Optionen (Auszug):
- `--config` Pfad zur YAML (Timeseries + Storage + Preise + Ökonomie)
- `--report-html` Pfad für HTML-Report (optional; wenn nicht gesetzt → kein Report)
- `--series-csv` Pfad für Timeseries-CSV (optional; wenn nicht gesetzt → keine CSV)
- `--no-export-feed-split` unterdrückt `feed_in_paid_kWh`/`feed_in_free_kWh` im CSV
- Storage-Overrides (optional): `--mode simulate|manual`, `--capacity-kwh`, `--p-charge-kw`, `--p-discharge-kw`, `--roundtrip-eff`, …

## Installation / Anforderungen

- Python >=3.10
- Abhängigkeiten aus `pyproject.toml` (u. a. matplotlib>=3.9, numpy, pandas, pydantic>=2, pyyaml, typer>=0.12).

## Hinweise

- Die CSV-Spalten `battery_*` sind konsistent:
  `battery_net_energy_kWh = battery_charge_kWh - (battery_discharge_kWh / sqrt(roundtrip_eff))`
  und `battery_energy_end_kWh - battery_energy_begin_kWh = battery_net_energy_kWh`.
- Wenn du den Ordner umbenennst/verschiebst, leg die `.venv` neu an.
