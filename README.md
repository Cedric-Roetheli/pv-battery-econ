# PV Battery Econ

Tool zur **Wirtschaftlichkeitsberechnung von PV‑Batteriespeichern** mit stündlichen oder 15‑minütigen Lastgängen (Timeseries‑Modus), optionaler **Vergütungs‑Kappe**, HTML‑Report und sauberem **Timeseries‑CSV‑Export** für weiterführende Analysen.

> Dieses README dokumentiert den aktuellen **Timeseries‑Runner**. Der frühere Jahres‑Aggregat‑Workflow ist optional nutzbar, aber hier nicht beschrieben.

---

## ✨ Features

* **Timeseries‑Simulation** (15 min oder 60 min) inkl. Leistungsgrenzen (p\_charge/p\_discharge) und Wirkungsgrad.
* **Vergütungskappe** (Paid vs. Free Einspeisung pro Intervall, z. B. nur die ersten 30 kW vergütet).
* **Ökonomie‑KPIs**: NPV, IRR, discounted Payback, LCOS.
* **HTML‑Report** (optional; headless‑safe via `matplotlib`/Agg‑Backend).
* **Timeseries‑CSV** inkl. SoC, Netto‑Energie/‑Leistung der Batterie (DC‑bezogen), Import/Export, Paid/Free, u. v. m.

---

## 🚀 Quickstart

### 1) Installation (Windows PowerShell)

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -e .
pip install matplotlib  # für den optionalen HTML-Report
```

### 2) Daten ablegen

CSV‑Dateien unter `data/`, z. B.:

* `data/2024_Gesamtverbrauch.csv`
* `data/2024_PV-Produktion.csv`

**CSV‑Format (Timeseries):** 2 Spalten

* Spalte 1: **Zeit in Sekunden** seit Start (z. B. 0, 900, 1800 …) oder Zeitstempel
* Spalte 2: **Energie in kWh pro Intervall**

Kopfzeilen sind ok; Komma-/Punkt‑Dezimalzeichen werden robust geparst.

### 3) Konfiguration erstellen (`configs/site_timeseries.yaml`)

```yaml
input_mode: timeseries

timeseries:
  consumption_csv: "data/2024_Gesamtverbrauch.csv"   # Sekunden oder Zeitstempel, kWh/Intervall
  pv_csv:          "data/2024_PV-Produktion.csv"
  step_seconds: 900                                   # 900 für 15 min; 3600 für 1 h
  start_datetime: "2024-01-01T00:00:00"              # nur nötig, wenn Spalte 1 Sekunden sind
  tz: "Europe/Zurich"                                 # optional

storage:
  mode: simulate               # 'simulate' oder 'manual'
  simulate:
    capacity_kwh: 40
    p_charge_kw: 15
    p_discharge_kw: 15
    roundtrip_eff: 0.92        # Gesamt-Wirkungsgrad; intern mit sqrt(eta) je Richtung
    soc_min: 0.05              # min. SoC (relativ 0..1)
    soc_max: 0.95              # max. SoC (relativ 0..1)
    soc0:    0.05              # Start‑SoC (relativ 0..1)

# Energiepreise (CHF/kWh)
energy_prices:
  import_chf_per_kWh:   0.18
  grid_chf_per_kWh:     0.00
  feed_in_chf_per_kWh:  0.09

# Ökonomie
economics:
  capex_battery_chf:        27000
  capex_installation_chf:    2000
  opex_annual_chf:             50
  lifetime_years:              25
  discount_rate_pct:          3.0
  subsidy_upfront_chf:     13000

# Optional: jährlicher Nutzen-/Performance-Abfall (%/a)
storage_effect:
  improvement_decay_pct_per_year: 0.0

# Optional: Vergütungskappe (nur die ersten X kW Einspeiseleistung werden bezahlt)
remuneration:
  paid_feed_in_cap_kW: 30.0         # 30 kW ⇒ bei 15 min = 7.5 kWh/Intervall
  apply_to_economics: true          # cap-bewusste Jahr-1-Kennzahlen & Cashflows

# Reports/Exports
report:
  enabled: true
  output_html:   "reports/site_2024.html"
  timeseries_csv:"reports/site_2024_timeseries.csv"
```

### 4) Run

**Windows PowerShell (eine Zeile):**

```powershell
python -m pv_battery_econ.cli.run_ts --config "configs\site_timeseries.yaml"
```

**Mit Report und Timeseries‑CSV explizit:**

```powershell
python -m pv_battery_econ.cli.run_ts `
  --config "configs\site_timeseries.yaml" `
  --report-html "reports\site_2024.html" `
  --series-csv  "reports\site_2024_timeseries.csv"
```

*(Backticks sind PowerShell‑Zeilenumbrüche. In CMD/Git‑Bash bitte alles in **eine** Zeile oder `\` korrekt escapen.)*

**CLI‑Overrides (überschreiben YAML‑Werte):**
`--mode`, `--capacity-kwh`, `--p-charge-kw`, `--p-discharge-kw`, `--roundtrip-eff`, `--delta-ev-abs-kwh`, `--delta-ev-rel`, `--evq-with-storage`, `--report-html`, `--series-csv`.

---

## 📦 Outputs

### 1) JSON auf STDOUT

Die wichtigsten KPIs und Konfigurationen (für Pipelines/Weiterverarbeitung). Enthält u. a. `npv_chf`, `irr`, `discounted_payback_years`, `timeseries_baseline`, ggf. cap‑bewusste Jahr‑1‑Zahlen.

### 2) HTML‑Report (optional)

Grafiken und Tabellen zu Energiebilanzen, Cashflows, Payback‑Kurve etc. (`matplotlib` mit `Agg`‑Backend, daher headless‑fähig).

### 3) Timeseries‑CSV (optional)

Intervallwerte (Index: `timestamp`). Typische Spalten:

**Grunddaten**

* `consumption_kWh` – Verbrauch (kWh/Intervall, **≥0**)
* `pv_generation_kWh` – PV‑Erzeugung (kWh/Intervall, **≥0**)

**Netzflüsse** (mit Speicher)

* `import_with_storage_kWh` – Netzbezug (kWh/Intervall, **≥0**)
* `feed_in_with_storage_kWh` – Einspeisung (kWh/Intervall, **≥0**)
* `self_consumption_kWh` – Eigenverbrauch (kWh/Intervall, **≥0**)

**Vergütungskappe (falls aktiv)**

* `feed_in_paid_kWh` – bezahlte Einspeisung (kWh/Intervall, **≥0**)
* `feed_in_free_kWh` – unbezahlte Einspeisung (kWh/Intervall, **≥0**)

**Batterie (DC‑Bezug und SoC)**

* `battery_soc_ending` – SoC am **Intervallende** (relativ 0..1)
* `battery_net_energy_kWh` – **Netto‑DC‑Energiefluss** im Intervall: **>0** Laden, **<0** Entladen (kWh/Intervall)
* `battery_net_power_kW` – **Netto‑DC‑Leistung**: `battery_net_energy_kWh / Δt_h` (kW)
* `battery_charge_kWh` – geladene **DC‑Energie** im Intervall (kWh, **≥0**)
* `battery_discharge_kWh` – abgegebene **AC‑Energie** ins Haus im Intervall (kWh, **≥0**)
* `battery_energy_begin_kWh` – Energie‑**Bestand am Intervallanfang** (SoC·Kapazität)
* `battery_energy_end_kWh` – Energie‑**Bestand am Intervallende** (SoC·Kapazität)

> **Wichtig (AC vs DC):**
>
> * `battery_net_energy_kWh` ist **DC‑bezogen** und konsistent zu `ΔSoC·Kapazität`.
> * AC‑Bilanz lautet `pv - consumption + import - feed` (AC‑Seite) und ist wegen Wirkungsgrad ≠ DC‑Netto. Näherungsweise:
>
>   * Laden: `net_ac ≈ net_dc / sqrt(eta_rt)`
>   * Entladen: `net_ac ≈ net_dc * sqrt(eta_rt)`

---

## ⚙️ Konventionen & Parameter

* **Zeitraster:** `step_seconds` = `900` (15 min) oder `3600` (1 h). Leistungskappen werden pro Intervall strikt erzwungen: `charge_kWh ≤ p_charge_kW·Δt_h`, `discharge_kWh ≤ p_discharge_kW·Δt_h`.
* **SoC‑Grenzen:** `soc_min` ≤ SoC ≤ `soc_max`; Startwert `soc0`. Beispiel: `soc0: 0.05` ⇒ Start bei 5 % und nie tiefer als 5 %.
* **Effizienz:** `roundtrip_eff = eta_rt`; intern wird `eta = sqrt(eta_rt)` je Richtung verwendet.
* **Vergütungskappe:** `paid_feed_in_cap_kW` ⇒ pro Intervall `cap_kWh = cap_kW·Δt_h`; Paid vs Free wird je Intervall gesplittet und für Jahr‑1‑Ökonomie optional berücksichtigt (`apply_to_economics`).

---

## 🧪 Beispiele

**15 min‑Run mit Export & Report:**

```powershell
python -m pv_battery_econ.cli.run_ts `
  --config "configs\site_timeseries.yaml" `
  --report-html "reports\site_2024.html" `
  --series-csv  "reports\site_2024_timeseries.csv"
```

**1 h‑Run (nur YAML anpassen):**

```yaml
step_seconds: 3600
```

---

## 🛠️ Troubleshooting

* **PowerShell meldet „unexpected token --…“** → Backslashes für Zeilenumbrüche funktionieren in PS nicht. Nutze **Backticks** (`` ` ``) oder eine **einzeilige** Command‑Line.
* **`matplotlib.pyplot` „could not be resolved“** → Stelle sicher, dass dein VS‑Code den **.venv** Interpreter nutzt und `pip install matplotlib` ausgeführt wurde. In `html_report.py` wird `Agg`‑Backend gesetzt.
* **CSV zeigt 4.000000** → Wir schreiben mit `float_format="%.12g"` und erzwingen `float64`‑Dtypes; keine unnötigen Nachkommastellen. Für feste drei Nachkommastellen: Excel‑Format `0.000`.
* **Netto‑Energie nur ganzzahlig** → Stelle sicher, dass Casting auf `float64` **vor** dem Export erfolgt (bereits implementiert).

---

## 📄 Lizenz & Beitrag

* Lizenz: (falls vorhanden ergänzen)

---

## Changelog (Auszug)

* **2025‑09**: Timeseries‑CSV erweitert (Paid/Free, SoC, Netto‑DC‑Leistung/Energie, Bestands‑Spalten), harte Leistungskappen pro Intervall, HTML‑Report headless‑safe.
