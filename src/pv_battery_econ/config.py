from pydantic import BaseModel
import yaml
from .economics.cashflow import Prices, AnnualEnergy, StorageEffect, EconParams

class Config(BaseModel):
    class EnergyPrices(BaseModel):
        import_chf_per_kWh: float
        grid_chf_per_kWh: float = 0.0
        feed_in_chf_per_kWh: float = 0.0

    class AnnualEnergyCfg(BaseModel):
        pv_gen_kWh: float
        load_kWh: float
        self_consumption_no_storage_kWh: float
        # Optional: Wenn vorhanden, prüfen wir PV ≈ EV + Rückspeisung
        feed_in_no_storage_kWh: float | None = None

    class StorageEffectCfg(BaseModel):
        evq_with_storage: float | None = None
        delta_ev_abs_kWh: float | None = None
        delta_ev_rel: float | None = None
        improvement_decay_pct_per_year: float = 0.0


    class EconomicsCfg(BaseModel):
        capex_battery_chf: float
        capex_installation_chf: float = 0.0
        opex_annual_chf: float = 0.0
        lifetime_years: int = 10
        discount_rate_pct: float = 5.0
        subsidy_upfront_chf: float = 0.0   # NEU


    energy_prices: EnergyPrices
    annual_energy: AnnualEnergyCfg
    storage_effect: StorageEffectCfg
    economics: EconomicsCfg

    def to_domain(self):
        ep = self.energy_prices
        ae = self.annual_energy
        se = self.storage_effect
        ec = self.economics

        # Optionaler Plausibilitätscheck: P ≈ EV + Rückspeisung
        if ae.feed_in_no_storage_kWh is not None:
            calc_feed = ae.pv_gen_kWh - ae.self_consumption_no_storage_kWh
            if abs(calc_feed - ae.feed_in_no_storage_kWh) > 1e-6:
                raise ValueError(
                    f"Plausibilitätsfehler: pv_gen_kWh ({ae.pv_gen_kWh}) "
                    f"!= self_consumption_no_storage_kWh ({ae.self_consumption_no_storage_kWh}) "
                    f"+ feed_in_no_storage_kWh ({ae.feed_in_no_storage_kWh}). "
                    f"Berechnet: {calc_feed}."
                )

        return (
            Prices(ep.import_chf_per_kWh, ep.grid_chf_per_kWh, ep.feed_in_chf_per_kWh),
            AnnualEnergy(ae.pv_gen_kWh, ae.load_kWh, ae.self_consumption_no_storage_kWh),
            StorageEffect(se.evq_with_storage, se.delta_ev_abs_kWh, se.delta_ev_rel, se.improvement_decay_pct_per_year),
            EconParams(ec.capex_battery_chf, ec.capex_installation_chf, ec.opex_annual_chf, ec.lifetime_years, ec.discount_rate_pct, ec.subsidy_upfront_chf),
        )

def load_config(path: str) -> "Config":
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
