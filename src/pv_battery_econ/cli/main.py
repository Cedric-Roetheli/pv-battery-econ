import click
import json
from pv_battery_econ.config import load_config
from pv_battery_econ.economics.cashflow import evaluate

@click.group(help="PV-Batterie Wirtschaftlichkeit")
def cli():
    pass

@cli.command(name="hello")
def hello_cmd():
    click.echo("pv-battery-econ CLI läuft.")

@cli.command(name="run")
@click.option("--config", "config_path", required=True, help="Pfad zur YAML-Config")
def run_cmd(config_path: str):
    cfg = load_config(config_path)
    prices, energy, effect, econ = cfg.to_domain()
    res = evaluate(energy, prices, effect, econ)
    click.echo(json.dumps(res, indent=2, ensure_ascii=False))

def main():
    cli()

if __name__ == "__main__":
    main()
