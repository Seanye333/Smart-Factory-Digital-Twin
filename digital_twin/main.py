"""
Smart Factory Digital Twin — Main Entry Point
===============================================
Modes:
  dashboard   → launches Streamlit live dashboard
  api         → launches FastAPI REST server
  simulate    → runs factory line simulation + prints KPIs to console
  demo        → runs all three components concurrently

Usage:
  python digital_twin/main.py                    # defaults to demo mode
  python digital_twin/main.py --mode dashboard
  python digital_twin/main.py --mode api
  python digital_twin/main.py --mode simulate --duration 300
"""

import argparse
import subprocess
import sys
import time
import threading
import logging
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table   import Table
from rich.live    import Live
from rich.panel   import Panel
from rich         import box

from ingestion.mock_plc      import MockPLCGenerator, MACHINE_PROFILES
from ml.failure_predictor    import FailurePredictor
from ml.anomaly_detector     import AnomalyDetector
from optimization.bottleneck import BottleneckAnalyzer
from simulation.factory_line import FactoryLine
from core.events             import event_bus, EventType, Event

console = Console()
logging.basicConfig(level=logging.WARNING)


# --------------------------------------------------------------------------
# Console simulation mode
# --------------------------------------------------------------------------

def _make_status_table(readings: dict, predictor: FailurePredictor,
                        detector: AnomalyDetector, tick: int) -> Table:
    table = Table(
        title=f"🏭 Smart Factory Digital Twin — Tick #{tick}",
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Machine",     style="bold", width=10)
    table.add_column("Type",        width=10)
    table.add_column("Status",      width=10)
    table.add_column("Temp (°C)",   justify="right", width=10)
    table.add_column("Vib (mm/s)",  justify="right", width=11)
    table.add_column("Current (A)", justify="right", width=12)
    table.add_column("Fail Risk",   justify="right", width=11)
    table.add_column("Anomaly",     justify="center", width=9)
    table.add_column("Parts",       justify="right", width=8)

    for mid, r in readings.items():
        fp      = predictor.predict_failure_probability(r.to_dict(), mid)
        anomaly = detector.detect(r.to_dict(), mid)

        status = "[red]FAULT[/red]"   if r.fault_code > 0 \
            else "[blue]IDLE[/blue]"  if r.speed == 0 \
            else "[green]RUN[/green]"

        risk_str = f"{fp*100:.1f}%"
        if fp > 0.7:
            risk_str = f"[red]{risk_str}[/red]"
        elif fp > 0.4:
            risk_str = f"[yellow]{risk_str}[/yellow]"
        else:
            risk_str = f"[green]{risk_str}[/green]"

        anom_str = "[red]YES[/red]" if anomaly else "[green]no[/green]"

        table.add_row(
            mid,
            MACHINE_PROFILES[mid].machine_type,
            status,
            f"{r.temperature:.1f}",
            f"{r.vibration:.3f}",
            f"{r.current:.1f}",
            risk_str,
            anom_str,
            str(r.parts_produced),
        )

    return table


def run_simulation(duration: float = 120.0, interval: float = 2.0) -> None:
    """Run simulation in console mode with live Rich table."""
    plc       = MockPLCGenerator()
    predictor = FailurePredictor()
    detector  = AnomalyDetector()
    analyzer  = BottleneckAnalyzer(list(MACHINE_PROFILES.keys()))

    # Factory line sim at 60× speed
    line = FactoryLine(sim_speed=60.0)
    line.start()

    tick     = 0
    end_time = time.time() + duration

    console.print(Panel.fit(
        "[bold cyan]Smart Factory Digital Twin[/bold cyan]\n"
        f"[dim]Running for {duration:.0f}s | Refresh every {interval}s[/dim]",
        border_style="cyan",
    ))

    # Subscribe to critical events
    def on_fault(event: Event):
        console.print(f"[red]⚠ FAULT[/red] {event.source} at sim-time {event.payload.get('fault_time', '?'):.0f}")

    event_bus.subscribe(EventType.MACHINE_FAULT, on_fault)

    with Live(console=console, refresh_per_second=1) as live:
        while time.time() < end_time:
            tick    += 1
            readings = plc.generate_all()
            table    = _make_status_table(readings, predictor, detector, tick)

            # Bottleneck check
            stats = [{"machine_id": m, "cycle_time": r.cycle_time,
                       "uptime_pct": 75.0, "queue_length": 0}
                     for m, r in readings.items()]
            bn = analyzer.find_bottleneck(stats)
            bn_text = (f"\n[yellow]Bottleneck: {bn['bottleneck_id']} "
                       f"(score={bn['score']:.2f})[/yellow]") if bn else ""

            # Line KPIs
            line_kpi = line.get_line_summary()
            kpi_text = (f"\n[cyan]Output: {line_kpi['output_good']} good / "
                        f"{line_kpi['output_total']} total  "
                        f"Yield: {line_kpi['yield_pct']}%  "
                        f"SimTime: {line_kpi['sim_time']:.0f}s[/cyan]")

            live.update(Panel(
                table.__rich_console__,
                subtitle=bn_text + kpi_text,
                border_style="blue",
            ))
            time.sleep(interval)

    line.stop()
    console.print("[green]✅ Simulation complete[/green]")


# --------------------------------------------------------------------------
# API mode
# --------------------------------------------------------------------------

def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    console.print(f"[cyan]Starting FastAPI server → http://{host}:{port}[/cyan]")
    console.print(f"[dim]Swagger UI → http://{host}:{port}/docs[/dim]")
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="warning",
    )


# --------------------------------------------------------------------------
# Dashboard mode
# --------------------------------------------------------------------------

def run_dashboard() -> None:
    dashboard_path = str(ROOT / "dashboard" / "app.py")
    console.print(f"[cyan]Launching Streamlit dashboard …[/cyan]")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.runOnSave=true",
        "--theme.base=dark",
    ])


# --------------------------------------------------------------------------
# Demo mode: all three in parallel
# --------------------------------------------------------------------------

def run_demo() -> None:
    console.print(Panel.fit(
        "[bold cyan]Smart Factory Digital Twin — Demo Mode[/bold cyan]\n\n"
        "[green]→ Dashboard[/green] : streamlit run digital_twin/dashboard/app.py\n"
        "[green]→ API[/green]       : uvicorn digital_twin.api.server:app --port 8000\n"
        "[green]→ Simulation[/green]: running in console\n\n"
        "[dim]Launching dashboard + API in background …[/dim]",
        border_style="cyan",
    ))

    # API in background thread
    api_thread = threading.Thread(
        target=run_api, daemon=True, name="api-server"
    )
    api_thread.start()
    time.sleep(1.5)

    # Dashboard in subprocess
    dash_thread = threading.Thread(
        target=run_dashboard, daemon=True, name="dashboard"
    )
    dash_thread.start()
    time.sleep(2.0)

    # Simulation in foreground
    run_simulation(duration=float("inf"), interval=2.0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Factory Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "api", "simulate", "demo"],
        default="demo",
        help="Operating mode (default: demo)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Simulation duration in seconds (simulate mode, default 300)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API server host (default 0.0.0.0)",
    )

    args = parser.parse_args()

    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "api":
        run_api(args.host, args.port)
    elif args.mode == "simulate":
        run_simulation(duration=args.duration)
    else:
        run_demo()


if __name__ == "__main__":
    main()
