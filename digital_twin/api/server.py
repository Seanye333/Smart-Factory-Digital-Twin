"""
Digital Twin REST API — FastAPI server
=======================================
Exposes the factory simulation state, sensor data, ML predictions,
bottleneck analysis and scheduling via a clean JSON API.

Run with:   uvicorn digital_twin.api.server:app --reload --port 8000
Or via:     python main.py --mode api

Endpoints:
  GET  /                       — health check
  GET  /machines               — list all machines + current state
  GET  /machines/{id}          — single machine detail
  GET  /machines/{id}/sensors  — latest sensor reading
  GET  /machines/{id}/history  — recent sensor history
  GET  /predictions            — failure probability for all machines
  GET  /predictions/{id}       — single machine failure prediction
  GET  /anomalies              — anomaly detection results
  GET  /bottleneck             — bottleneck analysis
  GET  /schedule               — current production schedule
  GET  /line/summary           — overall factory line KPIs
  POST /machines/{id}/fault    — inject a fault (testing)
  POST /machines/{id}/clear    — clear a fault
  GET  /events                 — recent event bus events
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingestion.mock_plc      import MockPLCGenerator, MACHINE_PROFILES
from ml.failure_predictor    import FailurePredictor
from ml.anomaly_detector     import AnomalyDetector
from optimization.bottleneck import BottleneckAnalyzer
from optimization.scheduler  import ProductionScheduler
from core.events             import event_bus, EventType

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------

app = FastAPI(
    title="Smart Factory Digital Twin API",
    description="Real-time factory simulation, predictive maintenance & optimisation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Singleton components (initialised on first startup)
# --------------------------------------------------------------------------

_plc:       Optional[MockPLCGenerator]  = None
_predictor: Optional[FailurePredictor]  = None
_detector:  Optional[AnomalyDetector]   = None
_analyzer:  Optional[BottleneckAnalyzer] = None
_scheduler: Optional[ProductionScheduler] = None
_latest_readings: Dict[str, Dict] = {}
_history: Dict[str, List[Dict]]  = {m: [] for m in MACHINE_PROFILES}
HISTORY_LEN = 120


def get_plc() -> MockPLCGenerator:
    global _plc
    if _plc is None:
        _plc = MockPLCGenerator()
        _plc.register_callback(_on_new_readings)
    return _plc


def get_predictor() -> FailurePredictor:
    global _predictor
    if _predictor is None:
        _predictor = FailurePredictor()
    return _predictor


def get_detector() -> AnomalyDetector:
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector


def get_analyzer() -> BottleneckAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BottleneckAnalyzer(list(MACHINE_PROFILES.keys()))
    return _analyzer


def get_scheduler() -> ProductionScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ProductionScheduler()
    return _scheduler


def _on_new_readings(readings: Dict) -> None:
    """Callback from MockPLC streaming thread."""
    global _latest_readings
    for mid, reading in readings.items():
        d = reading.to_dict()
        _latest_readings[mid] = d
        buf = _history.setdefault(mid, [])
        buf.append(d)
        if len(buf) > HISTORY_LEN:
            buf.pop(0)


# --------------------------------------------------------------------------
# Startup / shutdown
# --------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    plc = get_plc()
    plc.start_streaming(interval=2.0)
    # Warm up ML models
    get_predictor()
    get_detector()


# --------------------------------------------------------------------------
# Request / Response models
# --------------------------------------------------------------------------

class FaultRequest(BaseModel):
    fault_code: Optional[int] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _current_readings() -> Dict[str, Dict]:
    if not _latest_readings:
        return {mid: get_plc().generate_reading(mid).to_dict() for mid in MACHINE_PROFILES}
    return _latest_readings


def _machine_exists(machine_id: str) -> None:
    if machine_id not in MACHINE_PROFILES:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health() -> Dict:
    return {
        "status":    "ok",
        "service":   "Smart Factory Digital Twin API",
        "version":   "1.0.0",
        "timestamp": time.time(),
        "machines":  list(MACHINE_PROFILES.keys()),
    }


@app.get("/machines", tags=["Machines"])
def list_machines() -> List[Dict]:
    readings = _current_readings()
    result   = []
    for mid, profile in MACHINE_PROFILES.items():
        r = readings.get(mid, {})
        result.append({
            "machine_id":   mid,
            "machine_type": profile.machine_type,
            "status":       "FAULT" if r.get("fault_code", 0) > 0
                            else "IDLE" if r.get("speed", 0) == 0
                            else "RUNNING",
            "fault_code":   r.get("fault_code", 0),
            "degradation":  round(get_plc().get_degradation_states().get(mid, 0), 3),
            "is_faulted":   get_plc().get_fault_states().get(mid, False),
        })
    return result


@app.get("/machines/{machine_id}", tags=["Machines"])
def get_machine(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    readings = _current_readings()
    r        = readings.get(machine_id, {})
    profile  = MACHINE_PROFILES[machine_id]

    return {
        "machine_id":    machine_id,
        "machine_type":  profile.machine_type,
        "nominal_cycle": profile.nominal_cycle_time,
        "sensor":        r,
        "degradation":   round(get_plc().get_degradation_states().get(machine_id, 0), 3),
        "is_faulted":    get_plc().get_fault_states().get(machine_id, False),
    }


@app.get("/machines/{machine_id}/sensors", tags=["Machines"])
def get_sensors(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    readings = _current_readings()
    return readings.get(machine_id, {})


@app.get("/machines/{machine_id}/history", tags=["Machines"])
def get_history(machine_id: str, limit: int = 60) -> List[Dict]:
    _machine_exists(machine_id)
    buf = _history.get(machine_id, [])
    return buf[-limit:]


@app.get("/predictions", tags=["ML / Predictions"])
def all_predictions() -> List[Dict]:
    readings = _current_readings()
    pred     = get_predictor()
    return [
        pred.predict(r, mid).to_dict()
        for mid, r in readings.items()
    ]


@app.get("/predictions/{machine_id}", tags=["ML / Predictions"])
def machine_prediction(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    readings = _current_readings()
    r        = readings.get(machine_id, {})
    return get_predictor().predict(r, machine_id).to_dict()


@app.get("/anomalies", tags=["ML / Predictions"])
def all_anomalies() -> List[Dict]:
    readings = _current_readings()
    det      = get_detector()
    return [
        det.detect_full(r, mid).to_dict()
        for mid, r in readings.items()
    ]


@app.get("/anomalies/{machine_id}", tags=["ML / Predictions"])
def machine_anomaly(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    readings = _current_readings()
    r        = readings.get(machine_id, {})
    return get_detector().detect_full(r, machine_id).to_dict()


@app.get("/bottleneck", tags=["Optimisation"])
def bottleneck_analysis() -> Dict:
    readings = _current_readings()
    stats    = [
        {
            "machine_id":   mid,
            "cycle_time":   r.get("cycle_time", 0),
            "uptime_pct":   70 + 20 * (1 - get_plc().get_degradation_states().get(mid, 0)),
            "queue_length": 0,
        }
        for mid, r in readings.items()
    ]
    result = get_analyzer().find_bottleneck(stats)
    return result or {"message": "No bottleneck detected"}


@app.get("/schedule", tags=["Optimisation"])
def production_schedule(algorithm: str = "EDD", n_jobs: int = 10) -> Dict:
    sched_engine = get_scheduler()
    jobs         = sched_engine.generate_sample_jobs(n_jobs)

    algo = algorithm.upper()
    if algo == "EDD":
        schedule = sched_engine.schedule_edd(jobs)
    elif algo == "SPT":
        schedule = sched_engine.schedule_spt(jobs)
    elif algo == "WSJF":
        schedule = sched_engine.schedule_wsjf(jobs)
    elif algo == "GA":
        schedule = sched_engine.schedule_genetic(jobs)
    elif algo == "COMPARE":
        return sched_engine.compare_algorithms(jobs)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}. Use EDD|SPT|WSJF|GA|COMPARE")

    return schedule.to_dict()


@app.get("/line/summary", tags=["Factory Line"])
def line_summary() -> Dict:
    readings   = _current_readings()
    deg_states = get_plc().get_degradation_states()
    flt_states = get_plc().get_fault_states()

    running = sum(1 for r in readings.values() if r.get("fault_code", 0) == 0 and r.get("speed", 0) > 0)
    faulted = sum(1 for v in flt_states.values() if v)
    avg_deg = round(sum(deg_states.values()) / max(1, len(deg_states)), 4)
    throughput = sum(
        3600 / max(1, r.get("cycle_time", 1))
        for r in readings.values()
        if r.get("cycle_time", 0) > 0
    )

    return {
        "timestamp":       time.time(),
        "machines_total":  len(MACHINE_PROFILES),
        "machines_running": running,
        "machines_faulted": faulted,
        "avg_degradation":  avg_deg,
        "throughput_pph":   round(throughput, 2),
        "factory_oee_est":  round(max(55.0, 91.0 - avg_deg * 30), 2),
    }


@app.get("/events", tags=["Events"])
def recent_events(limit: int = 50, event_type: Optional[str] = None) -> List[Dict]:
    etype = None
    if event_type:
        try:
            etype = EventType[event_type.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown event type: {event_type}")

    events = event_bus.get_recent(event_type=etype, limit=limit)
    return [
        {
            "type":      e.type.value,
            "source":    e.source,
            "severity":  e.severity,
            "payload":   e.payload,
            "timestamp": e.timestamp,
        }
        for e in events
    ]


@app.post("/machines/{machine_id}/fault", tags=["Control"])
def inject_fault(machine_id: str, body: FaultRequest = FaultRequest()) -> Dict:
    _machine_exists(machine_id)
    get_plc().inject_fault(machine_id, body.fault_code)
    return {"status": "fault_injected", "machine_id": machine_id, "fault_code": body.fault_code}


@app.post("/machines/{machine_id}/clear", tags=["Control"])
def clear_fault(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    get_plc().clear_fault(machine_id)
    return {"status": "fault_cleared", "machine_id": machine_id}


@app.post("/machines/{machine_id}/stop", tags=["Control"])
def stop_machine(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    get_plc().set_running(machine_id, False)
    return {"status": "stopped", "machine_id": machine_id}


@app.post("/machines/{machine_id}/start", tags=["Control"])
def start_machine(machine_id: str) -> Dict:
    _machine_exists(machine_id)
    get_plc().set_running(machine_id, True)
    return {"status": "started", "machine_id": machine_id}


# --------------------------------------------------------------------------
# Entry point (optional — prefer uvicorn CLI)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("digital_twin.api.server:app", host="0.0.0.0", port=8000, reload=False)
