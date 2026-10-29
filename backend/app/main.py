"""AIRSYNC FastAPI Application Entry Point.
Synthetic High-Level Decision Support & Resource Optimisation Demo.
"""
import os
import time
import random
import threading
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.engine import World
from app import fusion
from app.db import Store

# Initialize FastAPI App
app = FastAPI(
    title="AIRSYNC - AI Air Operations Decision Support & Resource Optimisation",
    description="Fictional Synthetic High-Level Decision Support Demonstration System. All data synthetic.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent Store & Engine State
db_path = os.getenv("DB_PATH", "airsync.db")
store = Store(db_path)
world = World()
world.hook = store.save

# Initial baseline run so world has starting state
_init_res = world.optimize(trigger="system_init")
world.assignments = {x["task"]: x for x in _init_res["assignments"]}
world.last = _init_res
world._log(dict(
    ts=time.strftime("%H:%M:%S"),
    event="System startup & baseline optimization",
    changes=[],
    score=_init_res["score"],
    status="APPLIED",
    steps=[
        "AIRSYNC system online",
        "Synthetic source feeds normalized",
        "Data fusion pipeline initialized",
        "Baseline linear sum assignment executed",
        "Decision recorded"
    ]
))


class SimulationService:
    def __init__(self, w: World):
        self.world = w
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if not self.running:
                self.running = True
                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()

    def pause(self):
        with self._lock:
            self.running = False

    def reset(self):
        self.pause()
        self.world.reset()
        res = self.world.optimize(trigger="sim_reset")
        self.world.assignments = {x["task"]: x for x in res["assignments"]}
        self.world.last = res
        self.world._log(dict(
            ts=time.strftime("%H:%M:%S"),
            event="Simulation reset to baseline",
            changes=[],
            score=res["score"],
            status="APPLIED",
            steps=["World state reset", "Data feeds re-synchronized", "Optimizer re-executed", "Baseline restored"]
        ))

    def _run_loop(self):
        event_types = ["weather", "airspace", "crew_unavailable", "maintenance", "priority", "new_task"]
        while self.running:
            time.sleep(5)
            if not self.running:
                break
            try:
                ev = random.choice(event_types)
                self.world.apply_event(ev)
            except Exception:
                pass


sim_manager = SimulationService(world)

# Pydantic Request Models
class WeightsModel(BaseModel):
    priority: Optional[float] = Field(None, ge=0.0, le=1.0)
    readiness: Optional[float] = Field(None, ge=0.0, le=1.0)
    availability: Optional[float] = Field(None, ge=0.0, le=1.0)
    deadline: Optional[float] = Field(None, ge=0.0, le=1.0)
    risk: Optional[float] = Field(None, ge=0.0, le=1.0)
    weather: Optional[float] = Field(None, ge=0.0, le=1.0)

class EventModel(BaseModel):
    kind: Literal[
        "aircraft_unavailable",
        "weather",
        "airspace",
        "crew_unavailable",
        "maintenance",
        "priority",
        "new_task"
    ]
    id: Optional[str] = None
    region: Optional[str] = None


# Static file serving
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=FileResponse)
def index_page():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # Fallback to local root index if static not present
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="Dashboard index.html not found")


# API Routes (/api/v1)
@app.get("/api/v1/health")
def get_health():
    return {
        "status": "ok",
        "service": "AIRSYNC",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic_data": True,
        "environment": "FICTIONAL_DEMO",
        "simulation": sim_manager.running
    }

@app.get("/api/v1/dashboard/summary")
def get_summary():
    total_ac = len(world.aircraft)
    avail_ac = sum(1 for a in world.aircraft if a["status"] == "available")
    ready_ac = sum(1 for a in world.aircraft if a["status"] == "available" and a["maintenance"] != "blocked" and a["readiness"] >= 0.70)
    # Guarantee mathematical order: ready <= available <= resources
    ready_ac = min(ready_ac, avail_ac)
    avail_ac = min(avail_ac, total_ac)

    active_tasks = len(world.tasks)
    assigned_tasks = len(world.assignments)
    unassigned_tasks = len(world.last["unassigned"]) if world.last else 0
    conflicts_list = world.conflicts()

    return {
        "resources": total_ac,
        "available": avail_ac,
        "ready": ready_ac,
        "tasks": active_tasks,
        "assigned": assigned_tasks,
        "unassigned": unassigned_tasks,
        "conflicts": len(conflicts_list),
        "score": world.last["score"] if world.last else 0.0,
        "utilization": world.last["kpi"]["utilization"] if world.last else 0.0,
        "simulation": sim_manager.running,
        "history": list(world.history)
    }

@app.get("/api/v1/resources")
def get_resources():
    return world.aircraft

@app.get("/api/v1/resources/{id}")
def get_resource(id: str):
    for a in world.aircraft:
        if a["id"] == id:
            return a
    raise HTTPException(status_code=404, detail=f"Resource {id} not found")

@app.get("/api/v1/crew")
def get_crew():
    return world.crew

@app.get("/api/v1/tasks")
def get_tasks():
    return world.tasks

@app.get("/api/v1/recommendations")
def get_recommendations():
    return list(world.assignments.values())

@app.get("/api/v1/recommendations/{id}")
def get_recommendation(id: str):
    if id in world.assignments:
        return world.assignments[id]
    raise HTTPException(status_code=404, detail=f"No active recommendation for task {id}")

@app.get("/api/v1/audit")
def get_audit():
    # Return stored history or in-memory audit
    hist = store.history(50)
    if hist:
        return hist
    return world.audit

@app.get("/api/v1/analytics/readiness")
def get_analytics_readiness():
    return world.predictions()

@app.get("/api/v1/analytics/anomalies")
def get_analytics_anomalies():
    return world.anomalies()

@app.get("/api/v1/weather")
def get_weather():
    return world.weather

@app.get("/api/v1/airspace")
def get_airspace():
    return world.airspace

@app.get("/api/v1/risks")
def get_risks():
    return world.risk

@app.get("/api/v1/fusion")
def get_fusion():
    return fusion.fuse(world)

@app.get("/api/v1/events")
def get_events():
    return world.events

@app.get("/api/v1/optimization/latest")
def get_latest_optimization():
    if not world.last:
        res = world.optimize(trigger="query_latest")
        world.assignments = {x["task"]: x for x in res["assignments"]}
        world.last = res
    return world.last

@app.post("/api/v1/optimization/run")
def run_optimization(weights: Optional[WeightsModel] = None):
    if weights:
        w_dict = weights.model_dump(exclude_unset=True)
        world.weights.update(w_dict)

    before_alloc = {k: v["aircraft"] for k, v in world.assignments.items()}
    before_kpi = dict(
        score=world.last["score"] if world.last else 0.0,
        utilization=world.last["kpi"]["utilization"] if world.last else 0.0,
        unassigned=len(world.last["unassigned"]) if world.last else 0,
        assigned=len(world.assignments)
    )

    res = world.optimize(trigger="manual_weights_update")
    world.assignments = {x["task"]: x for x in res["assignments"]}
    world.last = res

    after_alloc = {k: v["aircraft"] for k, v in world.assignments.items()}
    after_kpi = dict(
        score=res["score"],
        utilization=res["kpi"]["utilization"],
        unassigned=len(res["unassigned"]),
        assigned=len(res["assignments"])
    )

    changed = [
        {"task": k, "before": before_alloc.get(k), "after": after_alloc.get(k)}
        for k in sorted(set(before_alloc) | set(after_alloc))
        if before_alloc.get(k) != after_alloc.get(k)
    ]
    for ch in changed:
        ch["reason"] = (
            world.assignments[ch["task"]]["reason"]
            if ch["task"] in world.assignments
            else ["No feasible allocation"]
        )

    ts = time.strftime("%H:%M:%S")
    rec = dict(
        ts=ts,
        event="Optimization executed with custom weights",
        changes=changed,
        score=res["score"],
        prev_score=before_kpi["score"],
        conflicts_detected=world.conflicts(),
        steps=[
            "Custom weights applied",
            f"Hungarian algorithm optimized {len(res['assignments'])} task allocations",
            f"{len(changed)} reallocations occurred",
            "Decision recorded"
        ],
        status="APPLIED"
    )
    world._log(rec)

    return {
        "compare": {
            "before": before_kpi,
            "after": after_kpi,
            "changed": changed
        },
        "result": res
    }

@app.post("/api/v1/events")
def post_event(ev: EventModel):
    try:
        kwargs = {}
        if ev.id:
            kwargs["id"] = ev.id
        if ev.region:
            kwargs["region"] = ev.region
        res = world.apply_event(ev.kind, **kwargs)
        return res
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/api/v1/demo/run")
def post_demo_run():
    return world.run_demo()

@app.post("/api/v1/simulation/start")
def post_simulation_start():
    sim_manager.start()
    return {"simulation": True, "status": "LIVE"}

@app.post("/api/v1/simulation/pause")
def post_simulation_pause():
    sim_manager.pause()
    return {"simulation": False, "status": "PAUSED"}

@app.post("/api/v1/simulation/reset")
def post_simulation_reset():
    sim_manager.reset()
    return {"simulation": False, "status": "RESET", "score": world.last["score"]}
