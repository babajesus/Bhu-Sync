"""Synthetic source adapters -> normalisation -> fused operational state (all SYNTHETIC)."""
import time

def aircraft_adapter(w):
    return [
        dict(
            tail=a["id"],
            cls=a["type"],
            ops=a["status"],
            rdy_pct=int(a["readiness"] * 100),
            loc=a["region"],
            fuel_pct=int(a["fuel"] * 100),
            mx=a["maintenance"]
        )
        for a in w.aircraft
    ]

def crew_adapter(w):
    return [
        dict(
            crew_id=c["id"],
            ac=c["aircraft"],
            on_duty=int(c["available"]),
            fatigue_idx=int(c["fatigue"] * 100),
            rdy_pct=int(c["readiness"] * 100)
        )
        for c in w.crew
    ]

def weather_adapter(w):
    return [dict(zone=k, severity_pct=int(v * 100)) for k, v in w.weather.items()]

def airspace_adapter(w):
    return [dict(sector=k, state=v.upper()) for k, v in w.airspace.items()]

def task_adapter(w):
    return [
        dict(
            task_no=t["id"],
            cat=t["category"],
            prio=t["priority"],
            zone=t["region"],
            due_h=t["deadline"],
            dur_h=t["duration"]
        )
        for t in w.tasks
    ]

def risk_adapter(w):
    return [dict(zone=k, risk_pct=int(v * 100)) for k, v in w.risk.items()]

def fuse(w):
    t0 = time.time()
    raw = {
        "Aircraft System": aircraft_adapter(w),
        "Crew System": crew_adapter(w),
        "Weather Feed": weather_adapter(w),
        "Airspace Feed": airspace_adapter(w),
        "Tasking System": task_adapter(w),
        "Risk Feed": risk_adapter(w)
    }
    ac = [
        dict(
            id=r["tail"],
            type=r["cls"],
            status=r["ops"],
            readiness=r["rdy_pct"] / 100.0,
            region=r["loc"],
            fuel=r["fuel_pct"] / 100.0,
            maintenance=r["mx"]
        )
        for r in raw["Aircraft System"]
    ]
    cr = [
        dict(
            id=r["crew_id"],
            aircraft=r["ac"],
            available=bool(r["on_duty"]),
            fatigue=r["fatigue_idx"] / 100.0
        )
        for r in raw["Crew System"]
    ]
    state = dict(
        aircraft=ac,
        crew=cr,
        weather={r["zone"]: r["severity_pct"] / 100.0 for r in raw["Weather Feed"]},
        airspace={r["sector"]: r["state"].lower() for r in raw["Airspace Feed"]},
        tasks=[dict(id=r["task_no"], priority=r["prio"], region=r["zone"]) for r in raw["Tasking System"]],
        risk={r["zone"]: r["risk_pct"] / 100.0 for r in raw["Risk Feed"]}
    )
    mism = sum(abs(a["readiness"] - w.aircraft_by(a["id"])["readiness"]) > 0.01 for a in ac)
    return dict(
        sources=[dict(name=k, status="ONLINE", records=len(v), synthetic=True) for k, v in raw.items()],
        operational_state={k: len(v) for k, v in state.items()},
        mismatches=mism,
        fused_ms=round((time.time() - t0) * 1000, 2),
        pipeline=["SOURCE DATA", "NORMALIZATION", "DATA FUSION", "OPERATIONAL STATE", "OPTIMIZATION"]
    )
