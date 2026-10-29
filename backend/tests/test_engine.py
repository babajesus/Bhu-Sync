from app.engine import World
from app import fusion

def fresh():
    w = World()
    w.last = w.optimize()
    w.assignments = {a["task"]: a for a in w.last["assignments"]}
    return w

def test_optimizer():
    r = fresh().last
    assert r["assignments"]
    assert r["score"] > 0
    assert all(a["reason"] and 0 < a["confidence"] <= 1 and a["constraints"] for a in r["assignments"])

def test_hard_constraints():
    w = World()
    for a in w.optimize()["assignments"]:
        t = next(t for t in w.tasks if t["id"] == a["task"])
        ac = w.aircraft_by(a["aircraft"])
        assert ac["type"] == t["needs"]
        assert ac["status"] == "available"
        assert ac["maintenance"] != "blocked"
        assert w.airspace[t["region"]] != "restricted"
        assert a["eta"] <= t["deadline"]

def test_no_double_booking():
    r = World().optimize()["assignments"]
    assert len({a["aircraft"] for a in r}) == len(r) == len({a["task"] for a in r})

def test_weather_constraint():
    w = World()
    w.weather = {k: 0.9 for k in w.weather}
    assert w.optimize()["assignments"] == []

def test_airspace_constraint():
    w = World()
    r0 = w.tasks[0]["region"]
    w.airspace[r0] = "restricted"
    assert not {t["id"] for t in w.tasks if t["region"] == r0} & {a["task"] for a in w.optimize()["assignments"]}

def test_crew_availability():
    w = fresh()
    ac = w.last["assignments"][0]["aircraft"]
    for c in w.crew:
        if c["aircraft"] == ac:
            c["available"] = False
    assert ac not in {a["aircraft"] for a in w.optimize()["assignments"]}

def test_dynamic_retasking_and_audit():
    w = fresh()
    r = w.apply_event("aircraft_unavailable")
    dead = r["event"].split()[0]
    assert r["conflicts_detected"]
    assert dead not in {a["aircraft"] for a in r["result"]["assignments"]}
    assert w.audit[0]["steps"][-1] == "Decision recorded"
    assert w.events
    assert len(w.history) >= 2

def test_all_event_types():
    w = fresh()
    for k in ("weather", "airspace", "crew_unavailable", "maintenance", "priority", "new_task"):
        assert w.apply_event(k)["result"]["score"] >= 0

def test_invalid_event():
    try:
        fresh().apply_event("bogus")
        assert False
    except ValueError:
        pass

def test_no_feasible_allocation():
    w = World()
    for a in w.aircraft:
        a["status"] = "unavailable"
    r = w.optimize()
    assert r["assignments"] == []
    assert all(u["message"] == "No feasible allocation found" and u["blocked_by"] for u in r["unassigned"])

def test_weights_change_result():
    w = World()
    a = w.optimize()["score"]
    w.weights.update(priority=1, risk=0, readiness=0)
    assert w.optimize()["score"] != a

def test_ml_prediction():
    p = World().predictions()
    assert len(p) == 24
    assert all(0 <= x["predicted"] <= 1 and x["risk"] in ("LOW", "MEDIUM", "HIGH") and x["factors"] for x in p)

def test_anomaly_detection():
    w = World()
    w.aircraft[0]["readiness"] = 0.2
    x = [a for a in w.anomalies() if a["id"] == "A-01"]
    assert x
    assert x[0]["severity"]
    assert x[0]["explanation"]

def test_fusion():
    f = fusion.fuse(World())
    assert len(f["sources"]) == 6
    assert all(s["synthetic"] for s in f["sources"])
    assert f["mismatches"] == 0

def test_demo_scenario():
    w = World()
    d = w.run_demo()
    assert len(d["steps"]) == 3
    assert d["final"]["score"] > 0
    assert len(w.audit) == 4
