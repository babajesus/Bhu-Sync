import os
os.environ["DB_PATH"] = ":memory:"
from fastapi.testclient import TestClient
from app.main import app, store

c = TestClient(app)
A = "/api/v1"

def test_health():
    res = c.get(f"{A}/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["synthetic_data"] is True

def test_resources_tasks_crew():
    assert len(c.get(f"{A}/resources").json()) == 24
    assert len(c.get(f"{A}/tasks").json()) >= 15
    assert len(c.get(f"{A}/crew").json()) == 36
    assert c.get(f"{A}/resources/A-01").json()["id"] == "A-01"
    assert c.get(f"{A}/resources/ZZ").status_code == 404

def test_summary_and_read_routes():
    s = c.get(f"{A}/dashboard/summary").json()
    assert s["ready"] <= s["available"] <= s["resources"]
    for u in (
        "weather",
        "airspace",
        "risks",
        "recommendations",
        "analytics/readiness",
        "analytics/anomalies",
        "fusion",
        "events",
        "optimization/latest"
    ):
        assert c.get(f"{A}/{u}").status_code == 200

def test_run_with_weights_compare_and_persist():
    r = c.post(
        f"{A}/optimization/run",
        json=dict(priority=1, readiness=0, availability=0, deadline=0, risk=0, weather=0)
    ).json()
    assert "changed" in r["compare"]
    assert store.history()

def test_event_invalid_and_valid():
    assert c.post(f"{A}/events", json={"kind": "bogus"}).status_code == 422
    assert c.post(f"{A}/events", json={"kind": "weather"}).json()["result"]["assignments"] is not None

def test_sim_and_demo():
    for a in ("start", "pause", "reset"):
        assert c.post(f"{A}/simulation/{a}").status_code == 200
    d = c.post(f"{A}/demo/run").json()
    assert len(d["steps"]) == 3
    audit_data = c.get(f"{A}/audit").json()
    assert audit_data[0]["steps"]

def test_recommendation_by_id():
    tasks = c.get(f"{A}/tasks").json()
    first_task = tasks[0]["id"]
    r = c.get(f"{A}/recommendations/{first_task}")
    assert r.status_code in (200, 404)
    # Check invalid task returns 404
    assert c.get(f"{A}/recommendations/NONEXISTENT").status_code == 404

def test_index_page():
    res = c.get("/")
    assert res.status_code == 200
