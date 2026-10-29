from app.engine import World

def test_full_demo_scenario():
    w = World(seed=42)
    demo_result = w.run_demo()

    assert "initial" in demo_result
    assert "steps" in demo_result
    assert "final" in demo_result
    assert "kpi_before" in demo_result
    assert "kpi_after" in demo_result

    # 3 progressive dynamic disruption events
    assert len(demo_result["steps"]) == 3
    s1, s2, s3 = demo_result["steps"]

    # Step 1: aircraft unavailable
    assert "unavailable" in s1["event"]
    assert s1["changes"]

    # Step 2: weather event
    assert "Weather" in s2["event"]

    # Step 3: new high priority task
    assert "New priority-5 task" in s3["event"]
    assert any(c["task"].startswith("T-") for c in s3["changes"])

    # Audit log verification
    assert len(w.audit) == 4
    for a in w.audit:
        assert a["status"] in ("APPLIED", "NO FEASIBLE")
        assert a["score"] > 0
        assert a["ts"]

def test_demo_reproducibility():
    w1 = World(seed=99)
    d1 = w1.run_demo()

    w2 = World(seed=99)
    d2 = w2.run_demo()

    assert d1["initial"]["score"] == d2["initial"]["score"]
    assert d1["final"]["score"] == d2["final"]["score"]
    assert len(d1["steps"]) == len(d2["steps"])
