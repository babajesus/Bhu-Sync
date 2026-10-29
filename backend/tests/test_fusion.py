from app.engine import World
from app import fusion

def test_source_adapters():
    w = World()
    ac = fusion.aircraft_adapter(w)
    cr = fusion.crew_adapter(w)
    wx = fusion.weather_adapter(w)
    ai = fusion.airspace_adapter(w)
    ta = fusion.task_adapter(w)
    rk = fusion.risk_adapter(w)

    assert len(ac) == 24
    assert len(cr) == 36
    assert len(wx) == 8
    assert len(ai) == 8
    assert len(ta) == 18
    assert len(rk) == 8

    # Validate schema of adapted records
    assert "tail" in ac[0] and "rdy_pct" in ac[0] and "fuel_pct" in ac[0]
    assert "crew_id" in cr[0] and "on_duty" in cr[0]
    assert "zone" in wx[0] and "severity_pct" in wx[0]
    assert "sector" in ai[0] and "state" in ai[0]
    assert "task_no" in ta[0] and "due_h" in ta[0]
    assert "zone" in rk[0] and "risk_pct" in rk[0]

def test_fusion_normalisation_and_pipeline():
    w = World()
    res = fusion.fuse(w)
    assert len(res["sources"]) == 6
    for s in res["sources"]:
        assert s["status"] == "ONLINE"
        assert s["synthetic"] is True
        assert s["records"] > 0

    assert res["mismatches"] == 0
    assert res["fused_ms"] >= 0
    assert res["pipeline"] == [
        "SOURCE DATA",
        "NORMALIZATION",
        "DATA FUSION",
        "OPERATIONAL STATE",
        "OPTIMIZATION"
    ]
    assert res["operational_state"]["aircraft"] == 24
    assert res["operational_state"]["crew"] == 36

def test_fusion_mismatch_detection():
    w = World()
    # Artificially alter internal aircraft readiness after adapter would read
    res = fusion.fuse(w)
    assert res["mismatches"] == 0
