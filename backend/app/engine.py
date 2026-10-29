"""AIRSYNC core: synthetic data, fusion state, optimizer, ML, anomalies, retasking. All data fictional."""
import random
import time
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestRegressor, IsolationForest

TYPES = ["Transport", "Surveillance", "Utility", "Rotary"]
CATS = {
    "Airlift": "Transport",
    "Recon-Survey": "Surveillance",
    "Patrol": "Surveillance",
    "Logistics": "Utility",
    "Medical-Support": "Rotary",
    "Training": "Utility"
}
REGIONS = {f"R{i}": (10 + (i % 4) * 2.0, 20 + (i // 4) * 2.0) for i in range(8)}  # fictional grid
DEFAULT_W = dict(priority=0.30, readiness=0.20, availability=0.20, deadline=0.15, risk=0.15, weather=0.10)

class World:
    def __init__(self, seed=42, weights=None):
        self.seed = seed
        self.weights = dict(weights or DEFAULT_W)
        self.audit = []
        self.events = []
        self.last = None
        self.hook = None
        self.history = []
        self.reset()

    def reset(self):
        r = random.Random(self.seed)
        self.clock = 0
        self.aircraft = [
            dict(
                id=f"A-{i:02d}",
                type=TYPES[i % 4],
                status="available",
                readiness=round(r.uniform(0.6, 0.98), 2),
                region=r.choice(list(REGIONS)),
                fuel=round(r.uniform(0.5, 1.0), 2),
                maintenance="ok",
                hours=r.randint(200, 3000),
                turnaround=r.choice([1, 2, 3]),
                failures=r.randint(0, 4),
                downtime=round(r.uniform(0, 0.2), 2),
                util=round(r.uniform(0.2, 0.8), 2),
                prev_readiness=0
            )
            for i in range(1, 25)
        ]
        for a in self.aircraft:
            a["prev_readiness"] = a["readiness"]

        self.crew = [
            dict(
                id=f"C-{i:02d}",
                aircraft=f"A-{(i - 1) % 24 + 1:02d}",
                available=True,
                fatigue=round(r.uniform(0.05, 0.6), 2),
                readiness=round(r.uniform(0.7, 1.0), 2)
            )
            for i in range(1, 37)
        ]
        self.weather = {k: round(r.uniform(0, 0.5), 2) for k in REGIONS}  # severity 0-1
        self.airspace = {k: "open" for k in REGIONS}
        self.risk = {k: round(r.uniform(0.05, 0.6), 2) for k in REGIONS}
        cats = list(CATS)
        self.tasks = [
            dict(
                id=f"T-{100 + i}",
                category=(c := r.choice(cats)),
                needs=CATS[c],
                priority=r.randint(1, 5),
                region=r.choice(list(REGIONS)),
                deadline=r.randint(4, 14),
                duration=r.randint(1, 3)
            )
            for i in range(1, 19)
        ]
        self.assignments = {}
        self.audit.clear()
        self.events.clear()
        self.last = None
        self.history = []
        self._train()

    # ---------- ML ----------
    def _train(self):
        r = np.random.RandomState(self.seed)
        n = 1500
        X = np.c_[
            r.uniform(0, 4000, n),
            r.uniform(0, 1, n),
            r.uniform(0, 0.5, n),
            r.randint(0, 8, n),
            r.uniform(0.4, 1, n)
        ]
        y = np.clip(
            X[:, 4] - 0.00004 * X[:, 0] - 0.15 * X[:, 2] - 0.02 * X[:, 3] - 0.05 * X[:, 1] + r.normal(0, 0.03, n),
            0,
            1
        )
        self.rf = RandomForestRegressor(60, random_state=self.seed).fit(X, y)
        self.feat = ["operating_hours", "utilization", "recent_downtime", "past_failures", "current_readiness"]
        self.iso = IsolationForest(contamination=0.08, random_state=self.seed).fit(
            np.c_[r.normal(0.8, 0.08, 400), r.poisson(0.5, 400), r.normal(0, 0.02, 400)]
        )

    def predictions(self):
        out = []
        for a in self.aircraft:
            x = np.array([[a["hours"], a["util"], a["downtime"], a["failures"], a["readiness"]]])
            preds = np.array([t.predict(x)[0] for t in self.rf.estimators_])
            p = float(preds.mean())
            conf = float(max(0, 1 - 4 * preds.std()))
            imp = {f: round(float(i), 3) for f, i in zip(self.feat, self.rf.feature_importances_)}
            out.append(dict(
                id=a["id"],
                predicted=round(p, 2),
                confidence=round(conf, 2),
                risk="LOW" if p > 0.75 else "MEDIUM" if p > 0.55 else "HIGH",
                factors=imp
            ))
        return out

    def _anom(self):
        out = []
        for a in self.aircraft:
            f = [[a["readiness"], 0 if a["maintenance"] == "ok" else 1, a["readiness"] - a["prev_readiness"]]]
            if self.iso.predict(f)[0] == -1:
                out.append(dict(id=a["id"], kind="Isolation Forest outlier (readiness/maintenance/delta)"))
            if a["status"] == "available" and a["readiness"] < 0.35:
                out.append(dict(id=a["id"], kind="Source inconsistency: marked available but readiness very low"))
            if a["readiness"] - a["prev_readiness"] < -0.2:
                out.append(dict(id=a["id"], kind="Abnormal readiness drop"))
        return out

    def anomalies(self):
        out = []
        for x in self._anom():
            k = x["kind"]
            t, sev = (
                ("Source inconsistency", "HIGH")
                if k.startswith("Source")
                else ("Readiness drop", "HIGH")
                if k.startswith("Abnormal")
                else ("Statistical outlier", "MEDIUM")
            )
            out.append(dict(x, type=t, severity=sev, resource=x["id"], ts=time.strftime("%H:%M:%S"), explanation=k))
        return out

    def _log(self, rec):
        self.audit.insert(0, rec)
        if self.hook:
            try:
                self.hook(rec, self.assignments)
            except Exception:
                pass

    def _why(self, comp, cm):
        d = [f"{k} (-{x:.2f})" for x, k in sorted(((comp[k] - cm[k], k) for k in comp), reverse=True)[:2] if x > 0.005]
        return "Lower on " + ", ".join(d) if d else "Marginally lower overall / reserved for another task"

    # ---------- optimizer ----------
    def _pair(self, t, a):
        """Return (feasible, violations, components, eta, crew_id)."""
        v = []
        if a["status"] != "available":
            v.append("aircraft unavailable")
        if a["maintenance"] == "blocked":
            v.append("maintenance-blocked")
        if a["type"] != t["needs"]:
            v.append(f"incompatible type ({a['type']} vs {t['needs']})")
        if self.airspace[t["region"]] == "restricted":
            v.append(f"airspace {t['region']} restricted")
        if self.weather[t["region"]] > 0.8:
            v.append("weather above safe threshold")
        crew = [c for c in self.crew if c["aircraft"] == a["id"] and c["available"] and c["fatigue"] < 0.7]
        if not crew:
            v.append("no available rested crew")
        (x1, y1), (x2, y2) = REGIONS[a["region"]], REGIONS[t["region"]]
        eta = a["turnaround"] + round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 / 2) + t["duration"]
        if eta > t["deadline"]:
            v.append(f"misses deadline (ETA {eta}h > {t['deadline']}h)")
        w = self.weights
        comp = dict(
            priority=w["priority"] * t["priority"] / 5,
            readiness=w["readiness"] * a["readiness"],
            availability=w["availability"] * (a["fuel"] * 0.5 + (1 if crew else 0) * 0.5),
            deadline=w["deadline"] * max(0, (t["deadline"] - eta) / t["deadline"]),
            risk=-w["risk"] * self.risk[t["region"]],
            weather=-w.get("weather", 0.1) * self.weather[t["region"]]
        )
        return not v, v, comp, eta, (crew[0]["id"] if crew else None)

    def optimize(self, trigger="manual"):
        t0 = time.time()
        T, A = self.tasks, [a for a in self.aircraft]
        M = np.full((len(T), len(A)), -1e6)
        info = {}
        for i, t in enumerate(T):
            for j, a in enumerate(A):
                ok, v, comp, eta, crew = self._pair(t, a)
                info[i, j] = (ok, v, comp, eta, crew)
                if ok:
                    M[i, j] = sum(comp.values()) * (1 + t["priority"] / 5)  # priority-weighted

        ri, ci = linear_sum_assignment(-M)
        res, used = {}, set()
        for i, j in zip(ri, ci):
            if M[i, j] <= -1e5:
                continue
            ok, v, comp, eta, crew = info[i, j]
            t, a = T[i], A[j]
            used.add(a["id"])
            alts = sorted(
                [
                    (sum(info[i, k][2].values()), A[k]["id"], info[i, k][2])
                    for k in range(len(A))
                    if info[i, k][0] and k != j
                ],
                reverse=True
            )[:2]
            res[t["id"]] = dict(
                task=t["id"],
                aircraft=a["id"],
                crew=crew,
                eta=eta,
                score=round(sum(comp.values()), 3),
                components={k: round(x, 3) for k, x in comp.items()},
                alternatives=[
                    dict(aircraft=n, score=round(s, 3), why=self._why(comp, cm))
                    for s, n, cm in alts
                ],
                confidence=round(
                    max(0.3, min(0.99, 0.6 + 3 * (sum(comp.values()) - alts[0][0]) / max(sum(comp.values()), 1e-6)))
                    if alts
                    else 0.95,
                    2
                ),
                constraints=[
                    "Aircraft available",
                    "Not maintenance-blocked",
                    "Type compatible with task",
                    "Airspace open",
                    "Weather below threshold",
                    "Rested crew available",
                    "Meets deadline"
                ],
                reason=self._explain(t, a, comp, eta, alts)
            )

        unassigned = []
        for i, t in enumerate(T):
            if t["id"] not in res:
                causes = {}
                for j in range(len(A)):
                    for x in info[i, j][1]:
                        x = x.split(" (")[0]
                        causes[x] = causes.get(x, 0) + 1
                unassigned.append(dict(
                    task=t["id"],
                    priority=t["priority"],
                    blocked_by=dict(sorted(causes.items(), key=lambda z: -z[1])[:3]),
                    message=(
                        "No feasible allocation found"
                        if not any(info[i, j][0] for j in range(len(A)))
                        else "Lost to higher-value assignment"
                    )
                ))

        n = len(res)
        score = round(sum(r["score"] for r in res.values()), 3)
        rdy = [a["readiness"] for a in A]
        out = dict(
            trigger=trigger,
            assignments=list(res.values()),
            unassigned=unassigned,
            score=score,
            kpi=dict(
                utilization=round(len(used) / len(A), 2),
                completion_rate=round(n / len(T), 2),
                avg_readiness=round(float(np.mean(rdy)), 2),
                unassigned=len(unassigned),
                compute_ms=round((time.time() - t0) * 1000, 1)
            ),
            conflicts=self.conflicts(res),
            weights=self.weights
        )
        self.history.append(dict(ts=time.strftime("%H:%M:%S"), score=score, label=trigger[:40]))
        return out

    def _explain(self, t, a, comp, eta, alts):
        L = [
            f"+ Readiness {a['readiness']:.0%}",
            "+ Crew available and rested",
            f"+ Type compatible ({a['type']} for {t['category']})",
            f"+ Airspace {t['region']} open",
            f"+ Weather acceptable ({self.weather[t['region']]:.0%} severity)",
            f"+ Meets deadline: ETA {eta}h <= {t['deadline']}h"
        ]
        if self.risk[t["region"]] > 0.4:
            L.append(f"- Elevated regional risk {self.risk[t['region']]:.0%}")
        if eta >= t["deadline"] - 1:
            L.append("- Little schedule slack")
        if alts:
            L.append(f"Next best: {alts[0][1]} ({alts[0][0]:.2f})")
        return L

    def conflicts(self, res=None):
        c = []
        for a in self.aircraft:
            if a["status"] != "available" and a["id"] in [x["aircraft"] for x in self.assignments.values()]:
                c.append(f"{a['id']} unavailable but previously assigned")
        for t in self.tasks:
            if self.airspace[t["region"]] == "restricted":
                c.append(f"{t['id']} located in restricted sector {t['region']}")
        return c

    # ---------- events / retasking ----------
    def apply_event(self, kind, **p):
        r = random.Random(self.clock + self.seed)
        self.clock += 1
        if kind == "aircraft_unavailable":
            used = [x["aircraft"] for x in self.assignments.values()]
            a = next((x for x in self.aircraft if x["id"] == p.get("id")), None) or \
                self.aircraft_by(r.choice(used) if used else r.choice(self.aircraft)["id"])
            a["status"] = "unavailable"
            d = f"{a['id']} became unavailable"
        elif kind == "weather":
            k = p.get("region") or r.choice(list(REGIONS))
            self.weather[k] = 0.9
            d = f"Weather severe in {k}"
        elif kind == "airspace":
            k = p.get("region") or r.choice(list(REGIONS))
            self.airspace[k] = "restricted"
            d = f"Airspace {k} restricted"
        elif kind == "crew_unavailable":
            c = r.choice(self.crew)
            c["available"] = False
            d = f"Crew {c['id']} unavailable"
        elif kind == "maintenance":
            a = self.aircraft_by(p.get("id") or r.choice(self.aircraft)["id"])
            a["maintenance"] = "blocked"
            a["prev_readiness"] = a["readiness"]
            a["readiness"] = round(a["readiness"] - 0.3, 2)
            d = f"Maintenance event on {a['id']}"
        elif kind == "priority":
            t = r.choice(self.tasks)
            t["priority"] = 5
            d = f"{t['id']} escalated to priority 5"
        elif kind == "new_task":
            c = r.choice(list(CATS))
            t = dict(
                id=f"T-{100 + len(self.tasks) + 1}",
                category=c,
                needs=CATS[c],
                priority=5,
                region=r.choice(list(REGIONS)),
                deadline=self.clock + 10,
                duration=2
            )
            self.tasks.append(t)
            d = f"New priority-5 task {t['id']} ({c})"
        else:
            raise ValueError(f"Unknown event type: {kind}")

        before = {k: v["aircraft"] for k, v in self.assignments.items()}
        pre = self.conflicts(None)
        prev_score = self.last["score"] if self.last else 0
        res = self.optimize(trigger=d)
        self.assignments = {x["task"]: x for x in res["assignments"]}
        after = {k: v["aircraft"] for k, v in self.assignments.items()}
        changes = [
            dict(task=k, before=before.get(k), after=after.get(k))
            for k in sorted(set(before) | set(after))
            if before.get(k) != after.get(k)
        ]
        for ch in changes:
            ch["reason"] = (
                self.assignments[ch["task"]]["reason"]
                if ch["task"] in self.assignments
                else ["No feasible allocation"]
            )
        ts = time.strftime("%H:%M:%S")
        steps = (
            [d, f"{len(pre)} conflict(s) detected", "Optimizer executed"]
            + [f"{c['task']} reassigned {c['after'] or 'NONE'}" for c in changes]
            + ["Decision recorded"]
        )
        rec = dict(
            ts=ts,
            event=d,
            changes=changes,
            score=res["score"],
            prev_score=prev_score,
            conflicts_detected=pre,
            steps=steps,
            status="APPLIED" if res["assignments"] else "NO FEASIBLE"
        )
        self._log(rec)
        self.events.insert(0, dict(kind=kind, detail=d, ts=ts))
        self.last = res
        return dict(
            event=d,
            ts=ts,
            changes=changes,
            result=res,
            conflicts_detected=pre,
            prev_score=prev_score
        )

    def aircraft_by(self, i):
        return next(a for a in self.aircraft if a["id"] == i)

    def run_demo(self):
        self.reset()
        base = self.optimize("initial")
        self.assignments = {x["task"]: x for x in base["assignments"]}
        self.last = base
        self._log(dict(
            ts=time.strftime("%H:%M:%S"),
            event="Initial optimisation",
            changes=[],
            score=base["score"],
            status="APPLIED",
            steps=["Baseline initialised", "Initial optimisation executed", "Decision recorded"]
        ))
        steps = [self.apply_event(k) for k in ("aircraft_unavailable", "weather", "new_task")]
        return dict(
            initial=base,
            steps=steps,
            final=self.last,
            kpi_before=base["kpi"],
            kpi_after=self.last["kpi"]
        )
