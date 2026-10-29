# AIRSYNC
### AI Air Operations Decision Support & Resource Optimisation
*Fictional Synthetic High-Level Decision Support Demonstration Prototype*

---

## 1. Problem Statement
Planning and dynamically retasking air operations is challenging in rapidly shifting operational environments. Crucial telemetry—such as aircraft availability, crew duty status, maintenance schedules, airspace sector clearances, meteorological conditions, and emergent task priorities—traditionally resides in siloed, heterogeneous systems. Commanders and planners lack a unified, explainable decision-support framework that can dynamically resolve conflicts and optimize asset allocation in real time when disruptions occur.

---

## 2. Solution Overview
**AIRSYNC** provides an end-to-end AI-enabled decision-support pipeline that ingests distributed synthetic source feeds, normalizes them into a unified operational state, computes optimal constraint-aware task-resource assignments using the Hungarian optimization algorithm, predicts asset readiness with scikit-learn Machine Learning, flags behavioral anomalies, and dynamically retasks assets upon environmental or resource disruptions—providing full mathematical explainability and auditability.

```
SOURCE DATA FEEDS
(Aircraft, Crew, Weather, Airspace, Tasking, Risk)
         │
         ▼
NORMALIZATION LAYER
(Standardized Schemas & Unit Alignment)
         │
         ▼
DATA FUSION ENGINE
(Unified Operational State & Cross-Source Reconciliation)
         │
         ▼
CONSTRAINT-BASED OPTIMIZER
(Linear Sum Assignment / Hungarian Algorithm)
         │
         ▼
EXPLAINABLE AI & DYNAMIC RETASKING
(Factor Attributions, Alternatives, Disruption Reallocation)
         │
         ▼
PERSISTENT AUDIT TRAIL & OPERATIONAL DASHBOARD
(SQLite, FastAPI, Glassmorphic Tactical UI)
```

---

## 3. Architecture

### Backend Stack
- **Framework**: Python 3.11+, FastAPI (Asynchronous high-performance REST API)
- **Mathematical Optimization**: `scipy.optimize.linear_sum_assignment` (Kuhn-Munkres Hungarian Algorithm)
- **Predictive Analytics**: `scikit-learn` `RandomForestRegressor` for asset operational readiness prediction and feature importance
- **Anomaly Detection**: `scikit-learn` `IsolationForest` statistical outlier detector combined with heuristic domain validation rules
- **Data Persistence**: SQLite (`airsync.db`) with thread-safe locking and structured audit payloads
- **Synthetic Data Engine**: 24 aircraft resources, 36 crew members, 18 tasks, 8 fictional sectors (R0–R7)

### Frontend Stack
- Single-page tactical operational dashboard served directly by FastAPI
- Vanilla CSS with Glassmorphism, Google Fonts (`Inter` + `JetBrains Mono`), and responsive CSS Grid
- Interactive SVG tactical sector grid visualizer with real-time vector connections and status blips
- Real-time polling and dynamic re-rendering of telemetry, KPI diffs, and audit logs

---

## 4. Key Capabilities & Modules

### A. Multi-Source Data Fusion (`app/fusion.py`)
Ingests and reconciles 6 synthetic streaming feeds:
1. **Aircraft System Feed**: Aircraft tail numbers, airframe classes, readiness percentages, current sector coordinates, fuel percentage, maintenance status.
2. **Crew System Feed**: Crew IDs, airframe type qualification, active duty availability, fatigue indices, readiness ratings.
3. **Weather Feed**: Sector-level meteorological severity index (0.0 to 1.0).
4. **Airspace Feed**: Airspace clearance status (`OPEN` vs `RESTRICTED`).
5. **Tasking System Feed**: Task priorities (P1–P5), required airframe capability, operating sector, deadlines, duration.
6. **Risk Feed**: Regional operational threat/risk index (0.0 to 1.0).

The fusion engine validates cross-source consistency, calculates fusion compute latency, and outputs a canonical `OperationalState`.

### B. Constraint-Based Optimization (`app/engine.py`)
The Hungarian algorithm minimizes total cost (maximizes multi-criteria utility) while enforcing strict hard constraints:
- **Aircraft Availability**: Status must be `available`.
- **Maintenance Integrity**: Aircraft must not be `maintenance-blocked`.
- **Capability Compatibility**: Airframe type must match task requirement (`Transport`, `Surveillance`, `Utility`, `Rotary`).
- **Airspace Clearance**: Operating sector must not be `restricted`.
- **Weather Limits**: Regional weather severity must be $\le 0.80$.
- **Crew Availability**: Rested, assigned crew must be on duty (fatigue index $< 0.70$).
- **Feasible Timeline**: $\text{ETA} = \text{turnaround} + \text{transit} + \text{duration} \le \text{deadline}$.
- **Zero Double-Booking**: Guaranteed 1-to-1 matching via linear sum assignment.

Utility components incorporate customizable weights:
$$\text{Score} = w_{\text{priority}} \cdot P + w_{\text{readiness}} \cdot R + w_{\text{avail}} \cdot A + w_{\text{deadline}} \cdot D - w_{\text{risk}} \cdot \text{Risk} - w_{\text{weather}} \cdot \text{Wx}$$

### C. Dynamic Retasking & Disruption Handling
Simulates realistic operational disruptions:
1. Resource breakdown / unavailability
2. Severe weather front closing an airspace sector
3. Emergent high-priority task insertion
4. Airspace sector restrictions
5. Crew fatigue exhaustion
6. Unscheduled maintenance event

When an event triggers:
1. **Impact Analysis**: Detects affected current assignments.
2. **Conflict Detection**: Flags newly violated hard constraints.
3. **Re-Optimization**: Re-runs the Hungarian optimizer on the adjusted state.
4. **Allocation Diff**: Compares previous allocation vs. new allocation.
5. **Audit Logging**: Persists the event, score delta, and decision rationale to SQLite.

### D. Explainable AI (XAI)
For every task assignment, the explainer produces:
- Top positive attribution factors (e.g., high readiness, short transit distance, rested crew)
- Negative factors / trade-offs (e.g., high regional risk, little deadline slack)
- Satisfied hard constraints
- Evaluated alternative resources with exact comparative delta: *Why candidate B was ranked lower than candidate A*.

### E. Predictive Readiness & Anomaly Detection
- **Random Forest**: Predicts next-cycle readiness and outputs feature importance (`operating_hours`, `utilization`, `recent_downtime`, `past_failures`, `current_readiness`).
- **Isolation Forest + Heuristics**: Detects statistical multi-dimensional outliers, abnormal readiness drops ($> 20\%$), and data source inconsistencies (e.g., marked available with low readiness).

---

## 5. API Reference

All endpoints return real application state:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health status and synthetic demo indicator |
| `GET` | `/api/v1/dashboard/summary` | Real-time KPI summary (resources, ready, tasks, simulation, history) |
| `GET` | `/api/v1/resources` | List of all 24 aircraft resources |
| `GET` | `/api/v1/resources/{id}` | Individual aircraft telemetry (404 if not found) |
| `GET` | `/api/v1/crew` | List of all 36 crew members |
| `GET` | `/api/v1/tasks` | Active operational tasks |
| `GET` | `/api/v1/recommendations` | Active task assignments and recommendations |
| `GET` | `/api/v1/recommendations/{id}` | Detailed recommendation and explanation for specific task |
| `GET` | `/api/v1/audit` | Chronological audit trail from SQLite store |
| `GET` | `/api/v1/analytics/readiness` | Random Forest ML readiness predictions & feature importance |
| `GET` | `/api/v1/analytics/anomalies` | Isolation Forest & rule-based operational anomalies |
| `GET` | `/api/v1/weather` | Current weather severity by sector |
| `GET` | `/api/v1/airspace` | Airspace sector clearance status |
| `GET` | `/api/v1/risks` | Sector operational risk levels |
| `GET` | `/api/v1/fusion` | Multi-source data fusion status & pipeline metrics |
| `GET` | `/api/v1/events` | Recent event telemetry log |
| `GET` | `/api/v1/optimization/latest` | Latest linear sum assignment state |
| `POST` | `/api/v1/optimization/run` | Execute optimizer with custom factor weights |
| `POST` | `/api/v1/events` | Inject operational disruption event (422 if invalid) |
| `POST` | `/api/v1/demo/run` | Run deterministic 3-step disruption demo |
| `POST` | `/api/v1/simulation/start` | Start background autonomous simulation |
| `POST` | `/api/v1/simulation/pause` | Pause background simulation |
| `POST` | `/api/v1/simulation/reset` | Reset world to baseline seed state |

Interactive Swagger API Documentation:
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`, `uv`, or `conda`)

### Step-by-Step Setup
1. Clone the repository and navigate to `backend`:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
5. Open your browser:
   - Dashboard: **http://localhost:8000**
   - API Docs: **http://localhost:8000/docs**

---

## 7. Running with Docker & Docker Compose

### Using Docker Compose
```bash
docker compose up --build
```
Access the dashboard at **http://localhost:8000**.

### Using Docker directly
```bash
docker build -t airsync .
docker run -p 8000:8000 -v airsync-data:/data airsync
```

---

## 8. Running Test Suite

Execute pytest in the `backend` directory:
```bash
cd backend
python -m pytest -q
```
Or with verbose output:
```bash
python -m pytest -v
```
All 28 unit and integration tests validate:
- Health, resources, crew, and tasks endpoints
- Hungarian optimization algorithm and objective scores
- Hard constraints (aircraft type, maintenance status, airspace clearance, weather limit, crew fatigue, deadlines)
- Double-booking prevention
- Dynamic retasking and conflict detection
- ML readiness predictions and Isolation Forest anomaly detector
- Multi-source data fusion pipeline
- End-to-end demo scenario execution and SQLite persistence

---

## 9. 10-Stage Demo Scenario Flow

Clicking **RUN DEMO SCENARIO** in the top navigation bar triggers a 10-stage live demonstration:
1. **Stage 1**: Generates baseline operational state.
2. **Stage 2**: Executes initial baseline optimization and establishes KPI benchmark.
3. **Stage 3**: Disruption Event 1 — Primary resource becomes unavailable.
4. **Stage 4**: Detects assignment conflict on affected missions.
5. **Stage 5**: Triggers dynamic re-optimization and identifies substitute aircraft.
6. **Stage 6**: Disruption Event 2 — Severe weather closes sector R2.
7. **Stage 7**: Re-optimizes, rerouting missions away from the restricted weather front.
8. **Stage 8**: Disruption Event 3 — Emergent Priority-5 critical task arrives.
9. **Stage 9**: Re-balances fleet allocation, preempting lower-priority tasks where necessary.
10. **Stage 10**: Displays final operational state with comprehensive **Before vs. After** delta metrics (utilization, score change, completion rate).

---

## 10. Synthetic Data Disclaimer & Limitations
> **IMPORTANT NOTICE: SYNTHETIC DATA ONLY**
> 
> AIRSYNC is an academic/hackathon concept demonstration for high-level decision support and resource optimization.
> - **All data is 100% fictional**: All aircraft IDs, categories, crew members, coordinates, weather fronts, airspace sectors, risk scores, and mission scenarios are synthetic and randomly generated.
> - **No Real-World Targeting or Weapons Employment**: This system contains NO weapon-release calculations, targeting logic, tactical routing, real military installations, or operational defense data.
> - **Demonstration Purpose Only**: This prototype is intended strictly to showcase multi-source data fusion, mathematical optimization algorithms, and explainable AI in a synthetic logistics and operations context.
