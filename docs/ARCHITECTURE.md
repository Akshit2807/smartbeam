# SmartBeam 2.0 — Architecture & API Contract (BINDING)

Everyone builds against this file. Do not rename/remove public names or change signatures of modules you do
not own. To extend a contract: ADD (new optional kwarg, new function, new field with a default) and log it under
**Contract changes** at the bottom. The source files are the ground truth for field order; this document
mirrors them.

## 1. Folder structure & ownership

```
smartbeam/                    project root (run everything from here)
├── app.py                    Streamlit entry: st.navigation + theme            [architect]
├── .streamlit/config.toml    dark theme                                        [architect]
├── requirements.txt                                                            [architect]
├── docs/REQUIREMENTS.md, docs/ARCHITECTURE.md                                  [architect]
├── smartbeam/                ENGINE (no Streamlit imports anywhere in here except smartbeam/ui)
│   ├── model.py              data classes, validation, JSON, statics bookkeeping [architect]  IMPLEMENTED
│   ├── sections.py           section properties, Q(y), b(y), kappa             [architect]  IMPLEMENTED
│   ├── materials.py          5-material library + custom                        [architect]  IMPLEMENTED
│   ├── units.py              unit conversion / formatting                       [architect]  IMPLEMENTED
│   ├── metrics.py            DesignMetrics, ConstraintCheck (shared)            [architect]  IMPLEMENTED
│   ├── analytical.py         classical engine + analyze() facade                [analytical engineer] (facade dispatch implemented)
│   ├── fem.py                1D beam / Timoshenko / frame FEM                   [FEM engineer]
│   ├── safety.py             yield, von Mises, buckling, fatigue                [safety engineer]
│   ├── optimize.py           SLSQP optimization, Pareto                         [optimization engineer]
│   ├── advisor.py            rule-based advisor                                 [advisor engineer]
│   ├── report.py             ReportLab PDF + CSV                                [reports engineer]
│   ├── lab.py                experiments, challenges, sensitivity               [lab engineer]
│   └── ui/
│       ├── theme.py          CSS, COLORS, Plotly template, page_header, badge   [architect]  IMPLEMENTED
│       ├── state.py          session-state helpers, cached analysis             [architect]  IMPLEMENTED
│       └── (playground component, plot helpers: owners add files here, e.g. ui/plots.py, ui/playground/)
├── pages/                    one file per page; each page's owner = owner of the matching engine module
│   home.py design_studio.py analysis.py numerical_fem.py beam_theories.py visualization_3d.py
│   safety_failure.py design_comparison.py optimization.py design_advisor.py virtual_lab.py
│   playground.py reports.py about.py
└── tests/                    pytest; conftest.py provides benchmark fixtures    [each owner adds test_<module>.py]
```

Rules: engine modules never import `streamlit` or `smartbeam.ui`. `fem.py` must not import `analytical.py`
(analytical imports fem lazily inside `analyze`). Pages call the engine; they never re-implement mechanics.
Pages do NOT call `st.set_page_config` (app.py does) and do NOT need to call `apply_theme()`.

## 2. Units

SI everywhere inside the engine: m, N, N/m, N·m, Pa, kg, kg/m³, rad. Convert only at the UI/report boundary
with `smartbeam.units` (`to_si(v, "kN")`, `from_si(v, "mm")`, `convert(v, src, dst)`, `fmt(v_si, "MPa")`).
Display units: mm, N/kN, MPa/GPa, N/m or kN/m, N·m or kN·m. Cost: INR (₹).

## 3. Sign convention (binding)

| Quantity | Convention |
|---|---|
| Axes | x along the beam from the LEFT end, 0 ≤ x ≤ L. y UP. Section coordinate y measured from the centroid, + up. |
| Point force | `PointLoad(x, magnitude, angle_deg)`: direction angle CCW from +x. **−90° = down (default)**, +90° = up, 0° = axial →, 180° = axial ←. Components `fx = P cos θ`, `fy = P sin θ` (+x, +y positive). |
| Distributed load | `DistributedLoad(x1, x2, w1, w2, direction)`: intensities along `direction` ∈ {'down' (default), 'up'}. `q(x)` = signed intensity in **+y** (down ⇒ negative). Linear between x1 and x2. Transverse only. |
| Applied moment | `MomentLoad(x, M)`: **CCW positive**, CW negative (`MomentLoad.cw(x, |M|)`, `.ccw(x, |M|)`). |
| Reactions | `Reaction.Fx, Fy` act ON the beam, + in +x/+y. `Reaction.M` CCW positive. |
| Shear V | V(x) = Σ Fy of all forces to the LEFT of the cut (incl. reactions). ⇒ **dV/dx = q**, i.e. = −w for a downward load w. |
| Moment M | **Sagging positive.** M(x) = Σ Fy_left·(x − a) − Σ C_left (C = applied CCW moments left of the cut). ⇒ **dM/dx = V**. A CCW applied moment makes M jump DOWN by C. |
| Axial N | **Tension positive.** N(x) = −Σ Fx_left. |
| Deflection v | + up (y). Slope θ = dv/dx, + CCW. **EI v'' = M**. |
| Normal stress | σ(x, y) = N/A − M·y/I. Sagging M ⇒ bottom fibre (y < 0) in tension. σ_top uses `y_top` (>0), σ_bottom uses `y_bottom` (<0). Tension positive. |
| Shear stress | τ(y) = V·Q(y)/(I·b(y)). |
| Strain | ε = σ/E. |

Checks: SS beam, central load P down → R = +P/2 each, M(L/2) = +PL/4, v_min = −PL³/48EI, σ_bottom > 0.
Cantilever fixed at x = 0, end load P down → R_y = +P, M_reaction = +PL (CCW), M(0) = −PL (hogging),
v(L) = −PL³/3EI, θ(L) = −PL²/2EI.

**Reported maxima** (`Extreme(value, x, where)`) keep the SIGN of the value at the location of max |value|.
Comparisons with textbook magnitudes use `abs()`.

## 4. Supports, determinacy, routing

`Support(type, x)` with type ∈ {'fixed', 'pin', 'roller'}; restrained planar DOFs:
fixed (u, v, θ), pin (u, v), roller (v) — roller resists only a force normal to the beam.

`check_stability(model, mode=None) -> Stability` — mode 'transverse' when `is_transverse_only(model)`
(rigid DOFs v, θ; axial restraints ignored), else 'planar' (u, v, θ). Stable iff rank of the restraint matrix
equals the rigid-body DOF count; `degree = n_restraints − required`.

**`analytical.analyze(model)` routing (implemented, binding):**

| Condition | Route | `result.method` |
|---|---|---|
| `validate(model)` non-empty | raise `ModelError` | — |
| unstable | raise `UnstableStructureError(stability.message)` | — |
| transverse-only **and** determinate | statics + numerical integration (`_analyze_determinate`) | `'analytical'` |
| transverse-only, indeterminate | `fem.solve(theory='euler', frame=False)` + warning "statics insufficient" | `'fem-euler'` |
| any axial / inclined point load | `fem.solve(theory='euler', frame=True)` + warning | `'fem-frame'` |

Textbook formulas (`textbook_check`) are returned ONLY for: cantilever + single vertical end load,
cantilever + full-length UDL, simply supported + central vertical point load, simply supported + full-length UDL.
Everything else → `None` (never forced).

## 5. Engine API

### model.py (implemented)
```python
SUPPORT_TYPES = ("fixed","pin","roller"); SHAPES = ("rect","solid_circ","hollow_circ"); LOAD_DIRECTIONS = ("down","up")
RESTRAINTS = {"fixed": ("u","v","theta"), "pin": ("u","v"), "roller": ("v",)}; POS_TOL = 1e-9
class ModelError(ValueError); class UnstableStructureError(ValueError)

@dataclass Material(name, E, yield_strength, nu, density, uts=None, cost_per_kg=None,
                    sigma_f_prime=None, b_fatigue=None, endurance_limit=None, approximate=True, note="")   .G
@dataclass Section(shape, b=None, h=None, d=None, D=None)
    Section.rect(b, h) | Section.solid_circ(d) | Section.hollow_circ(D, d)   # hollow: D outer, d inner
    .depth, .dims() -> {"b","h"} | {"d"} | {"D","d"}
@dataclass Support(type, x)                                  .restrained
@dataclass PointLoad(x, magnitude, angle_deg=-90.0)          .fx .fy .is_transverse   kind='point'
@dataclass DistributedLoad(x1, x2, w1, w2, direction="down") .q(x) .resultant .centroid .sign   kind='distributed'
@dataclass MomentLoad(x, M)  MomentLoad.cw(x, m) / .ccw(x, m)                         kind='moment'
Load = PointLoad | DistributedLoad | MomentLoad
@dataclass Reaction(support_index, type, x, Fx=0.0, Fy=0.0, M=0.0)
@dataclass Extreme(value, x, where="")   Extreme.of(x, values, where="")   # signed value at argmax |values|
@dataclass BeamModel(L, section, material, supports=[], loads=[], name="")
    .point_loads .distributed_loads .moment_loads .copy()
    .to_dict() / BeamModel.from_dict(d) / .to_json(indent=2) / BeamModel.from_json(s)   # loads tagged by "kind"
@dataclass Stability(stable, determinate, degree, n_restraints, rank, required, mode, message)

validate(model) -> list[str]               # data errors only (empty = valid)
is_transverse_only(model) -> bool
classify(model) -> 'cantilever' | 'simply-supported' | 'general'
check_stability(model, mode=None) -> Stability
```
JSON schema: `{"version":1,"name","L","section":{shape,b,h,d,D},"material":{...Material fields},
"supports":[{type,x}],"loads":[{"kind":"point",x,magnitude,angle_deg} | {"kind":"distributed",x1,x2,w1,w2,direction} | {"kind":"moment",x,M}]}`.
Playground config save/load uses exactly this schema.

### sections.py (implemented)
```python
area(s); centroid(s)  # height above bottom fibre
second_moment(s); section_modulus(s)  # Z = I / (depth/2)
radius_of_gyration(s); y_top(s) (>0); y_bottom(s) (<0)
width_at(s, y); first_moment(s, y)  # Q(y) of area above y; scalar or array
shear_stress(s, V, y)                # V Q / (I b)
shear_correction_factor(s, nu=0.3)   # rect 5/6, solid 0.9, hollow: Cowper (≈0.53 thin tube)
properties(s, nu=0.3) -> SectionProperties(A, I, Z, r_g, y_top, y_bottom, centroid, depth, kappa, Q_max)
```
Bending is about the horizontal axis (strong axis for rect: depth h vertical). Buckling uses the minimum I
(safety owner computes it; for rect min(bh³, hb³)/12).

### materials.py (implemented)
`MATERIALS: dict[key, Material]`, keys `structural_steel, mild_steel, aluminium_6061_t6, copper, brass`
(approximate values, fatigue None). `get_material(key) -> Material` (copy), `material_keys()`,
`custom_material(name, E, yield_strength, nu, density, uts=None, cost_per_kg=None)` (raises ValueError).

### metrics.py (implemented)
```python
@dataclass DesignMetrics(name, material, shape, dims, L, A, I, volume, mass, cost, delta_max, sigma_max, M_max, fos, method)
    # delta_max, sigma_max, M_max are ABS values; fos = sigma_y/|sigma|max (inf if unstressed); cost None if no rate
@dataclass ConstraintCheck(name, value, limit, sense, satisfied, margin_pct)   # sense '<=' | '>='
check(name, value, limit, sense="<=") -> ConstraintCheck
evaluate(model, result=None, cost_per_kg=None) -> DesignMetrics   # calls analyze() if result is None
```

### analytical.py [analytical engineer]
```python
@dataclass TextbookCheck(case, formulas: dict[str,str], textbook: dict[str,float], numerical={}, rel_error={})
    # case: 'cantilever-end-load' | 'cantilever-udl' | 'ss-central-load' | 'ss-udl'
    # keys used: 'R', 'M_max', 'V_max', 'theta_max', 'delta_max' (+ 'sigma_max'), magnitudes
@dataclass AnalysisResult(model, method, x, V, M, N, theta, v, sigma_top, sigma_bottom, eps_top, eps_bottom,
    reactions: list[Reaction], V_max, M_max, N_max, theta_max, v_max, sigma_max, sigma_tension_max,
    sigma_compression_max (all Extreme), stability: Stability, equilibrium: dict[str,float],
    textbook: TextbookCheck|None = None, warnings: list[str] = [], fem: FEMResult|None = None)
    # x non-decreasing; x REPEATED at concentrated forces/moments (left value then right value)
reactions(model) -> list[Reaction]                    # determinate transverse only, else ValueError
internal_forces(model, reacts, x) -> (V, M, N)
sample_points(model, n_points=501) -> np.ndarray
slope_deflection(model, x, M) -> (theta, v)
fibre_stresses(model, M, N=0.0) -> (sigma_top, sigma_bottom)            # implemented
stress_distribution(model, M, N=0.0, n=51) -> (y, sigma)                # implemented
equilibrium_residual(model, reacts) -> {'sum_Fx','sum_Fy','sum_M0'}
textbook_check(model, result=None) -> TextbookCheck | None
analyze(model, n_points=501, n_elements=40) -> AnalysisResult           # dispatch implemented
```
Private hooks the owner must implement: `_analyze_determinate(model, n_points, stab)` and
`_from_fem(model, fem_result, stab, warnings)`.

### fem.py [FEM engineer]
Global DOF of node i, local k: `i*dofs_per_node + k` (beam: v=0, θ=1; frame: u=0, v=1, θ=2).
```python
COND_LIMIT = 1e12
@dataclass Mesh(nodes, elements)  .n_nodes .n_elements .lengths
@dataclass FEMResult(model, theory, element, mesh, dofs_per_node, U, u_nodes, v_nodes, theta_nodes,
    reactions: list[Reaction], element_end_forces, x, v, theta, V, M, N, sigma_top, sigma_bottom,
    v_max, theta_max, M_max, V_max, sigma_max (Extreme), condition_number, n_dof, n_free, warnings=[])
@dataclass ConvergenceResult(n_elements, theory, v_max, theta_max, M_max, sigma_max, reference, error_pct, note)
eb_element_stiffness(E, I, Le) -> (4,4)
timoshenko_element_stiffness(E, I, Le, G, A, kappa) -> (4,4)        # phi = 12EI/(kappa G A Le^2)
frame_element_stiffness(E, A, I, Le, phi=0.0) -> (6,6)              # [u1,v1,th1,u2,v2,th2]
build_mesh(model, n_elements) -> Mesh                               # nodes at all supports/loads/UDL ends
consistent_load(q1, q2, Le) -> (4,)
assemble(model, mesh, theory="euler", frame=False) -> (K, F)
restrained_dofs(model, mesh, frame=False) -> np.ndarray
solve_linear(K, F, fixed) -> (U, R, cond)                           # raises UnstableStructureError
reconstruct(mesh, U, frame=False, n_per_element=10) -> (x, v, theta) # Hermite cubic
solve(model, n_elements=20, theory="euler", frame=None, n_per_element=10) -> FEMResult  # frame=None: auto
convergence_study(model, n_list=(2,4,8,16,32), theory="euler") -> ConvergenceResult
```
Beam-theory comparisons (Beam Theories page) call `fem.solve(model, theory="timoshenko")` directly.

### safety.py [safety engineer]
```python
STATUS_SAFE="SAFE"; STATUS_WARNING="WARNING"; STATUS_YIELD="YIELD PREDICTED"
BUCKLING_K = {"fixed-free":2.0, "pinned-pinned":1.0, "fixed-pinned":0.699, "fixed-fixed":0.5}
@dataclass YieldResult(sigma_max: Extreme, sigma_y, fos, first_yield_load_factor, status, min_fos, message)
@dataclass CombinedStressResult(x, y, sigma, tau, von_mises, fos, y_profile, vm_profile, assumptions)
@dataclass BucklingResult(applicable, end_condition, K, L_eff, Pcr, P_applied, fos, slenderness,
    slenderness_limit, mode_x, mode_shape, warnings=[])
@dataclass FatigueResult(sigma_a, sigma_m, sigma_ar, cycles_to_failure, goodman_fos, infinite_life,
    sn_N, sn_S, assumptions, warnings=[])
@dataclass SafetyReport(yield_result, combined, buckling, status, messages)
yield_check(result, min_fos=1.5) -> YieldResult   # fos<1 YIELD PREDICTED; fos<min_fos WARNING; else SAFE
von_mises_check(result, n_y=41) -> CombinedStressResult
infer_end_condition(model) -> str | None
buckling(model, end_condition=None, P_applied=None, n_points=201) -> BucklingResult
fatigue(sigma_max, sigma_min, uts, sigma_f_prime=None, b=None, endurance_limit=None) -> FatigueResult
assess(model, result, min_fos=1.5) -> SafetyReport
```

### optimize.py [optimization engineer]
```python
@dataclass OptimizationProblem(objective="volume", delta_allow=0.005, fos_min=1.5, bounds={}, t_min=1e-3, cost_per_kg=None)
    # objective 'volume'|'mass'|'cost'; bounds keyed like Section.dims(): {'b':(lo,hi),'h':(lo,hi)} (m)
@dataclass OptimizationResult(status, success, feasible, initial_dims, optimal_dims, initial: DesignMetrics,
    optimized: DesignMetrics|None, reduction_pct, checks: list[ConstraintCheck], message,
    infeasibility_reasons: list[str], n_iterations, optimized_model)
    # status 'OPTIMIZED' only if independent re-verification passes; else 'INFEASIBLE' | 'FAILED'
optimize(model, problem) -> OptimizationResult
pareto_front(model, problem, material_keys=None, n_points=15) -> pd.DataFrame
```

### advisor.py [advisor engineer]
```python
DEFAULT_SPAN_RATIO = 250   # delta_allow = L/250 if not given (stated as an editable assumption)
@dataclass Recommendation(issue, quantity, current, limit, action, modified_model, before, after, checks_after, satisfies_all)
@dataclass AdvisorReport(limiting, utilisation, checks, recommendations)
advise(model, delta_allow=None, min_fos=1.5) -> AdvisorReport
```

### report.py [reports engineer]
```python
@dataclass ReportMeta(title="SmartBeam 2.0 Engineering Report", project="", team="[Team members]",
                      institute="[Institute]", guide="[Guide]", date="")
build_pdf(model, analysis=None, fem=None, convergence=None, safety=None, optimization=None, advisor=None, meta=None) -> bytes
analysis_csv(result) -> str; fem_csv(result) -> str; lab_report_pdf(experiment_result, meta=None) -> bytes
```

### lab.py [lab engineer]
```python
PARAMS = ("L","E","load","b","h","d","D")
@dataclass Experiment(key, title, objective, explanation, equations, parameter)
@dataclass ExperimentResult(key, title, objective, explanation, equations, parameter, table: DataFrame, conclusion)
EXPERIMENTS: dict[str, Experiment]   # keys: height_vs_stiffness, length_vs_deflection, steel_vs_aluminium,
    # rect_vs_circular, solid_vs_hollow, point_vs_distributed, load_position, cantilever_vs_ss,
    # eb_vs_timoshenko, mesh_convergence, volume_vs_performance
@dataclass Challenge(key, title, brief, base_model, delta_allow, min_fos, max_mass, editable=("section","material"))
@dataclass ChallengeGrade(passed, score, checks, metrics, feedback)
CHALLENGES: list[Challenge]
scale_parameter(model, param, factor) -> BeamModel
run_experiment(key, base_model=None, values=None) -> ExperimentResult
grade_challenge(challenge, model) -> ChallengeGrade
sensitivity_sweep(model, param, factors=np.linspace(0.5,2,16)) -> DataFrame
tornado(model, pct=10.0, params=("L","E","load","b","h")) -> DataFrame
loglog_slope(x, y) -> float
```

## 6. UI layer

`smartbeam/ui/theme.py`: `COLORS` (bg #0B1220, panel #111A2B, border #24324D, text #E6EDF7, muted #8FA0BA,
**cyan #22D3EE** primary — geometry/deflection, **amber #F5A524** — loads/maxima, green SAFE, red YIELD),
`SERIES`, `PLOTLY_TEMPLATE = "plotly_dark+smartbeam"` (registered and set as Plotly default on import),
`apply_theme()` (called by app.py), `page_header(title, subtitle=None)`, `badge(text, color=None) -> html`.
Plotly: build figures normally (the template is the default once theme.py is imported — app.py imports it) and
render with **`st.plotly_chart(fig, theme=None)`**; without `theme=None` Streamlit overrides our template.

`smartbeam/ui/state.py`:
```python
default_model() -> BeamModel   # structural steel, rect 30x60 mm, L=1 m, fixed at x=0, 1 kN down at x=L
get_model() -> BeamModel       # fresh copy from st.session_state['model']
set_model(model); reset_model()
analyze_model(model=None, n_points=501, n_elements=40, record=True) -> AnalysisResult  # cached by model JSON;
    # raises ModelError / UnstableStructureError -> pages show st.error(str(e))
add_recent(model, result); recent_analyses() -> list[dict]  # keys time,name,config,method,L_m,delta_max_m,sigma_max_Pa,model
saved_designs() -> dict[str, BeamModel]; save_design(name, model); delete_design(name)
```
Session keys owned by state.py: `model`, `recent`, `saved_designs`. Page-private keys: prefix with the page
name (e.g. `playground_selected`).

Navigation (app.py): Overview [Home] · Design & Analysis [Design Studio, Analysis, Numerical FEM, Beam Theories,
3D Visualization] · Evaluate & Improve [Safety & Failure, Design Comparison, Optimization, Design Advisor] ·
Explore [Virtual Lab, Structural Playground] · Output [Reports, About & Methodology].
To link: `st.page_link("pages/analysis.py", label=...)` / `st.switch_page("pages/visualization_3d.py")`.

## 7. Testing

`tests/conftest.py` fixtures: `steel`, `rect_section`, `cantilever_end_load`, `cantilever_udl`,
`ss_central_load`, `ss_udl` (structural steel, rect 30×60 mm, L = 2 m, P = 1000 N down, w = 500 N/m down).
Each owner adds `tests/test_<module>.py`. Run: `.venv/bin/python -m pytest -q`.
Run app: `.venv/bin/python -m streamlit run app.py`.

## Contract changes
(append entries: date · module · change · author)
- 2026-11-03 · analytical.py · `slope_deflection(model, x, M, V=None)`: optional `V` (= dM/dx) adds the Euler–Maclaurin end correction to the cumulative trapezoid, making the slope integral exact for the cubic M of linearly varying loads. Without `V`, the result is the plain trapezoid. · analytical engineer
- 2026-11-03 · analytical.py · new `differential_residuals(model, x, V, M) -> {'dM_dx_minus_V', 'dV_dx_minus_q'}` (dimensionless relative residuals). On the analytical route, `AnalysisResult.equilibrium` carries these 2 keys in addition to `sum_Fx/sum_Fy/sum_M0`. The FEM route has only the 3 sums. · analytical engineer
- 2026-11-03 · analytical.py · `sample_points` also repeats x = 0 / x = L when a support or concentrated load sits there. So V and M start and end at their left/right limits: V(0⁻) = 0, and at a free end V(L⁺) = M(L⁺) = 0, which closes the diagram. `internal_forces` treats the 2nd copy of a repeated x as the right-hand value. · analytical engineer
- 2026-11-03 · analytical.py · `TextbookCheck` also covers a mirrored cantilever (fixed at x = L, load at x = 0). It adds `theta_max` for the SS cases (PL²/16EI, wL³/24EI) and `sigma_max = M_max/Z` for all 4 cases. · analytical engineer
- 2026-11-03 · fem.py · additions only (no signature removed/renamed). `consistent_load(q1, q2, Le, phi=0.0)`: with `phi > 0` it integrates the Timoshenko (interdependent) shape functions, which keeps nodal values exact. `reconstruct(..., phi=None)`: Timoshenko shape functions; theta is then the section rotation ψ. New `timoshenko_phi(E, I, Le, G, A, kappa)`, `THEORIES`, `MECHANISM_MSG`. New `compare_theories(model, n_elements=20, frame=None) -> TheoryComparison(eb, timoshenko, v_eb, v_timoshenko, diff_pct, span_to_depth, phi_beam, kappa, G)`. New defaulted fields: `FEMResult.phi` (per-element φ) and `FEMResult.N_max` (Extreme); `ConvergenceResult.M_fe_max` (|EI v''| of the raw cubic field, which converges under distributed loads, unlike the equilibrium-recovered M) and `ConvergenceResult.n_requested`. `ConvergenceResult.n_elements` holds the ACTUAL element counts, since nodes are forced at key points. `error_pct` adds the key `'M_fe'`. · FEM engineer
- 2026-11-03 · fem.py · behaviour notes. `condition_number` is the 2-norm condition number of the Jacobi-scaled K_ff (so units are not mixed). Key points closer than 1e-4·L share one node, and a warning gives the shift; tinier elements made the K·U − F reactions lose digits. Supports merged this way report their reaction once. `frame=True` on a transverse-only model with no axial restraint holds u at the first support, with a warning (zero reaction). FEM `x` is repeated at EVERY element boundary, a superset of the concentrated-load points. · FEM engineer
- 2026-11-03 · safety.py · ADDED: `STATUS_BUCKLING = "BUCKLING PREDICTED"` (used by `assess()` when the Euler FOS < 1, so instability is never mislabelled as yield) and `STATUS_RANK` (severity: SAFE 0, WARNING 1, YIELD/BUCKLING 2). UI colour maps must handle this 4th status (treat like YIELD). Also: `status_for(fos, min_fos)`, `min_second_moment(section)`, `compressive_force(result) -> N` (max compression of `result.N`, 0 for transverse-only), `DISCLAIMER` (yield ≠ fracture text, already appended to `SafetyReport.messages`), and `FatigueResult.sigma_ar_gerber` (new last field, default None). · safety engineer
- 2026-11-03 · optimize.py · ADDED helpers `with_dims(model, dims)`, `resolve_bounds(model, problem)` (default bounds 0.2×–5× current dims when not given), `objective_value`, `verify(model, problem, bounds=None) -> list[ConstraintCheck]`. A Euler-buckling constraint (FOS ≥ fos_min) is enforced automatically when the member carries axial compression with identifiable end conditions. `pareto_front` returns an extra `material_key` column; its first Pareto axis is `problem.objective` (volume / mass / cost), and every material uses its own library cost/kg. · optimization engineer
- 2026-11-03 · advisor.py · `Recommendation.quantity == 'mass'` (material-excess case): `current` = current mass, and `limit` = the mass reached by the verified downsized design. Constants `EXCESS_UTILISATION = 0.5` and `EXCESS_TARGET = 0.9` are stated assumptions. · advisor engineer
- 2026-11-03 · viz3d.py (new) · `beam_figure(result, amplification=None, show_undeformed=True, show_deformed=True, show_stress=True, show_supports=True, show_loads=True, show_markers=True, height=560) -> go.Figure`, `auto_amplification(result, target=0.08) -> float`, `DISCLAIMER`. Plotly 3D built only from `AnalysisResult` fields: v, theta, the frame u, and σ = N/A − My/I. Plot axes: X = x, Y = width z, Z = y. No Streamlit import. · lead UI engineer
- 2026-11-03 · ui/plots_core.py (new, UI-only helpers, free to reuse) · `require_result(model=None, **kw)` (analyse or show the error and `st.stop()`), `show_warnings`, cached `fem_solve(json, n, theory)` / `fem_convergence(json, theory)` / `fem_compare(json, n)`, `beam_schematic(model, mesh=None)`, `section_figure`, `section_props_metrics`, `field_figure` / `field_trace` / `extreme_marker`, `status_badge(status)` + `STATUS_COLOR` (covers BUCKLING PREDICTED, which `theme.badge` does not), `model_caption`, `dims_text`. · lead UI engineer
- 2026-11-03 · session keys · About page stores the editable placeholders in `st.session_state['about_team' | 'about_institute' | 'about_guide' | 'about_course']` (persist_state="session"; default "[Team members]" etc.). The Reports page may read them for `ReportMeta`. Design Studio's private keys use the prefix `ds_`; it auto-saves to `state.set_model` whenever its inputs form a valid model. · lead UI engineer
- 2026-11-03 · lab.py · additions only. `Experiment` gains defaulted fields `hypothesis, x, y, log_axes, group, plot` (plot metadata: table columns to draw). `ExperimentResult` gains the same, plus `initial_conditions: dict[str,str]`, `findings: dict[str,float]` (fitted slopes/ratios) and `parameter_changes: dict[str,str]`. `Challenge` gains `shapes` (allowed shapes), `bounds` (shape → dim → (lo, hi) m), `min_wall`, `beat_material` (design must be lighter than that material's verified optimum). `ChallengeGrade.reference` (optimize result, default None). `tornado()` adds an `elasticity` column = ln(Y+/Y−)/ln(X+/X−) (exactly n for Y ∝ Xⁿ) and skips parameters whose ±pct change is invalid. New helpers: `COLUMNS`, `display_column`, `display_table`, `default_base_model`, `total_load`, `run_case`, `describe`, `build_beam`, `support_set`, `SUPPORT_CONFIGS`, `LOAD_TYPES`, `challenge_bounds`, `challenge_design`, `reference_optimum`, `utilisation`, `challenge_report`, `sensitivity_params`, `sensitivity_report`, `numerical_comparison`, `CHALLENGE_BY_KEY`, `SCORE_RULE`, `OUTPUTS`, `PARAM_LABELS`. Lab PDFs: `smartbeam/ui/lab_plots.lab_pdf(result, meta)` → `report.build_lab_report`. · lab engineer
- 2026-11-03 · report.py · ADDED (nothing renamed or removed): `build_lab_report(title, objective, initial_conditions: dict, equations: list[str], parameter_changes, results_df, figures: list[matplotlib Figure], conclusions: list[str], meta=None) -> bytes` for the Virtual Lab. `parameter_changes` may be a str, list, dict or DataFrame. LaTeX equations are rendered with matplotlib mathtext; anything mathtext cannot parse is printed verbatim. `lab_report_pdf(experiment_result)` wraps it. Also new: `summary_csv(model, result, safety=None) -> str` (quantity, value, unit rows in SI); `format_check(ConstraintCheck) -> (name, value, limit, unit)`, which shows "(m)" constraints in mm and is shared by the PDF and the UI; the figure builders `fig_sfd_bmd`, `fig_deflection`, `fig_stress`, `fig_strain`, `fig_fem_vs_analytical`, `fig_convergence` (matplotlib `Figure`, no pyplot); and `GENERAL_ASSUMPTIONS`. `fem_csv` adds `Rx_N, Ry_N, M_Nm, support` columns, filled only at restrained nodes. `build_pdf` with `analysis=None` emits the model sections only (geometry, material, loads, supports, section properties). PDF text uses matplotlib's bundled DejaVu fonts, so σ/δ/₹ render correctly. · reports engineer
- 2026-11-03 · ui · NEW `smartbeam/ui/design_tools.py` (helpers for Design Comparison / Optimization / Advisor / Reports). Page-private session keys: `optimization_last` {model, problem, result} (the Reports page reads it to include the optimization section, but only when `model` equals the current model JSON), `optimization_applied`, `optimization_pareto`, `advisor_criteria` (δ_allow m, min FOS; the Reports page reuses it), `advisor_applied`, `comparison_rates` (material name → ₹/kg), `comparison_import_id`, `comparison_import_msg`, `reports_pdf`. · reports engineer
- 2026-11-03 · playground (new package, additions only) · `smartbeam/playground/convert.py` (engine-side, no Streamlit): `model_to_config(model, units=None)`, `config_to_model(config) -> BeamModel` (raises `ModelError` listing every problem), `PRESETS` (6 keys: `cantilever_end`, `cantilever_udl`, `ss_central`, `ss_udl`, `ss_partial_udl`, `ss_multi_point`; steel, rect 40×80 mm, L = 2 m), `preset_model`, `preset_configs`, `result_payload(result)`, `accept_message(state, value)`, `save_config/list_configs/load_config` (files in `saved_configs/`). The canvas config is the BeamModel JSON schema plus UI-only keys `id` (on supports/loads), `material_key` and `units`; the converter strips them. `smartbeam/playground/component.py`: `beam_canvas(...)` wraps `declare_component("smartbeam_canvas", path=frontend/)`. Session keys owned by the playground page: `playground_*`. · frontend/canvas engineer
- 2026-11-03 · model.py · behaviour fix, no API change (test & validation engineer): `PointLoad.fx/.fy` are now exact at multiples of 90° (`model._cos_sin`). Before, cos(−90°) = 6e-17 gave every vertical load a ~1e-13 N axial component, so N(x) and σ = N/A were non-zero on a beam with no stress, and `metrics.evaluate`/`safety.yield_check` reported FOS ≈ 7e18 instead of inf (contract §5 metrics). Regression tests are in `tests/test_validation.py` §21.
- 2026-11-03 · ui/plots_core.py · additions only: `extreme_marker(..., L=None)` (label above a positive and below a negative extreme, leaning inward near the span ends when `L` is given), and new `label_range(y, ext_value, frac=0.2)`, which gives a y-axis range with headroom for that label. `field_figure` uses both. `section_props_metrics` now lays its 4 metrics out 2×2. · integration & QA lead
- 2026-11-03 · viz3d.py · behaviour only: the scene uses `aspectmode="manual"` with true proportions normalised to the longest axis. Plotly's `"data"` mode normalises by the geometric mean, which put the camera inside the scene for slender beams. · integration & QA lead
- 2026-11-03 · ui/theme.py · metric labels are no longer `text-transform: uppercase` (it turned σ into Σ). Metric values and labels wrap instead of being cut off with an ellipsis. `_BADGE` covers `BUCKLING PREDICTED`. app.py: `st.navigation(..., expanded=True)` so all 14 pages are listed. · integration & QA lead
