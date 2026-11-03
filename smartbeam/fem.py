"""1D line-element FEM (independent of the analytical engine). OWNER: FEM engineer.

Elements (no 2D continuum elements, ever):
* 'beam'  : 2-node Euler-Bernoulli or Timoshenko beam, nodal DOFs [v, theta].
* 'frame' : 2-node planar frame, nodal DOFs [u, v, theta] (axial bar + EB/Timoshenko bending).
Global DOF index of node i, local dof k: i * dofs_per_node + k   (beam: v=0, theta=1; frame: u=0, v=1, theta=2).
Signs follow docs/ARCHITECTURE.md (y up, CCW theta/moment positive, sagging M positive, tension N positive).

Method summary
* Mesh: nodes at every support, point load, moment and distributed-load end, each span subdivided uniformly.
* Timoshenko: interdependent-interpolation element (exact for a prismatic Timoshenko beam), phi = 12EI/(kappa G A Le^2).
  Its nodal rotation DOF is the cross-section rotation psi; psi = dv/dx only when phi = 0 (Euler-Bernoulli).
* Distributed loads: work-equivalent (consistent) nodal loads, integral of N_i q over the element with the element's
  own shape functions (3-point Gauss, exact for cubic shape x linear load).
* Internal forces: element end forces f = k_e u_e - f_e, then V, M inside each element by exact free-body
  equilibrium including the distributed load, so V and M are not limited by the cubic displacement field.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg

from smartbeam import sections
from smartbeam.model import (BeamModel, DistributedLoad, Extreme, ModelError, PointLoad, Reaction,
                             UnstableStructureError, classify, is_transverse_only, validate)

COND_LIMIT = 1e12  # condition number above which K_ff is treated as singular (mechanism)
THEORIES = ("euler", "timoshenko")
MECHANISM_MSG = "Mechanism: insufficient supports to prevent rigid-body motion"
# Key points closer than this (relative to L) share one node. A shorter element would be near-rigid next to its
# neighbours, and K u - F would lose most significant digits (reactions wrong by %). The shift is reported.
_MERGE_TOL = 1e-4


@dataclass
class Mesh:
    nodes: np.ndarray  # (n_nodes,) sorted x coordinates, m
    elements: np.ndarray  # (n_elements, 2) int node indices

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def lengths(self) -> np.ndarray:
        return self.nodes[self.elements[:, 1]] - self.nodes[self.elements[:, 0]]


@dataclass
class FEMResult:
    model: BeamModel
    theory: str  # 'euler' | 'timoshenko'
    element: str  # 'beam' | 'frame'
    mesh: Mesh
    dofs_per_node: int  # 2 (beam) | 3 (frame)
    U: np.ndarray  # (n_dof,) global displacement vector
    u_nodes: np.ndarray  # (n_nodes,) axial displacement (zeros for 'beam'), m
    v_nodes: np.ndarray  # (n_nodes,) deflection, m (+y)
    theta_nodes: np.ndarray  # (n_nodes,) rotation, rad (CCW +)
    reactions: list[Reaction]
    element_end_forces: np.ndarray  # (n_elements, 2*dofs_per_node): k_e u_e - f_e (forces ON element, global axes)
    # Dense reconstruction (Hermite cubic for v/theta inside each element; x repeated at element boundaries):
    x: np.ndarray
    v: np.ndarray
    theta: np.ndarray
    V: np.ndarray  # shear force, N   (sign: V = dM/dx)
    M: np.ndarray  # bending moment, N*m (sagging +)
    N: np.ndarray  # axial force, N (tension +; zeros for 'beam')
    sigma_top: np.ndarray  # Pa, N/A - M*y_top/I
    sigma_bottom: np.ndarray  # Pa, N/A - M*y_bottom/I
    v_max: Extreme
    theta_max: Extreme
    M_max: Extreme
    V_max: Extreme
    sigma_max: Extreme  # max |sigma| over both fibres
    condition_number: float  # of the reduced stiffness matrix K_ff (Jacobi-scaled, see solve_linear)
    n_dof: int
    n_free: int
    warnings: list[str] = field(default_factory=list)
    phi: np.ndarray | None = None  # (n_elements,) Timoshenko shear parameter per element (zeros for 'euler')
    N_max: Extreme | None = None  # largest |N| (tension +)


@dataclass
class ConvergenceResult:
    n_elements: list[int]
    theory: str
    v_max: np.ndarray  # |v|max per mesh, m
    theta_max: np.ndarray  # |theta|max, rad
    M_max: np.ndarray  # |M|max, N*m
    sigma_max: np.ndarray  # |sigma|max, Pa
    reference: dict[str, float] | None  # analytical values when a valid analytical solution exists
    error_pct: dict[str, np.ndarray] | None  # 100*|fem - ref|/|ref| per quantity
    note: str  # honest explanation (e.g. EB nodal values are exact for point loads -> flat curve)
    # |EI * curvature|max of the raw cubic displacement field (no equilibrium recovery). Unlike M_max this is
    # the "pure displacement-method" moment and genuinely converges under distributed loads.
    M_fe_max: np.ndarray | None = None
    n_requested: list[int] | None = None  # requested counts (meshes may be finer to put nodes at key points)


@dataclass
class TheoryComparison:
    """Euler-Bernoulli vs Timoshenko on the same mesh."""

    eb: FEMResult
    timoshenko: FEMResult
    v_eb: float  # |v|max, m
    v_timoshenko: float  # |v|max, m
    diff_pct: float  # 100 (|v_T| - |v_EB|) / |v_EB|  (shear-deformation share of the EB value)
    span_to_depth: float  # L / depth
    phi_beam: float  # 12 EI / (kappa G A L^2) for the whole span: shear-to-bending flexibility indicator
    kappa: float
    G: float  # Pa


# --------------------------------------------------------------------------- element matrices
def _bending_k(EI: float, Le: float, phi: float) -> np.ndarray:
    L, p = Le, phi
    return EI / ((1.0 + p) * L**3) * np.array([
        [12.0, 6 * L, -12.0, 6 * L],
        [6 * L, (4 + p) * L**2, -6 * L, (2 - p) * L**2],
        [-12.0, -6 * L, 12.0, -6 * L],
        [6 * L, (2 - p) * L**2, -6 * L, (4 + p) * L**2],
    ])


def eb_element_stiffness(E: float, I: float, Le: float) -> np.ndarray:
    """4x4 Euler-Bernoulli K_e = EI/Le^3 [[12,6L,-12,6L],[6L,4L^2,-6L,2L^2],[-12,-6L,12,-6L],[6L,2L^2,-6L,4L^2]]."""
    return _bending_k(E * I, Le, 0.0)


def timoshenko_phi(E: float, I: float, Le: float, G: float, A: float, kappa: float) -> float:
    """phi = 12 EI / (kappa G A Le^2): ratio of shear to bending flexibility of an element."""
    return 12.0 * E * I / (kappa * G * A * Le**2)


def timoshenko_element_stiffness(E: float, I: float, Le: float, G: float, A: float, kappa: float) -> np.ndarray:
    """4x4 interdependent-interpolation Timoshenko element, phi = 12 EI / (kappa G A Le^2):
    K_e = EI/((1+phi) Le^3) [[12,6L,-12,6L],[6L,(4+phi)L^2,-6L,(2-phi)L^2],[-12,-6L,12,-6L],[6L,(2-phi)L^2,-6L,(4+phi)L^2]].
    phi -> 0 recovers eb_element_stiffness."""
    return _bending_k(E * I, Le, timoshenko_phi(E, I, Le, G, A, kappa))


def frame_element_stiffness(E: float, A: float, I: float, Le: float, phi: float = 0.0) -> np.ndarray:
    """6x6 planar frame element, DOFs [u1,v1,theta1,u2,v2,theta2]: axial EA/Le block + bending block
    (EB when phi=0, Timoshenko otherwise). Elements are collinear with x, so no rotation transform."""
    k = np.zeros((6, 6))
    ka = E * A / Le
    k[np.ix_([0, 3], [0, 3])] = ka * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k[np.ix_(_BEND_IN_FRAME, _BEND_IN_FRAME)] = _bending_k(E * I, Le, phi)
    return k


_BEND_IN_FRAME = [1, 2, 4, 5]


def _shape_coeffs(Le: float, phi: float = 0.0) -> np.ndarray:
    """(4, 4) polynomial coefficients C[i] of the transverse shape function N_i(xi) = sum_k C[i,k] xi^k, xi = x/Le.
    Interdependent interpolation (exact homogeneous Timoshenko solution); phi = 0 gives the Hermite cubics."""
    p = phi
    return np.array([
        [1 + p, -p, -3.0, 2.0],
        [0.0, Le * (1 + p / 2), -Le * (2 + p / 2), Le],
        [0.0, p, 3.0, -2.0],
        [0.0, -Le * p / 2, Le * (-1 + p / 2), Le],
    ]) / (1 + p)


_GAUSS_XI, _GAUSS_W = np.polynomial.legendre.leggauss(3)
_GAUSS_XI, _GAUSS_W = 0.5 * (_GAUSS_XI + 1.0), 0.5 * _GAUSS_W  # mapped to [0, 1]


def consistent_load(q1: float, q2: float, Le: float, phi: float = 0.0) -> np.ndarray:
    """Consistent nodal loads [F1, M1, F2, M2] of a linearly varying +y intensity q1 -> q2 (N/m) on an element:
    Le*[(7q1+3q2)/20, Le(3q1+2q2)/60, (3q1+7q2)/20, -Le(2q1+3q2)/60]   (phi = 0, Euler-Bernoulli).
    With phi > 0 the element's own (Timoshenko) shape functions are integrated instead, which keeps nodal values
    exact. 3-point Gauss quadrature is exact here (cubic shape function x linear load)."""
    xi = _GAUSS_XI
    q = q1 + (q2 - q1) * xi
    N = _shape_coeffs(Le, phi) @ np.vstack([np.ones_like(xi), xi, xi**2, xi**3])  # (4, n_gauss)
    return Le * (N * q) @ _GAUSS_W


# --------------------------------------------------------------------------- mesh
def _key_points(model: BeamModel) -> list[float]:
    pts = [0.0, model.L] + [s.x for s in model.supports] + [ld.x for ld in model.point_loads]
    pts += [ld.x for ld in model.moment_loads]
    for ld in model.distributed_loads:
        pts += [ld.x1, ld.x2]
    return pts


def build_mesh(model: BeamModel, n_elements: int) -> Mesh:
    """Uniform mesh of at least n_elements, with nodes inserted at every support, point load, moment
    and distributed-load start/end so loads/BCs act at nodes. Actual element count may exceed n_elements."""
    L = model.L
    n = max(1, int(n_elements))
    keys = np.sort(np.clip(_key_points(model), 0.0, L))
    merged = [keys[0]]
    for k in keys[1:]:
        if k - merged[-1] > _MERGE_TOL * L:
            merged.append(k)
    merged[-1] = L
    h = L / n
    nodes = [merged[0]]
    for a, b in zip(merged[:-1], merged[1:]):
        m = max(1, math.ceil((b - a) / h - 1e-9))
        nodes.extend(np.linspace(a, b, m + 1)[1:])
    nodes = np.array(nodes)
    idx = np.arange(len(nodes) - 1)
    return Mesh(nodes, np.column_stack([idx, idx + 1]))


def _node_at(mesh: Mesh, x: float) -> int:
    return int(np.argmin(np.abs(mesh.nodes - x)))


# --------------------------------------------------------------------------- assembly
def _check_theory(theory: str) -> None:
    if theory not in THEORIES:
        raise ValueError(f"Unknown beam theory {theory!r}; expected one of {THEORIES}.")


def _phis(model: BeamModel, mesh: Mesh, theory: str) -> np.ndarray:
    if theory == "euler":
        return np.zeros(mesh.n_elements)
    s, mat = model.section, model.material
    kappa = sections.shear_correction_factor(s, mat.nu)
    return timoshenko_phi(mat.E, sections.second_moment(s), mesh.lengths, mat.G, sections.area(s), kappa)


def _element_q(model: BeamModel, xa: float, xb: float) -> tuple[float, float]:
    """Total +y intensity at the element ends from every distributed load covering the element."""
    q1 = q2 = 0.0
    mid = 0.5 * (xa + xb)
    for ld in model.distributed_loads:
        if ld.x1 <= mid <= ld.x2:
            slope = (ld.w2 - ld.w1) / (ld.x2 - ld.x1)
            q1 += ld.sign * (ld.w1 + slope * (xa - ld.x1))
            q2 += ld.sign * (ld.w1 + slope * (xb - ld.x1))
    return q1, q2


def _elements(model: BeamModel, mesh: Mesh, theory: str, frame: bool):
    """Per element: (global dofs, k_e, f_e, q1, q2, phi)."""
    E, A, I = model.material.E, sections.area(model.section), sections.second_moment(model.section)
    dpn = 3 if frame else 2
    out = []
    for (i, j), Le, phi in zip(mesh.elements, mesh.lengths, _phis(model, mesh, theory)):
        dofs = np.r_[i * dpn:(i + 1) * dpn, j * dpn:(j + 1) * dpn]
        q1, q2 = _element_q(model, mesh.nodes[i], mesh.nodes[j])
        fb = consistent_load(q1, q2, Le, phi)
        if frame:
            ke, fe = frame_element_stiffness(E, A, I, Le, phi), np.zeros(6)
            fe[_BEND_IN_FRAME] = fb
        else:
            ke, fe = _bending_k(E * I, Le, phi), fb
        out.append((dofs, ke, fe, q1, q2, phi))
    return out


def _assemble(model: BeamModel, mesh: Mesh, elems, frame: bool) -> tuple[np.ndarray, np.ndarray]:
    dpn = 3 if frame else 2
    n = mesh.n_nodes * dpn
    K, F = np.zeros((n, n)), np.zeros(n)
    for dofs, ke, fe, *_ in elems:
        K[np.ix_(dofs, dofs)] += ke
        F[dofs] += fe
    kv, kt = (1, 2) if frame else (0, 1)
    for ld in model.point_loads:
        node = _node_at(mesh, ld.x)
        F[node * dpn + kv] += ld.fy
        if frame:
            F[node * dpn] += ld.fx
    for ld in model.moment_loads:
        F[_node_at(mesh, ld.x) * dpn + kt] += ld.M
    return K, F


def assemble(model: BeamModel, mesh: Mesh, theory: str = "euler", frame: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Global stiffness K (n_dof x n_dof) and load vector F (n_dof,) incl. point loads, moments, consistent loads.
    frame=False (beam element) ignores the axial component of any point load."""
    _check_theory(theory)
    return _assemble(model, mesh, _elements(model, mesh, theory, frame), frame)


def restrained_dofs(model: BeamModel, mesh: Mesh, frame: bool = False) -> np.ndarray:
    """Global DOF indices fixed by the supports (beam element ignores u restraints)."""
    local = {"u": 0, "v": 1, "theta": 2} if frame else {"v": 0, "theta": 1}
    dpn = len(local)
    fixed = {_node_at(mesh, s.x) * dpn + local[d] for s in model.supports for d in s.restrained if d in local}
    return np.array(sorted(fixed), dtype=int)


# --------------------------------------------------------------------------- solve
def solve_linear(K: np.ndarray, F: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Partitioned solve K_ff U_f = F_f. Returns (U, R, cond) with R = K U - F (reaction vector, nonzero at fixed DOFs).
    Raises model.UnstableStructureError if K_ff is singular or cond(K_ff) > COND_LIMIT (rigid-body mode).

    cond is the 2-norm condition number of the Jacobi-scaled K_ff (D^-1/2 K_ff D^-1/2, D = diag K_ff). The scaling
    removes the artificial spread caused by mixing N/m, N*m/rad and EA vs EI terms, so the number measures how
    close K_ff is to singular rather than the choice of units."""
    n = len(F)
    fixed = np.unique(np.asarray(fixed, dtype=int))
    free = np.setdiff1d(np.arange(n), fixed)
    U = np.zeros(n)
    cond = 1.0
    if free.size:
        Kff = K[np.ix_(free, free)]
        d = np.diag(Kff)
        if np.any(d <= 0):
            raise UnstableStructureError(f"{MECHANISM_MSG} (zero pivot: a free DOF has no stiffness).")
        s = 1.0 / np.sqrt(d)
        Ks = Kff * np.outer(s, s)
        sv = np.linalg.svd(Ks, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * free.size * np.finfo(float).eps))
        cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
        if rank < free.size or cond > COND_LIMIT:
            raise UnstableStructureError(
                f"{MECHANISM_MSG}: the reduced stiffness matrix is singular (rank {rank} of {free.size}, "
                f"condition number {cond:.2e} > {COND_LIMIT:.0e}). Add or change supports.")
        try:
            U[free] = s * scipy.linalg.solve(Ks, s * F[free], assume_a="pos")
        except np.linalg.LinAlgError as exc:
            raise UnstableStructureError(f"{MECHANISM_MSG} (K_ff is not positive definite).") from exc
    R = K @ U - F
    R[free] = 0.0  # equilibrium residual at free DOFs is round-off only
    return U, R, cond


# --------------------------------------------------------------------------- post-processing
def _fields(mesh: Mesh, U: np.ndarray, frame: bool, n_per_element: int, phi):
    """Dense x, v, theta, curvature v'' inside every element from the element shape functions."""
    dpn = 3 if frame else 2
    kv = 1 if frame else 0
    phi = np.broadcast_to(np.asarray(0.0 if phi is None else phi, dtype=float), (mesh.n_elements,))
    xi = np.linspace(0.0, 1.0, max(1, int(n_per_element)) + 1)
    P = np.vstack([np.ones_like(xi), xi, xi**2, xi**3])
    dP = np.vstack([np.zeros_like(xi), np.ones_like(xi), 2 * xi, 3 * xi**2])
    ddP = np.vstack([np.zeros_like(xi), np.zeros_like(xi), 2 * np.ones_like(xi), 6 * xi])
    xs, vs, ts, ks = [], [], [], []
    for (i, j), Le, p in zip(mesh.elements, mesh.lengths, phi):
        d = U[[i * dpn + kv, i * dpn + kv + 1, j * dpn + kv, j * dpn + kv + 1]]
        c = _shape_coeffs(Le, p).T @ d  # polynomial coefficients of v(xi)
        xs.append(mesh.nodes[i] + Le * xi)
        vs.append(c @ P)
        # section rotation psi = v' + V/(kappa G A) = v' + (phi Le^2 / 12) v''' (= v' for EB), v''' = 6 c3 / Le^3.
        # (shear strain v' - psi = -V/(kappa G A) because V here is the sum of forces LEFT of the cut)
        ts.append((c @ dP) / Le + p * c[3] / (2 * Le))
        ks.append((c @ ddP) / Le**2)
    return np.concatenate(xs), np.concatenate(vs), np.concatenate(ts), np.concatenate(ks), xi


def reconstruct(mesh: Mesh, U: np.ndarray, frame: bool = False, n_per_element: int = 10,
                phi=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermite cubic reconstruction inside each element -> (x, v, theta) dense arrays.
    phi (scalar or per element, default 0) selects the Timoshenko interdependent shape functions; theta is then
    the section rotation psi."""
    x, v, theta, _, _ = _fields(mesh, U, frame, n_per_element, phi)
    return x, v, theta


def _internal_forces(mesh: Mesh, elems, end_forces: np.ndarray, frame: bool, xi: np.ndarray):
    """V, M, N inside each element by free-body equilibrium of the part [xa, x] with the exact linear load."""
    kF, kM = (1, 2) if frame else (0, 1)
    Vs, Ms, Ns = [], [], []
    for (dofs, ke, fe, q1, q2, phi), f, Le in zip(elems, end_forces, mesh.lengths):
        t = Le * xi
        V = f[kF] + q1 * t + (q2 - q1) * t**2 / (2 * Le)
        M = -f[kM] + f[kF] * t + q1 * t**2 / 2 + (q2 - q1) * t**3 / (6 * Le)
        Vs.append(V)
        Ms.append(M)
        Ns.append(np.full_like(t, -f[0] if frame else 0.0))
    return np.concatenate(Vs), np.concatenate(Ms), np.concatenate(Ns)


def solve(model: BeamModel, n_elements: int = 20, theory: str = "euler", frame: bool | None = None,
          n_per_element: int = 10) -> FEMResult:
    """Full FEM pipeline. frame=None -> auto: frame element iff not model.is_transverse_only(model).
    Validates the model (raises model.ModelError) and detects mechanisms (raises model.UnstableStructureError)."""
    errors = validate(model)
    if errors:
        raise ModelError("; ".join(errors))
    _check_theory(theory)
    transverse = is_transverse_only(model)
    frame = (not transverse) if frame is None else bool(frame)
    warnings: list[str] = []
    if not frame and not transverse:
        warnings.append("Beam elements carry no axial DOF: the axial components of inclined/axial loads were "
                        "ignored. Use frame elements for this load case.")

    mesh = build_mesh(model, n_elements)
    shift = max(abs(mesh.nodes[_node_at(mesh, x)] - x) for x in _key_points(model))
    if shift > 1e-9 * model.L:
        warnings.append(f"Supports/loads closer than {_MERGE_TOL:g} L were placed on a shared node "
                        f"(largest shift {shift:.3g} m).")
    elems = _elements(model, mesh, theory, frame)
    K, F = _assemble(model, mesh, elems, frame)
    fixed = restrained_dofs(model, mesh, frame)
    dpn = 3 if frame else 2
    has_u = frame and np.any(fixed % 3 == 0)
    if frame and not has_u and transverse:
        # No axial load and no axial restraint: the axial rigid-body mode carries nothing. Pin it (reaction = 0).
        fixed = np.append(fixed, _node_at(mesh, model.supports[0].x) * 3)
        warnings.append("No support restrains axial motion; with no axial load this is harmless, so u was held "
                        "at the first support (its reaction is zero).")
    try:
        U, R, cond = solve_linear(K, F, fixed)
    except UnstableStructureError as exc:
        if frame and not has_u:
            raise UnstableStructureError(f"{exc} No support restrains axial (u) motion: rollers cannot resist the "
                                         f"axial load component; add a pin or fixed support.") from exc
        raise
    if cond > 1e10:
        warnings.append(f"Stiffness matrix is poorly conditioned (cond = {cond:.1e}); results may lose accuracy.")

    end_forces = np.array([ke @ U[dofs] - fe for dofs, ke, fe, *_ in elems])
    phis = np.array([e[5] for e in elems])
    x, v, theta, _, xi = _fields(mesh, U, frame, n_per_element, phis)
    V, M, N = _internal_forces(mesh, elems, end_forces, frame, xi)

    s = model.section
    A, I = sections.area(s), sections.second_moment(s)
    sig_top = N / A - M * sections.y_top(s) / I
    sig_bot = N / A - M * sections.y_bottom(s) / I
    top = Extreme.of(x, sig_top, "top")
    bot = Extreme.of(x, sig_bot, "bottom")

    kv = 1 if frame else 0
    reacts, used = [], set()
    for k, sp in enumerate(model.supports):
        node = _node_at(mesh, sp.x)
        r = np.zeros(dpn) if node in used else R[node * dpn:(node + 1) * dpn]  # merged supports: report once
        used.add(node)
        reacts.append(Reaction(k, sp.type, sp.x, Fx=float(r[0]) if frame else 0.0,
                               Fy=float(r[kv]), M=float(r[kv + 1])))

    return FEMResult(
        model=model, theory=theory, element="frame" if frame else "beam", mesh=mesh, dofs_per_node=dpn, U=U,
        u_nodes=U[0::3] if frame else np.zeros(mesh.n_nodes), v_nodes=U[kv::dpn], theta_nodes=U[kv + 1::dpn],
        reactions=reacts, element_end_forces=end_forces, x=x, v=v, theta=theta, V=V, M=M, N=N,
        sigma_top=sig_top, sigma_bottom=sig_bot,
        v_max=Extreme.of(x, v), theta_max=Extreme.of(x, theta), M_max=Extreme.of(x, M), V_max=Extreme.of(x, V),
        sigma_max=top if abs(top.value) >= abs(bot.value) else bot,
        condition_number=cond, n_dof=len(U), n_free=len(U) - len(np.unique(fixed)), warnings=warnings,
        phi=phis, N_max=Extreme.of(x, N),
    )


# --------------------------------------------------------------------------- studies
def _textbook_reference(model: BeamModel, theory: str) -> tuple[str, dict[str, float]] | None:
    """Closed-form |v|max, |theta|max, |M|max, |sigma|max for the four textbook cases only (else None).
    Timoshenko adds the shear deflection term; slope (section rotation) and moment are unchanged."""
    kind = classify(model)
    if kind not in ("cantilever", "simply-supported") or len(model.loads) != 1:
        return None
    ld, L = model.loads[0], model.L
    s, mat = model.section, model.material
    E, I, A = mat.E, sections.second_moment(s), sections.area(s)
    kGA = sections.shear_correction_factor(s, mat.nu) * mat.G * A
    shear = theory == "timoshenko"
    tol = 1e-9 * L
    if isinstance(ld, PointLoad) and ld.is_transverse:
        P = abs(ld.fy)
        free_end = L if abs(model.supports[0].x) <= tol else 0.0
        if kind == "cantilever" and abs(ld.x - free_end) <= tol:
            case, v, th, M = "cantilever-end-load", P * L**3 / (3 * E * I) + shear * P * L / kGA, P * L**2 / (2 * E * I), P * L
        elif kind == "simply-supported" and abs(ld.x - L / 2) <= tol:
            case, v, th, M = "ss-central-load", P * L**3 / (48 * E * I) + shear * P * L / (4 * kGA), P * L**2 / (16 * E * I), P * L / 4
        else:
            return None
    elif isinstance(ld, DistributedLoad) and abs(ld.x1) <= tol and abs(ld.x2 - L) <= tol and abs(ld.w1 - ld.w2) <= 1e-12 * max(1.0, abs(ld.w1)):
        w = abs(ld.w1)
        if kind == "cantilever":
            case, v, th, M = "cantilever-udl", w * L**4 / (8 * E * I) + shear * w * L**2 / (2 * kGA), w * L**3 / (6 * E * I), w * L**2 / 2
        else:
            case, v, th, M = "ss-udl", 5 * w * L**4 / (384 * E * I) + shear * w * L**2 / (8 * kGA), w * L**3 / (24 * E * I), w * L**2 / 8
    else:
        return None
    return case, {"v_max": v, "theta_max": th, "M_max": M, "sigma_max": M * sections.y_top(s) / I}


def convergence_study(model: BeamModel, n_list=(2, 4, 8, 16, 32), theory: str = "euler") -> ConvergenceResult:
    """Run solve() for each mesh and compare with the analytical reference when one is valid."""
    runs = [solve(model, n_elements=n, theory=theory) for n in n_list]
    EI = model.material.E * sections.second_moment(model.section)
    vals = {
        "v_max": np.array([abs(r.v_max.value) for r in runs]),
        "theta_max": np.array([abs(r.theta_max.value) for r in runs]),
        "M_max": np.array([abs(r.M_max.value) for r in runs]),
        "sigma_max": np.array([abs(r.sigma_max.value) for r in runs]),
    }
    m_fe = np.array([EI * np.max(np.abs(_fields(r.mesh, r.U, r.element == "frame", 10, r.phi)[3])) for r in runs])
    counts = [r.mesh.n_elements for r in runs]

    ref = _textbook_reference(model, theory)
    reference = error = None
    lines = []
    if ref:
        case, reference = ref
        error = {k: 100 * np.abs(vals[k] - reference[k]) / abs(reference[k]) for k in vals}
        error["M_fe"] = 100 * np.abs(m_fe - reference["M_max"]) / abs(reference["M_max"])
        lines.append(f"Reference: closed-form {case} solution ({theory}).")
    else:
        lines.append("No closed-form reference applies to this configuration, so errors are not reported; "
                     "judge convergence by how little the values change between meshes.")
    lines.append("The element stiffness is exact for a prismatic beam and distributed loads use consistent "
                 "(work-equivalent) nodal loads, so nodal deflections and rotations are exact on every mesh; "
                 "V and M are recovered from element equilibrium and are exact too. Flat curves at round-off level "
                 "are the correct, expected result, not a sign that nothing was computed.")
    if model.distributed_loads:
        lines.append("Between nodes, v and θ come from the cubic interpolation, which does not see the load "
                     "inside the element, so the interpolated slope can overshoot the exact nodal peak slightly; that "
                     "error shrinks rapidly (about h³) with refinement.")
        lines.append("M_fe = EI v'' taken directly from the cubic displacement field is linear inside each element, "
                     "while the true moment under a distributed load is quadratic or cubic: M_fe converges as the "
                     "mesh is refined. That is the classic displacement-method convergence.")
    else:
        lines.append("With only point loads and moments the exact moment is linear between nodes, so even "
                     "M_fe = EI v'' is exact.")
    lines.append("A maximum that falls between nodes is read from the dense reconstruction, which converges with "
                 "refinement.")
    if counts != [int(n) for n in n_list]:
        lines.append(f"Element counts differ from the requested {list(n_list)}: nodes are always inserted at "
                     f"supports and load points.")
    return ConvergenceResult(n_elements=counts, theory=theory, reference=reference, error_pct=error,
                             note=" ".join(lines), M_fe_max=m_fe, n_requested=[int(n) for n in n_list], **vals)


def compare_theories(model: BeamModel, n_elements: int = 20, frame: bool | None = None) -> TheoryComparison:
    """Solve the same model with Euler-Bernoulli and Timoshenko elements and compare peak deflection."""
    eb = solve(model, n_elements=n_elements, theory="euler", frame=frame)
    ti = solve(model, n_elements=n_elements, theory="timoshenko", frame=frame)
    s, mat = model.section, model.material
    kappa = sections.shear_correction_factor(s, mat.nu)
    v_eb, v_ti = abs(eb.v_max.value), abs(ti.v_max.value)
    return TheoryComparison(
        eb=eb, timoshenko=ti, v_eb=v_eb, v_timoshenko=v_ti,
        diff_pct=100 * (v_ti - v_eb) / v_eb if v_eb > 0 else 0.0,
        span_to_depth=model.L / s.depth,
        phi_beam=timoshenko_phi(mat.E, sections.second_moment(s), model.L, mat.G, sections.area(s), kappa),
        kappa=kappa, G=mat.G,
    )
