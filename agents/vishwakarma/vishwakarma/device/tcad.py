# vishwakarma/device/tcad.py
#
# Drift-diffusion device simulation with DEVSIM, parameterised by a
# MaterialParams set (see params.py) instead of DEVSIM's hard-coded silicon.
#
# Devices (v0):
#   pn_diode_1d  -- abrupt p-n junction, SRH recombination, ohmic contacts
#   nmosfet_2d   -- planar n-channel MOSFET, metal gate on oxide, p-type body
#
# Physics included: Poisson + electron/hole continuity (Scharfetter-Gummel),
# Boltzmann statistics, SRH recombination with mid-gap traps, constant
# (low-field) mobility. NOT included yet: field- or doping-dependent mobility,
# velocity saturation, surface-roughness scattering, quantum confinement,
# Fermi-Dirac statistics, band-gap narrowing, heterojunctions, interface or
# oxide charge, ferroelectric polarisation. Results are long-channel /
# low-field prototypes; every result reports this list.
#
# DEVSIM keeps global state and is not thread-safe, so each simulation runs in
# a fresh spawned process (run_isolated). That also keeps a solver crash from
# taking down the API or the MCP server.
#
# Units follow DEVSIM: cm, cm^-3, cm^2/(V s), F/cm. 1D currents are A/cm^2;
# 2D currents are A/cm of device width and are reported as A/um.

from __future__ import annotations

import math
import os
from typing import Optional

from .params import MaterialParams, Q_E, EPS0
from . import analytic

PHYSICS_LIMITS = [
    "constant low-field mobility (no doping/field dependence, no velocity saturation)",
    "Boltzmann statistics (invalid for degenerate doping near contacts)",
    "SRH recombination with mid-gap traps only (no Auger, radiative, or tunnelling)",
    "no band-gap narrowing, no quantum confinement, no interface/oxide charge",
    "homojunction only (single semiconductor region)",
]

DEFAULT_MATH_LIBS = "liblapack.so.3:libblas.so.3"


# ─── process isolation ──────────────────────────────────────────────────────

def _worker(kind: str, spec: dict) -> dict:
    """Runs inside the child process (see _worker_main)."""
    if kind == "pn_diode_1d":
        return _pn_diode_1d(**spec)
    if kind == "nmosfet_2d":
        return _nmosfet_2d(**spec)
    raise ValueError(f"unknown device kind {kind!r}")


def _worker_main(argv: list) -> None:
    """python -m vishwakarma.device.tcad <spec.json> <result.json>"""
    import json, traceback
    spec_path, out_path = argv[1], argv[2]
    with open(spec_path) as f:
        job = json.load(f)
    try:
        res = _worker(job["kind"], job["spec"])
    except Exception as exc:
        res = {"success": False, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-3000:]}
    with open(out_path, "w") as f:
        json.dump(res, f)


def run_isolated(kind: str, spec: dict, timeout: Optional[float] = 1800) -> dict:
    """
    Run one simulation in a fresh interpreter. A subprocess (not
    multiprocessing) so it works whatever the parent's __main__ is (uvicorn,
    the stdio MCP server, a REPL), and so DEVSIM's C-level stdout chatter goes
    to a log file instead of the MCP JSON-RPC stream.
    """
    import json, subprocess, sys, tempfile
    from pathlib import Path
    pkg_root = str(Path(__file__).resolve().parents[2])      # dir containing vishwakarma/
    env = dict(os.environ)
    env.setdefault("DEVSIM_MATH_LIBS", DEFAULT_MATH_LIBS)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with tempfile.TemporaryDirectory(prefix="vk_tcad_") as td:
        sp_, rp_, lp_ = (Path(td) / n for n in ("spec.json", "result.json", "solver.log"))
        sp_.write_text(json.dumps({"kind": kind, "spec": spec}))
        with open(lp_, "w") as log:
            try:
                proc = subprocess.run([sys.executable, "-m", "vishwakarma.device.tcad",
                                       str(sp_), str(rp_)], stdout=log,
                                      stderr=subprocess.STDOUT, env=env, cwd=td,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                return {"success": False, "error": f"simulation exceeded {timeout} s",
                        "solver_log_tail": lp_.read_text(errors="replace")[-3000:]}
        tail = lp_.read_text(errors="replace")[-3000:]
        if not rp_.exists():
            return {"success": False, "error": f"solver process exited with code "
                    f"{proc.returncode} and wrote no result", "solver_log_tail": tail}
        res = json.loads(rp_.read_text())
        res["solver_log_tail"] = tail
        return res


# ─── shared DEVSIM setup ────────────────────────────────────────────────────

def _set_semiconductor_params(device: str, region: str, m: MaterialParams) -> None:
    import devsim as ds
    T = m.T_K
    ni = m.ni()
    for name, val in (
        ("Permittivity", m.v("eps_r") * EPS0),
        ("ElectronCharge", Q_E),
        ("n_i", ni),
        ("T", T),
        ("kT", 1.380649e-23 * T),
        ("V_t", m.Vt),
        ("mu_n", m.v("mu_n_cm2Vs")),
        ("mu_p", m.v("mu_p_cm2Vs")),
        ("n1", ni), ("p1", ni),                      # mid-gap SRH traps
        ("taun", m.v("srh_lifetime_n_s")),
        ("taup", m.v("srh_lifetime_p_s")),
    ):
        ds.set_parameter(device=device, region=region, name=name, value=val)


def _bias_name(contact: str) -> str:
    from devsim.python_packages.simple_physics import GetContactBiasName
    return GetContactBiasName(contact)


def _current(device: str, contact: str) -> float:
    import devsim as ds
    return (ds.get_contact_current(device=device, contact=contact,
                                   equation="ElectronContinuityEquation")
            + ds.get_contact_current(device=device, contact=contact,
                                     equation="HoleContinuityEquation"))


def _solve(rel=1e-10, abs_=1e10, it=40):
    import devsim as ds
    ds.solve(type="dc", absolute_error=abs_, relative_error=rel, maximum_iterations=it)


def _ramp(device: str, contact: str, target: float, step: float,
          min_step: float = 1e-4, rel=1e-10, abs_=1e10) -> None:
    """Move a contact bias to target with step halving on non-convergence."""
    import devsim as ds
    name = _bias_name(contact)
    v = ds.get_parameter(device=device, name=name)
    step = abs(step)
    while abs(target - v) > 1e-12:
        nxt = v + math.copysign(min(step, abs(target - v)), target - v)
        ds.set_parameter(device=device, name=name, value=nxt)
        try:
            _solve(rel, abs_)
            v = nxt
            step = min(step * 1.5, abs(step) * 4)
        except ds.error:
            ds.set_parameter(device=device, name=name, value=v)
            step *= 0.5
            if step < min_step:
                raise RuntimeError(f"bias ramp on {contact} stalled at {v:.4f} V")
            _solve(rel, abs_)


# ─── 1D p-n diode ───────────────────────────────────────────────────────────

def _pn_diode_1d(material: dict, NA: float, ND: float,
                 length_p_cm: float, length_n_cm: float,
                 v_forward: list, v_reverse: list) -> dict:
    import devsim as ds
    from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
    from devsim.python_packages import simple_physics as sp

    m = MaterialParams.from_dict(material)
    dev, reg, mesh = "diode", "semi", "dio"
    xj = length_p_cm
    xe = length_p_cm + length_n_cm
    W0 = analytic.depletion_width(m, NA, ND, 0.0)

    ds.create_1d_mesh(mesh=mesh)
    lines = [(0.0, min(length_p_cm / 50, 1e-4), "anode"),
             (xj - 3 * W0, W0 / 40, None),
             (xj, min(W0 / 200, 1e-7), "jn"),
             (xj + 3 * W0, W0 / 40, None),
             (xe, min(length_n_cm / 50, 1e-4), "cathode")]
    for pos, ps, tag in lines:
        if tag:
            ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps, tag=tag)
        elif 0 < pos < xe:
            ds.add_1d_mesh_line(mesh=mesh, pos=pos, ps=ps)
    ds.add_1d_contact(mesh=mesh, name="anode", tag="anode", material="metal")
    ds.add_1d_contact(mesh=mesh, name="cathode", tag="cathode", material="metal")
    ds.add_1d_region(mesh=mesh, material="semi", region=reg, tag1="anode", tag2="cathode")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    nnodes = len(ds.get_node_model_values(device=dev, region=reg, name="x"))

    _set_semiconductor_params(dev, reg, m)
    CreateNodeModel(dev, reg, "Acceptors", f"{NA:.6e}*step({xj:.9e}-x)")
    CreateNodeModel(dev, reg, "Donors", f"{ND:.6e}*step(x-{xj:.9e})")
    CreateNodeModel(dev, reg, "NetDoping", "Donors-Acceptors")

    CreateSolution(dev, reg, "Potential")
    sp.CreateSiliconPotentialOnly(dev, reg)
    for c in ("anode", "cathode"):
        ds.set_parameter(device=dev, name=_bias_name(c), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, c)
    _solve(rel=1e-10, abs_=1.0, it=50)

    CreateSolution(dev, reg, "Electrons")
    CreateSolution(dev, reg, "Holes")
    ds.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
    ds.set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
    sp.CreateSiliconDriftDiffusion(dev, reg)
    for c in ("anode", "cathode"):
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, c)
    _solve(it=50)

    iv = []
    for branch in (sorted(v_forward), sorted(v_reverse, reverse=True)):
        _ramp(dev, "anode", 0.0, 0.05)
        for v in branch:
            _ramp(dev, "anode", v, 0.05)
            iv.append({"V": v, "J_A_cm2": _current(dev, "anode")})
    iv.sort(key=lambda r: r["V"])

    check = analytic.diode_check(m, NA, ND, length_p_cm, length_n_cm, iv)
    return {"success": True, "device": "pn_diode_1d", "mesh_nodes": nnodes,
            "iv": iv, "analytic_check": check}


# ─── 2D n-channel MOSFET ────────────────────────────────────────────────────

def _gate_contact(dev: str, region: str, contact: str) -> None:
    """Metal gate on oxide: Potential = Vg - phi_ms (work-function difference)."""
    import devsim as ds
    from devsim.python_packages.model_create import (
        CreateContactNodeModel, CreateContactNodeModelDerivative,
        CreateEdgeModel, CreateEdgeModelDerivatives, InEdgeModelList)
    eq = f"Potential - {_bias_name(contact)} + phi_ms"
    name = f"{contact}nodemodel"
    CreateContactNodeModel(dev, contact, name, eq)
    CreateContactNodeModelDerivative(dev, contact, name, eq, "Potential")
    if not InEdgeModelList(dev, region, "contactcharge_edge"):
        CreateEdgeModel(dev, region, "contactcharge_edge", "Permittivity*ElectricField")
        CreateEdgeModelDerivatives(dev, region, "contactcharge_edge",
                                   "Permittivity*ElectricField", "Potential")
    ds.contact_equation(device=dev, contact=contact, name="PotentialEquation",
                        node_model=name, edge_charge_model="contactcharge_edge")


def _nmosfet_2d(material: dict, NA_body: float, N_sd: float, L_gate_cm: float,
                tox_cm: float, eps_ox: float, gate_workfunction_eV: float,
                vd_lin: float, vg_sweep: list, vd_sweep: list,
                vg_output: list, sd_junction_depth_cm: float) -> dict:
    import devsim as ds
    from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
    from devsim.python_packages import simple_physics as sp

    m = MaterialParams.from_dict(material)
    dev, mesh = "nmos", "mos"
    L_sd = max(0.5e-4, 0.25 * L_gate_cm)          # source/drain extension length
    gap = 0.1e-4                                  # free surface between contact and gate
    x_s, x_d = L_sd, L_sd + L_gate_cm
    x_end = x_d + L_sd
    Wdmax = analytic.max_depletion_width(m, NA_body)
    depth = max(1.0e-4, 6 * Wdmax, 4 * sd_junction_depth_cm)
    air = 1e-7
    yj = sd_junction_depth_cm
    sig = 0.2 * yj                                # lateral straggle of S/D profile

    ds.create_2d_mesh(mesh=mesh)
    xl = [(-air, 1e-5), (0.0, 1e-5), (x_s - 2 * sig, 2e-6), (x_s, 5e-7),
          (x_s + 4 * sig, 2e-6), (0.5 * (x_s + x_d), min(L_gate_cm / 20, 2e-5)),
          (x_d - 4 * sig, 2e-6), (x_d, 5e-7), (x_d + 2 * sig, 2e-6),
          (x_end, 1e-5), (x_end + air, 1e-5)]
    for pos, ps in sorted(set(xl)):
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=pos, ps=ps)
    yl = [(-tox_cm - air, tox_cm / 3), (-tox_cm, tox_cm / 3), (0.0, 1e-7),
          (2e-6, 4e-7), (yj, yj / 10), (Wdmax, Wdmax / 10), (depth, depth / 8),
          (depth + air, depth / 8)]
    for pos, ps in sorted(set(yl)):
        ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=pos, ps=ps)

    ds.add_2d_region(mesh=mesh, material="Air", region="air")
    ds.add_2d_region(mesh=mesh, material="semi", region="bulk",
                     xl=0.0, xh=x_end, yl=0.0, yh=depth)
    ds.add_2d_region(mesh=mesh, material="oxide", region="oxide",
                     xl=x_s, xh=x_d, yl=-tox_cm, yh=0.0)
    ds.add_2d_contact(mesh=mesh, name="gate", material="metal", region="oxide",
                      xl=x_s, xh=x_d, yl=-tox_cm, yh=-tox_cm)
    ds.add_2d_contact(mesh=mesh, name="source", material="metal", region="bulk",
                      xl=-air, xh=x_s - gap, yl=0.0, yh=0.0)
    ds.add_2d_contact(mesh=mesh, name="drain", material="metal", region="bulk",
                      xl=x_d + gap, xh=x_end + air, yl=0.0, yh=0.0)
    ds.add_2d_contact(mesh=mesh, name="body", material="metal", region="bulk",
                      yl=depth, yh=depth)
    ds.add_2d_interface(mesh=mesh, name="bulk_oxide", region0="bulk", region1="oxide")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=dev)
    nnodes = sum(len(ds.get_node_model_values(device=dev, region=r, name="x"))
                 for r in ("bulk", "oxide"))

    # parameters
    _set_semiconductor_params(dev, "bulk", m)
    ds.set_parameter(device=dev, region="oxide", name="Permittivity", value=eps_ox * EPS0)
    ds.set_parameter(device=dev, region="oxide", name="ElectronCharge", value=Q_E)
    phi_ms = gate_workfunction_eV - m.intrinsic_workfunction()
    ds.set_parameter(device=dev, name="phi_ms", value=phi_ms)

    # doping: uniform p body + Gaussian-in-depth, erfc-lateral n+ source/drain
    CreateNodeModel(dev, "bulk", "Acceptors", f"{NA_body:.6e}")
    # NB: DEVSIM's parser binds unary minus tighter than ^, so "-(a)^2" is
    # (+a^2). Write the Gaussian exponent as -1*((a)^2).
    sd = (f"{N_sd:.6e}*exp(-1*((y/{yj:.6e})^2))*0.5*("
          f"erfc((x-{x_s:.9e})/{sig:.6e}) + erfc(({x_d:.9e}-x)/{sig:.6e}))")
    CreateNodeModel(dev, "bulk", "Donors", sd)
    CreateNodeModel(dev, "bulk", "NetDoping", "Donors-Acceptors")
    dmax = max(ds.get_node_model_values(device=dev, region="bulk", name="Donors"))
    if not (0.5 * N_sd <= dmax <= 1.01 * N_sd):
        raise RuntimeError(f"S/D doping profile evaluates to max {dmax:.3e}, expected "
                           f"~{N_sd:.3e} -- doping expression is wrong")

    # potential-only equilibrium
    CreateSolution(dev, "bulk", "Potential")
    CreateSolution(dev, "oxide", "Potential")
    sp.CreateSiliconPotentialOnly(dev, "bulk")
    sp.CreateOxidePotentialOnly(dev, "oxide", "log_damp")
    for c in ("source", "drain", "body", "gate"):
        ds.set_parameter(device=dev, name=_bias_name(c), value=0.0)
    for c in ("source", "drain", "body"):
        sp.CreateSiliconPotentialOnlyContact(dev, "bulk", c)
    _gate_contact(dev, "oxide", "gate")
    sp.CreateSiliconOxideInterface(dev, "bulk_oxide")
    # Starting guess: local charge neutrality in the semiconductor, a linear
    # drop across the oxide from the body value to the gate value. From an
    # all-zero start the damped Newton steps (log_damp caps updates near V_t)
    # need far more than 50 iterations on the 2D mesh.
    psi_body = -m.Vt * math.asinh(NA_body / (2 * m.ni()))
    CreateNodeModel(dev, "bulk", "Psi0", "V_t*asinh(NetDoping/(2*n_i))")
    ds.set_node_values(device=dev, region="bulk", name="Potential", init_from="Psi0")
    CreateNodeModel(dev, "oxide", "Psi0",
                    f"{psi_body:.9e} + ({-phi_ms - psi_body:.9e})*(-y/{tox_cm:.9e})")
    ds.set_node_values(device=dev, region="oxide", name="Potential", init_from="Psi0")
    _solve(rel=1e-12, abs_=1e-13, it=100)

    # drift-diffusion
    CreateSolution(dev, "bulk", "Electrons")
    CreateSolution(dev, "bulk", "Holes")
    ds.set_node_values(device=dev, region="bulk", name="Electrons", init_from="IntrinsicElectrons")
    ds.set_node_values(device=dev, region="bulk", name="Holes", init_from="IntrinsicHoles")
    sp.CreateSiliconDriftDiffusion(dev, "bulk")
    for c in ("source", "drain", "body"):
        sp.CreateSiliconDriftDiffusionAtContact(dev, "bulk", c)
    _solve(rel=1e-10, abs_=1e30, it=60)

    REL, ABS = 1e-10, 1e30
    to_A_per_um = 1e-4

    # transfer curve at low drain bias
    _ramp(dev, "drain", vd_lin, 0.025, rel=REL, abs_=ABS)
    transfer = []
    vgs = sorted(vg_sweep)
    _ramp(dev, "gate", vgs[0], 0.1, rel=REL, abs_=ABS)
    for vg in vgs:
        _ramp(dev, "gate", vg, 0.1, rel=REL, abs_=ABS)
        transfer.append({"Vg": vg, "Id_A_per_um": _current(dev, "drain") * to_A_per_um})

    # output curves
    output = []
    for vg in sorted(vg_output or []):
        _ramp(dev, "drain", vd_lin, 0.1, rel=REL, abs_=ABS)
        _ramp(dev, "gate", vg, 0.1, rel=REL, abs_=ABS)
        curve = []
        for vd in sorted(vd_sweep):
            _ramp(dev, "drain", vd, 0.1, rel=REL, abs_=ABS)
            curve.append({"Vd": vd, "Id_A_per_um": _current(dev, "drain") * to_A_per_um})
        output.append({"Vg": vg, "curve": curve})

    check = analytic.mosfet_check(m, NA_body, tox_cm, eps_ox, gate_workfunction_eV,
                                  L_gate_cm, vd_lin, transfer)
    return {"success": True, "device": "nmosfet_2d", "mesh_nodes": nnodes,
            "phi_ms_eV": phi_ms, "transfer_vd_lin": transfer, "output": output,
            "analytic_check": check}


# ─── public entry points (run in an isolated process) ──────────────────────

def _material_ready(m: MaterialParams) -> dict:
    problems = m.validate()
    if problems:
        raise ValueError("material parameters not usable: " + "; ".join(problems))
    return m.to_dict()


def _wrap(result: dict, m: MaterialParams, inputs: dict) -> dict:
    result["inputs"] = inputs
    result["material"] = m.name
    result["provenance"] = m.provenance_table()
    result["physics_limits"] = PHYSICS_LIMITS
    warn = list(m.warnings)
    if m.assumed_keys():
        warn.append("parameters tagged 'assumed' were used: " + ", ".join(m.assumed_keys()))
    chk = result.get("analytic_check") or {}
    if chk and not chk.get("pass", True):
        warn.append("analytic cross-check FAILED -- treat this result as suspect: "
                    + chk.get("summary", ""))
    result["warnings"] = warn
    return result


def pn_diode_1d(m: MaterialParams, NA: float = 1e17, ND: float = 1e16,
                length_p_cm: float = 0.03, length_n_cm: float = 0.03,
                v_forward=None, v_reverse=None, timeout: float = 600) -> dict:
    v_forward = v_forward or ([round(0.05 * i, 3) for i in range(1, 9)] +
                              [round(0.4 + 0.025 * i, 3) for i in range(1, 13)])
    v_reverse = v_reverse or [-0.5, -1.0, -2.0]
    inputs = dict(NA=NA, ND=ND, length_p_cm=length_p_cm, length_n_cm=length_n_cm,
                  v_forward=v_forward, v_reverse=v_reverse)
    spec = dict(material=_material_ready(m), **inputs)
    return _wrap(run_isolated("pn_diode_1d", spec, timeout), m, inputs)


def nmosfet_2d(m: MaterialParams, NA_body: float = 1e17, N_sd: float = 1e20,
               L_gate_cm: float = 2e-4, tox_cm: float = 1e-6, eps_ox: float = 3.9,
               gate_workfunction_eV: Optional[float] = None, vd_lin: float = 0.05,
               vg_sweep=None, vd_sweep=None, vg_output=None,
               sd_junction_depth_cm: float = 1e-5, timeout: float = 1800) -> dict:
    extra_warn = []
    if gate_workfunction_eV is None:
        gate_workfunction_eV = m.v("electron_affinity_eV")
        extra_warn.append("gate_workfunction_eV not given: assumed equal to the "
                          "electron affinity (n+ poly-like gate)")
    vt_est = analytic.mos_threshold(m, NA_body, tox_cm, eps_ox, gate_workfunction_eV)["Vt"]
    if vg_sweep is None:
        lo, hi = vt_est - 0.6, vt_est + 1.5
        vg_sweep = [round(lo + i * (hi - lo) / 42, 4) for i in range(43)]
    vd_sweep = vd_sweep or [round(0.1 * i, 2) for i in range(1, 21)]
    inputs = dict(NA_body=NA_body, N_sd=N_sd, L_gate_cm=L_gate_cm, tox_cm=tox_cm,
                  eps_ox=eps_ox, gate_workfunction_eV=gate_workfunction_eV,
                  vd_lin=vd_lin, vg_sweep=vg_sweep, vd_sweep=vd_sweep,
                  vg_output=vg_output or [], sd_junction_depth_cm=sd_junction_depth_cm)
    spec = dict(material=_material_ready(m), **inputs)
    res = _wrap(run_isolated("nmosfet_2d", spec, timeout), m, inputs)
    res["warnings"] = extra_warn + res["warnings"]
    return res


if __name__ == "__main__":
    import sys as _sys
    _worker_main(_sys.argv)
