# vishwakarma/device/params.py
#
# Material parameters for device simulation, every one carrying its provenance.
#
# The device layer sits between DFT (unit cell, eV, Bohr) and drift-diffusion
# (cm, cm^-3, cm^2/Vs). Most numbers a TCAD simulation needs do not come out
# of a plain DFT run -- the PBE gap is too small, lifetimes depend on defects,
# mobility needs a scattering time -- so each parameter records where it came
# from and the simulation result carries that table forward. The rule mirrors
# the TRL-5 "0 fabricated values" bar used for SHANI/GANESH: nothing reaches
# DEVSIM without a source string, and anything tagged "assumed" is surfaced
# as a warning on the result rather than silently used.

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

# CODATA 2018 exact / recommended values. DEVSIM's bundled simple_physics uses
# rounded constants (q = 1.6e-19, eps_si = 11.1); we set our own parameters.
Q_E    = 1.602176634e-19      # C
K_B    = 1.380649e-23         # J/K
EPS0   = 8.8541878128e-14     # F/cm
M_E    = 9.1093837015e-31     # kg
H_PL   = 6.62607015e-34       # J s

SOURCES = ("dft", "dft+boltztrap2", "literature", "user", "derived", "assumed")


@dataclass
class Param:
    value: float
    unit: str
    source: str                      # one of SOURCES
    ref: str = ""                    # job id, DOI, book, or free text
    note: str = ""

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.value is None or not math.isfinite(float(self.value)):
            raise ValueError(f"non-finite value for parameter ({self.unit}, {self.source})")
        self.value = float(self.value)
        if self.source in ("literature",) and not self.ref:
            raise ValueError("a literature value needs a ref (citation or URL)")


# Parameters drift-diffusion needs. Everything else (ni, D, L) is derived.
REQUIRED = {
    "band_gap_eV":            "eV",
    "electron_affinity_eV":   "eV",
    "eps_r":                  "",
    "Nc_cm3":                 "cm^-3",
    "Nv_cm3":                 "cm^-3",
    "mu_n_cm2Vs":             "cm^2/(V s)",
    "mu_p_cm2Vs":             "cm^2/(V s)",
    "srh_lifetime_n_s":       "s",
    "srh_lifetime_p_s":       "s",
}
OPTIONAL = {
    "m_cond_n":               "m_e",       # conductivity effective mass
    "m_cond_p":               "m_e",
    "tau_momentum_n_s":       "s",         # momentum relaxation time (mobility)
    "tau_momentum_p_s":       "s",
    "band_gap_dft_eV":        "eV",        # uncorrected DFT gap, informational
}


@dataclass
class MaterialParams:
    name: str
    T_K: float = 300.0
    params: dict = field(default_factory=dict)       # key -> Param
    warnings: list = field(default_factory=list)

    # ── construction ──────────────────────────────────────────────────────
    def set(self, key: str, value: float, source: str, ref: str = "",
            note: str = "", unit: Optional[str] = None) -> None:
        if key not in REQUIRED and key not in OPTIONAL:
            raise KeyError(f"unknown material parameter {key!r}")
        unit = unit or REQUIRED.get(key, OPTIONAL.get(key, ""))
        self.params[key] = Param(value, unit, source, ref, note)

    def merge(self, overrides: dict | None, default_source: str = "user") -> None:
        """
        Apply {key: value} or {key: {value, source, ref, note}} on top.
        A bare number is tagged with default_source ("user").
        """
        for key, spec in (overrides or {}).items():
            if isinstance(spec, dict):
                self.set(key, spec["value"], spec.get("source", default_source),
                         spec.get("ref", ""), spec.get("note", ""))
            else:
                self.set(key, spec, default_source)

    # ── access ────────────────────────────────────────────────────────────
    def v(self, key: str) -> float:
        return self.params[key].value

    def missing(self) -> list[str]:
        return [k for k in REQUIRED if k not in self.params]

    def validate(self) -> list[str]:
        """
        Return blocking problems. Ranges are physical sanity bounds, not
        tuning targets: a value outside them is almost certainly a unit slip
        (m^2 vs cm^2, Ry vs eV) rather than an exotic material.
        """
        problems = [f"missing required parameter: {k}" for k in self.missing()]
        bounds = {
            "band_gap_eV":          (0.05, 10.0),
            "electron_affinity_eV": (0.0, 7.0),
            "eps_r":                (1.0, 1000.0),
            "Nc_cm3":               (1e16, 1e22),
            "Nv_cm3":               (1e16, 1e22),
            "mu_n_cm2Vs":           (1e-4, 1e6),
            "mu_p_cm2Vs":           (1e-4, 1e6),
            "srh_lifetime_n_s":     (1e-15, 1.0),
            "srh_lifetime_p_s":     (1e-15, 1.0),
        }
        for k, (lo, hi) in bounds.items():
            if k in self.params and not (lo <= self.v(k) <= hi):
                problems.append(f"{k}={self.v(k):g} outside sanity range [{lo:g}, {hi:g}] "
                                f"({self.params[k].unit}) -- check units")
        return problems

    def assumed_keys(self) -> list[str]:
        return [k for k, p in self.params.items() if p.source == "assumed"]

    # ── derived quantities ────────────────────────────────────────────────
    @property
    def Vt(self) -> float:
        return K_B * self.T_K / Q_E

    def ni(self) -> float:
        """Intrinsic density, nondegenerate, from Nc, Nv, Eg (self-consistent)."""
        return math.sqrt(self.v("Nc_cm3") * self.v("Nv_cm3")) * \
            math.exp(-self.v("band_gap_eV") / (2 * self.Vt))

    def intrinsic_workfunction(self) -> float:
        """phi_i = chi + (Ec - Ei); Ei sits (kT/2) ln(Nc/Nv) below midgap."""
        return (self.v("electron_affinity_eV") + 0.5 * self.v("band_gap_eV")
                + 0.5 * self.Vt * math.log(self.v("Nc_cm3") / self.v("Nv_cm3")))

    def diffusivity(self, carrier: str) -> float:
        return self.Vt * self.v("mu_n_cm2Vs" if carrier == "n" else "mu_p_cm2Vs")

    def diffusion_length(self, carrier: str) -> float:
        tau = self.v("srh_lifetime_n_s" if carrier == "n" else "srh_lifetime_p_s")
        return math.sqrt(self.diffusivity(carrier) * tau)

    # ── serialisation ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = {
            "name": self.name, "T_K": self.T_K,
            "params": {k: asdict(p) for k, p in self.params.items()},
            "warnings": list(self.warnings),
            "missing": self.missing(),
        }
        if not self.missing():
            d["derived"] = {
                "ni_cm3": self.ni(),
                "intrinsic_workfunction_eV": self.intrinsic_workfunction(),
                "Dn_cm2s": self.diffusivity("n"), "Dp_cm2s": self.diffusivity("p"),
                "Ln_cm": self.diffusion_length("n"), "Lp_cm": self.diffusion_length("p"),
            }
        if self.assumed_keys():
            d["assumed"] = self.assumed_keys()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MaterialParams":
        m = cls(name=d.get("name", "material"), T_K=float(d.get("T_K", 300.0)))
        for k, p in (d.get("params") or {}).items():
            if isinstance(p, dict):
                m.set(k, p["value"], p.get("source", "user"), p.get("ref", ""),
                      p.get("note", ""), p.get("unit"))
            else:
                m.set(k, p, "user")
        m.warnings = list(d.get("warnings", []))
        return m

    def provenance_table(self) -> list[dict]:
        return [{"param": k, "value": p.value, "unit": p.unit, "source": p.source,
                 "ref": p.ref, "note": p.note} for k, p in self.params.items()]
