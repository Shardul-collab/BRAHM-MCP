"""Reference DFT for the device-module validation: primitive (2-atom FCC) Si,
SCF 8x8x8 -> NSCF 16x16x16 (16 bands) -> dos.x, through the normal
Vishwakarma workflow. The NSCF save directory is what BoltzTraP2 reads.

Run:  QE_BIN_DIR=... .venv/bin/python validation/device/run_si_reference_dft.py
Writes the workflow result to validation/device/si_reference_dft.json.
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from vishwakarma import structure_builder as sb, workflow as wf

BIN = os.environ.get("QE_BIN_DIR", "/media/shardul/4202386902386457/miniforge3/bin")
WORK = os.environ.get("QE_WORKDIR", str(HERE.parents[1] / "jobs"))
PSEUDO = str(HERE.parents[1] / "pseudo")

s = sb.to_qe_structure(sb.build("Si"), prefix="si_dev")
s["kpoints"] = {"mode": "automatic", "mesh": [8, 8, 8], "shift": [0, 0, 0]}
p = sb.suggest_calc_params(sb.build("Si"), pseudo_dir=PSEUDO)["suggested_params"]
p.update(conv_thr=1e-10, outdir="./out", nbnd=16)

res = wf.dos_workflow(s, p, dense_kmesh=[16, 16, 16], emin=-10, emax=15,
                      label="si_device_ref", workdir=WORK, bin_dir=BIN,
                      timeout=7200, mpi_np=1)
(HERE / "si_reference_dft.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res.get("summary", {}), indent=2, default=str))
