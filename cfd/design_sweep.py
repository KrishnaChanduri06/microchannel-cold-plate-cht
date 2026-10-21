"""Design study (channel width x fin width x flow, copper and silicon) on a 1 cm wide cold plate.
Meshes with mesh.py and solves with case.py; skips points whose results/*.json already exist.

    python design_sweep.py
"""
import itertools
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
Q = 100                                    # W/cm^2
FLOWS = (200, 500, 1000)                   # ml/min for the whole 1 x 4.48 cm plate
H_BASE = 1000                              # um, base under the channels
H_LID = 1000                               # um, lid of the same solid as the plate
T_IN = 30.0                                # C, data-centre warm-water inlet
DESIGNS = {                                # name: (W_ch, full fin width) in um
    "copper": list(itertools.product((100, 150, 231, 350), (100, 236))),
    "silicon": [(w, 100) for w in (100, 150, 231, 350)],
}


def main() -> None:
    (HERE / "runs" / "design").mkdir(parents=True, exist_ok=True)
    for solid, designs in DESIGNS.items():
        for w_ch, w_fin in designs:
            name = f"d_wch{w_ch}_wfin{w_fin}"
            msh = HERE / "runs" / "design" / f"{name}.msh"
            todo = [f for f in FLOWS
                    if not (HERE / "results" / f"{name}_{solid}_q{Q}_flow{f}.json").exists()]
            if not todo:
                continue
            if not msh.exists():
                subprocess.run([PY, "mesh.py", "1.0", str(msh), "--w-ch", str(w_ch), "--w-fin", str(w_fin),
                                "--h-base", str(H_BASE), "--h-lid", str(H_LID), "--no-slots"], cwd=HERE, check=True,
                               stdout=subprocess.DEVNULL)
            print(f"== {solid} {name}: flows {todo}", flush=True)
            subprocess.run([PY, "case.py", "--mesh", str(msh), "--q", str(Q), "--flow", *map(str, todo),
                            "--solid", solid, "--lid", "solid", "--t-in", str(T_IN)], cwd=HERE, check=True)


if __name__ == "__main__":
    main()
