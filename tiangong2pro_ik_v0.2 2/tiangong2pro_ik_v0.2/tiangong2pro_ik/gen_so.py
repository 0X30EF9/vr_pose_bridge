from __future__ import annotations

import os
import sys
from pathlib import Path

from tiangong_ik import TiangongArmIK

PACKAGE_DIR = Path(__file__).resolve().parent
URDF_PATH = PACKAGE_DIR / "tienkung2_pro" / "urdf" / "tiangong2.0_pro.urdf"
MESH_PATH = URDF_PATH.parent
SO_DIR = PACKAGE_DIR / ".casadi_cache"
SO_PATH = SO_DIR / "ik_solver.so"


def main() -> None:
    SO_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(PACKAGE_DIR)
    arm_ik = TiangongArmIK(
        urdf_path=str(URDF_PATH),
        mesh_path=str(MESH_PATH),
        jit=False,
    )
    arm_ik.export_so(output_so=str(SO_PATH), condaPath=sys.prefix)
    if not SO_PATH.is_file():
        raise RuntimeError(f"IK 动态库生成失败: {SO_PATH}")
    print(f"IK_SOLVER_SO={SO_PATH}")


if __name__ == "__main__":
    main()
