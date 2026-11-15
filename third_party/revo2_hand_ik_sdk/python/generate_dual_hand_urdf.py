#!/usr/bin/env python3
"""Create an RViz-only dual-hand URDF from the supplied left/right Revo2 URDFs.

The source hand URDFs remain unchanged.  This tool joins their uniquely named
links/joints under a world link and resolves all mesh paths to absolute file://
URIs, so the generated robot_description can be published by a ROS topic.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET


def resolve_mesh_paths(element: ET.Element, source_dir: Path) -> None:
    for mesh in element.findall(".//mesh"):
        filename = mesh.get("filename", "").strip()
        if not filename or filename.startswith(("file://", "package://", "http://", "https://")):
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (source_dir / mesh_path).resolve()
        mesh.set("filename", f"file://{mesh_path}")


def append_hand(root: ET.Element, urdf_path: Path) -> None:
    tree = ET.parse(urdf_path)
    source_root = tree.getroot()
    if source_root.tag != "robot":
        raise RuntimeError(f"not a URDF robot document: {urdf_path}")

    for child in source_root:
        # The Revo2 hand files contain links and joints only.  Keep any material
        # definitions as well, while avoiding an XML declaration nested in root.
        if child.tag not in {"link", "joint", "material", "gazebo", "transmission"}:
            continue
        cloned = copy.deepcopy(child)
        resolve_mesh_paths(cloned, urdf_path.parent)
        root.append(cloned)


def fixed_joint(name: str, parent: str, child: str, xyz: tuple[float, float, float]) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {
        "xyz": f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}",
        "rpy": "0 0 0",
    })
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return joint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a combined Revo2 dual-hand RViz URDF.")
    parser.add_argument("--left", required=True, type=Path, help="left-hand source URDF")
    parser.add_argument("--right", required=True, type=Path, help="right-hand source URDF")
    parser.add_argument("--output", required=True, type=Path, help="generated combined URDF")
    parser.add_argument("--left-y", type=float, default=0.14, help="world Y offset for left hand, meters")
    parser.add_argument("--right-y", type=float, default=-0.14, help="world Y offset for right hand, meters")
    args = parser.parse_args()

    for path in (args.left, args.right):
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")

    root = ET.Element("robot", {"name": "revo2_dual_hand_rviz"})
    root.append(ET.Element("link", {"name": "world"}))
    append_hand(root, args.left.resolve())
    append_hand(root, args.right.resolve())
    root.append(fixed_joint("world_to_left_base", "world", "left_base_link", (0.0, args.left_y, 0.0)))
    root.append(fixed_joint("world_to_right_base", "world", "right_base_link", (0.0, args.right_y, 0.0)))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"[dual-urdf][OK] {args.output}")


if __name__ == "__main__":
    main()
