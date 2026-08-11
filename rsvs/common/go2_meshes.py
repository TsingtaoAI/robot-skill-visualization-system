"""从 newtest/assets 加载 Go2 真实连杆网格（禁止静默退回方块，不索引外部路径）。"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from newtest.paths import GO2_MJCF, GO2_SKILL_MJCF


def find_go2_mjcf() -> Path:
    for p in (GO2_MJCF, GO2_SKILL_MJCF):
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "未找到 Go2 MJCF。请确认 newtest/assets/go2/go2.xml 或 assets/go2_skill/go2.xml 存在。"
    )


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def load_go2_body_meshes(xml_path: Optional[Path] = None) -> Dict[str, object]:
    """返回 {body_name: trimesh.Trimesh}，网格失败则抛错（不用方块占位）。"""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("需要 trimesh 才能渲染真实 Go2 网格：pip install trimesh") from exc

    xml_path = Path(xml_path) if xml_path is not None else find_go2_mjcf()
    if not xml_path.is_file():
        raise FileNotFoundError(f"未找到 Go2 MJCF: {xml_path}")
    xml_path = xml_path.resolve()
    root = ET.parse(xml_path).getroot()
    xml_dir = os.path.dirname(str(xml_path))
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "meshes") if compiler is not None else "meshes"

    mesh_files: Dict[str, str] = {}
    materials: Dict[str, np.ndarray] = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            fname = mesh.get("file")
            if not fname:
                continue
            name = mesh.get("name") or Path(fname).stem
            mesh_files[name] = os.path.join(xml_dir, meshdir, fname)
        for mat in asset.findall("material"):
            name = mat.get("name")
            rgba = mat.get("rgba")
            if name and rgba:
                materials[name] = np.array([float(x) for x in rgba.split()], dtype=np.float64)

    def parse_xyz(s, default=(0.0, 0.0, 0.0)):
        if not s:
            return np.array(default, dtype=np.float64)
        return np.array([float(x) for x in s.split()], dtype=np.float64)

    body_meshes: Dict[str, object] = {}
    cache: Dict[str, object] = {}

    def walk(body_elem):
        body_name = body_elem.get("name")
        parts = []
        for geom in body_elem.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in mesh_files:
                continue
            class_name = geom.get("class", "")
            if class_name and "collision" in class_name:
                continue
            full = mesh_files[mesh_name]
            if not os.path.isfile(full):
                continue
            if full in cache:
                mesh = cache[full].copy()
            else:
                loaded = trimesh.load(full, force="mesh")
                if isinstance(loaded, trimesh.Scene):
                    geoms = [
                        g
                        for g in loaded.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                    if not geoms:
                        continue
                    loaded = (
                        trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
                    )
                if not isinstance(loaded, trimesh.Trimesh):
                    continue
                mesh = loaded
                cache[full] = mesh.copy()
            T = np.eye(4)
            T[:3, 3] = parse_xyz(geom.get("pos"))
            euler = geom.get("euler")
            if euler:
                T[:3, :3] = _rpy_to_matrix(parse_xyz(euler))
            mesh.apply_transform(T)
            mat_name = geom.get("material")
            rgba = materials.get(mat_name, np.array([0.75, 0.78, 0.82, 1.0]))
            color = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            if color.shape[0] == 3:
                color = np.append(color, 255)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(color, (len(mesh.vertices), 1))
            )
            parts.append(mesh)
        if body_name and parts:
            body_meshes[body_name] = trimesh.util.concatenate(parts)
        for child in body_elem.findall("body"):
            walk(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"MJCF 无 worldbody: {xml_path}")
    for body in worldbody.findall("body"):
        walk(body)

    if not body_meshes:
        raise RuntimeError(
            f"未能从 {xml_path} 加载任何视觉网格"
            f"（检查 meshdir={meshdir!r} 下的 .obj/.stl，且 mesh 名称可解析）"
        )

    print(f"[mesh] 已加载 Go2 真实网格 {len(body_meshes)} 个连杆 ← {xml_path}", flush=True)
    return body_meshes
