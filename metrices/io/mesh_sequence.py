from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import trimesh


_OBJ_INDEX_RE = re.compile(r".*_(\d+)\.obj$", re.IGNORECASE)


def _extract_frame_index(path: Path) -> Optional[int]:
    match = _OBJ_INDEX_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def list_indexed_objs(video_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    paths = sorted(video_dir.glob(f"{prefix}_*.obj"))
    indexed: List[Tuple[int, Path]] = []
    for path in paths:
        index = _extract_frame_index(path)
        if index is None:
            continue
        indexed.append((index, path))
    return sorted(indexed, key=lambda item: item[0])


def load_trimesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected triangular mesh at {mesh_path}")
    mesh.remove_unreferenced_vertices()
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()
    return mesh


@dataclass(frozen=True)
class FrameMeshes:
    frame_index: int
    cloth: trimesh.Trimesh
    body: trimesh.Trimesh


class MeshSequence:
    def __init__(self, video_dir: Path, max_frames: Optional[int] = None):
        self.video_dir = video_dir
        self.max_frames = max_frames

        self._cloth_paths = list_indexed_objs(video_dir, "cloth")
        self._body_paths = list_indexed_objs(video_dir, "body")
        self._index_to_body = {index: path for index, path in self._body_paths}

        if not self._cloth_paths:
            raise FileNotFoundError(f"No cloth frames found in {video_dir} (expected cloth_*.obj)")
        if not self._body_paths:
            raise FileNotFoundError(f"No body frames found in {video_dir} (expected body_*.obj)")

        common_indices = [i for i, _ in self._cloth_paths if i in self._index_to_body]
        if not common_indices:
            raise FileNotFoundError(f"No matching cloth/body indices found in {video_dir}")

        self._indices = common_indices[: max_frames] if max_frames else common_indices

    def __len__(self) -> int:
        return len(self._indices)

    def iter_frames(self) -> Iterable[FrameMeshes]:
        cloth_by_index = {index: path for index, path in self._cloth_paths}
        for frame_index in self._indices:
            cloth_path = cloth_by_index[frame_index]
            body_path = self._index_to_body[frame_index]
            yield FrameMeshes(
                frame_index=frame_index,
                cloth=load_trimesh(cloth_path),
                body=load_trimesh(body_path),
            )

    def load_frame_vertices(self, which: str) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        if which not in {"cloth", "body"}:
            raise ValueError("which must be 'cloth' or 'body'")
        indexed = list_indexed_objs(self.video_dir, which)
        index_to_path = {i: p for i, p in indexed}
        frames: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for frame_index in self._indices:
            mesh = load_trimesh(index_to_path[frame_index])
            frames.append((frame_index, mesh.vertices.copy(), mesh.faces.copy()))
        return frames

