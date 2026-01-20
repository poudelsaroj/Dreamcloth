"""
Blender script: import MPM cloth/body OBJ sequences.

Usage (in Blender):
1) Open Scripting workspace.
2) Load this file and run it.
3) Adjust OUTPUT_DIR and FRAME_RANGE if needed.
"""

import os
import bpy

# ======== Configure ========
OUTPUT_DIR = r"F:\mpm\github\Dreamcloth\output\running_v28"
FRAME_START = 0
FRAME_END = 98  # inclusive; matches saved every 2 frames by default
FRAME_STEP = 2

# File patterns written by the simulation
CLOTH_PATTERN = "cloth_{:04d}.obj"
BODY_PATTERN = "body_{:04d}.obj"


def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _import_obj(path: str, collection: bpy.types.Collection) -> bpy.types.Object:
    # Blender 4.x uses the new OBJ importer.
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    if obj.name not in collection.objects:
        collection.objects.link(obj)
        bpy.context.scene.collection.objects.unlink(obj)
    return obj


def _keyframe_visibility(obj: bpy.types.Object, frame: int, visible: bool):
    obj.hide_viewport = not visible
    obj.hide_render = not visible
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    obj.keyframe_insert(data_path="hide_render", frame=frame)


def import_sequence(output_dir: str, frame_start: int, frame_end: int, frame_step: int):
    cloth_col = _ensure_collection("MPM_Cloth")
    body_col = _ensure_collection("MPM_Body")

    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end

    for frame in range(frame_start, frame_end + 1, frame_step):
        cloth_path = os.path.join(output_dir, CLOTH_PATTERN.format(frame))
        body_path = os.path.join(output_dir, BODY_PATTERN.format(frame))

        if not (os.path.exists(cloth_path) and os.path.exists(body_path)):
            print(f"Missing frame {frame}: {cloth_path} or {body_path}")
            continue

        bpy.context.scene.frame_set(frame)

        cloth_obj = _import_obj(cloth_path, cloth_col)
        body_obj = _import_obj(body_path, body_col)

        cloth_obj.name = f"cloth_{frame:04d}"
        body_obj.name = f"body_{frame:04d}"

        # Make only this frame visible
        _keyframe_visibility(cloth_obj, frame, True)
        _keyframe_visibility(body_obj, frame, True)

        # Hide on neighboring frames to avoid overlap
        _keyframe_visibility(cloth_obj, frame - frame_step, False)
        _keyframe_visibility(body_obj, frame - frame_step, False)
        _keyframe_visibility(cloth_obj, frame + frame_step, False)
        _keyframe_visibility(body_obj, frame + frame_step, False)


if __name__ == "__main__":
    import_sequence(OUTPUT_DIR, FRAME_START, FRAME_END, FRAME_STEP)
