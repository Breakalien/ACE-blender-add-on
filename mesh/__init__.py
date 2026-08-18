"""
Blender addon: import Assetto Corsa EVO ".mesh" files, edit the geometry,
and save back to .mesh.

Installation:
    Blender > Edit > Preferences > Add-ons > Install from Disk...
    -> select this whole folder zipped, OR copy the folder directly into
       Blender's "scripts/addons" directory and enable it from the list.

Usage:
    File > Import > Assetto Corsa Mesh (.mesh)
        "Preset" (Car/Track) sets a quick starting combination of every
        option below - each stays individually editable afterwards.
        Pick the file(s) - multi-select works, one LOD to edit (LOD 0 =
        highest detail) or tick "Import all LODs" to bring in every LOD as
        its own object.
        "Split" chooses how to break the LOD into separate objects instead
        of one combined multi-material mesh: "By material" imports each
        submesh as its own child object (safest for isolated UV edits); "By
        rigid part" imports each movable part (door, mirror, wing, ...) as
        its own multi-material object, parented directly to its bone Empty
        when "Import hierarchy" is also on. Both are 1:1 with what the
        .mesh file already stores (material ranges / per-vertex rigid-part
        index respectively) - nothing gets merged that wasn't already
        merged in the file, and export re-merges everything back into the
        single buffer the format expects either way.
        "Import hierarchy" (on by default) also brings in the rigid-part
        bone tree (doors, mirrors, wipers, ...) as correctly positioned and
        parented Empty objects, purely for reference - the mesh isn't
        parented to them, and the hierarchy is never edited or re-derived
        from them on export.
        "Fix orientation" (on by default) rotates the whole LOD 90 degrees
        so it lies flat in Blender's Z-up world instead of on its side (the
        .mesh format itself is Y-up, like the AC engine) - via a wrapping
        Empty, not by touching any vertex/normal/bone data, so export needs
        no "undo" step and isn't affected either way.
        "Link textures" (on by default) resolves and wires up the real
        diffuse/normal/opacity textures onto each material's shader nodes,
        instead of leaving plain named slots with no texture. The car's
        content folder is found automatically from the .mesh file's own
        location (two levels up, e.g. .../meshes/foo.mesh -> .../) - no
        manual input needed; only an optional separate "Assets" folder for
        content shared across cars (common_assets, editor textures) still
        needs to be set by hand. Purely visual/reference - the .mesh format
        itself has no notion of texture bitmaps at all, so none of this is
        written back on export.
        "Fix glass transparency" (on by default) forces a near-zero alpha on
        the known glass materials (EXT_WINDOWS, INT_WINDOWS, INT_WINDSHIELD,
        EXT_LIGHTS_GLASS_FRONT) so they render see-through.
        "Cam and lights" (on by default) adds a ready-to-render
        "Camera_Lighting_Rig" (3/4 exterior hero camera, interior cabin
        camera if available, 3-point area lighting), sized to whatever AC
        content is currently in the scene.
    ... edit the mesh(es) (move/add/remove vertices, edit UVs, etc.) ...
    File > Export > Assetto Corsa Mesh (.mesh)
        Only works on an object that was imported by this addon (it needs
        to know which original file to write back to). Export always looks
        at *every* object in the scene tagged with that same source file
        (however many there are - one combined mesh, or several split-by-
        material ones per LOD) and re-encodes each from its current Blender
        state; any LOD with no matching object in the scene is copied
        byte-for-byte from the original file instead. Import all the LODs
        you plan to touch before exporting - keeping them mutually
        consistent (all re-encoded the same way) is safer than mixing a
        freshly re-encoded LOD with others that never went through Blender
        at all.

Scope / limitations (see mesh_codec.py's module docstring for the full
reverse-engineered format notes):
    - Positions, per-polygon material assignment, UV0 and the per-vertex
      rigid bone-index attribute are fully editable and round-trip exactly.
    - Normals and tangents are NOT preserved from the source file - Blender
      recomputes them from the (smooth-shaded) geometry, and export bakes
      whatever Blender currently computes. If you don't touch the mesh at
      all, a round-trip is still byte-identical for every field EXCEPT
      normals/tangents, which will be numerically close but not identical.
    - UV1/UV2/UV3 (extra channels, always empty on every sample file seen)
      are written back as zero-filled rather than preserved.
    - The bone hierarchy (rigid parts used to animate doors/mirrors/wipers)
      itself (names/parenting) is read-only, but each part's position/
      rotation is editable by moving its imported Empty - re-encoded and
      written back on export; parts whose Empty wasn't touched are still
      copied byte-for-byte from the original file.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import tempfile

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty,
)
from bpy.types import Operator, OperatorFileListElement, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper, ExportHelper

from . import ace_texture
from . import material_codec
from . import mesh_codec
from . import normal_map
from . import project_resolver

bl_info = {
    "name": "Assetto Corsa Mesh (.mesh) Import/Export",
    "author": "Ultimate ACE Convertor project",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "File > Import/Export > Assetto Corsa Mesh (.mesh)",
    "description": "Import, edit and re-save Assetto Corsa EVO .mesh files",
    "category": "Import-Export",
}

_BONE_ATTR = "ac_bone_index"
_EXTRA_ATTR = "ac_extra"
_PROP_SOURCE_PATH = "ac_mesh_source_path"
_PROP_LOD_INDEX = "ac_mesh_lod_index"
_PROP_BONE_NAME = "ac_bone_name"
# Set on an Image datablock that came from an existing .texture, holding its
# engine-relative path so it can be written back under the same name.
_PROP_TEXTURE_PATH = "ac_texture_path"
_BONE_MATRIX_EPS = 1e-5

_lod_items_cache: dict[str, list] = {}

# .mesh positions/normals/bone transforms are Y-up (X=width, Y=height,
# Z=length - matches the AC engine convention, confirmed by cross-checking
# real car dimensions and cabin layout). Blender's world is Z-up, so an
# unrotated import ends up lying on its side. Applied only to a wrapping
# Empty (see IMPORT_OT_ac_mesh.execute) - never baked into vertex/bone data -
# so export reads local mesh coordinates completely unaffected by it.
_AXIS_FIX_ROTATION = Matrix.Rotation(math.radians(90), 4, "X")

# Materials that model glass in the source content but have no dedicated
# transparency/refraction shader on our side (plain Principled BSDF) - forcing
# a near-zero alpha instead makes them see-through rather than solid grey/
# white panes, which otherwise makes windows/windshields/canopy glass look
# like body panels.
_GLASS_ALPHA_MATERIALS = {"EXT_WINDOWS", "INT_WINDOWS", "INT_WINDSHIELD", "EXT_LIGHTS_GLASS_FRONT"}
_GLASS_ALPHA_VALUE = 0.02

_RIG_COLLECTION_NAME = "Camera_Lighting_Rig"

# Blender's DDS loader on older versions (confirmed: 3.6 LTS) has two gaps,
# neither of which affects the actual compressed bytes, only how they're
# tagged - so both are fixed by rewriting the DDS header for our own preview
# purposes (verified pixel-value-identical against the untouched original,
# loaded on 4.2/5.0 where both gaps are already handled correctly):
#
#   1. DX10 "_SRGB" DXGI format codes aren't recognised at all (silently
#      fails to load -> 0x0 broken image) - downgrade to the UNORM
#      counterpart and let colorspace_settings apply the curve instead.
#   2. BC1/BC2/BC3 specifically aren't recognised via the modern DX10
#      extended header either, only via their legacy pre-DX10 FourCC
#      ("DXT1"/"DXT3"/"DXT5") - BC4/5/6/7 have no legacy FourCC (DX10-only
#      formats) and load fine via DX10 there, so this only applies to BC1-3.
_SRGB_TO_UNORM_DXGI = {29: 28, 91: 87, 72: 71, 75: 74, 78: 77, 99: 98}
_LEGACY_FOURCC_BY_DXGI = {71: b"DXT1", 72: b"DXT1", 74: b"DXT3", 75: b"DXT3", 77: b"DXT5", 78: b"DXT5"}


# Fixed preset values applied by IMPORT_OT_ac_mesh.preset's update callback -
# a quick starting point for the two content types this addon is used for,
# not a locked mode: every toggle stays individually editable afterwards.
_IMPORT_PRESETS = {
    "CAR": dict(
        import_all_lods=False, split_mode="NONE", import_hierarchy=True,
        link_textures=True, fix_glass_alpha=True, add_camera_lights=True, fix_orientation=True,
    ),
    "TRACK": dict(
        import_all_lods=False, split_mode="NONE", import_hierarchy=True,
        link_textures=True, fix_glass_alpha=False, add_camera_lights=False, fix_orientation=True,
    ),
    # Geared to the light-authoring workflow: one object per material so light
    # parts can be isolated and painted, with everything else stripped out.
    "LIGHT_PAINT": dict(
        import_all_lods=False, split_mode="MATERIAL", import_hierarchy=False,
        link_textures=True, fix_glass_alpha=False, add_camera_lights=False, fix_orientation=True,
    ),
}


def _apply_import_preset(self, context):
    for prop_name, value in _IMPORT_PRESETS[self.preset].items():
        setattr(self, prop_name, value)


def _get_lod_items(self, context):
    path = self.filepath
    if not path or not os.path.isfile(path):
        return [("0", "LOD 0", "")]
    if path not in _lod_items_cache:
        try:
            mf = mesh_codec.load_mesh_file(path)
            items = [
                (str(lod.index), f"LOD {lod.index} (dist={lod.distance:g}, {lod.vertex_count} verts)", "")
                for lod in mf.lods
            ]
        except Exception as exc:  # noqa: BLE001
            items = [("0", f"LOD 0 (erreur de lecture: {exc})", "")]
        _lod_items_cache.clear()  # keep memory bounded - we only ever need the current file
        _lod_items_cache[path] = items or [("0", "LOD 0", "")]
    return _lod_items_cache[path]


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

class IMPORT_OT_ac_mesh(Operator, ImportHelper):
    bl_idname = "import_scene.ac_mesh"
    bl_label = "Import AC Mesh (.mesh)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".mesh"
    filter_glob: StringProperty(default="*.mesh", options={"HIDDEN"})
    # Multi-select support: when several files are picked in the browser,
    # Blender fills `files` (one entry per picked filename) + `directory`
    # (their common folder) instead of just `filepath` (which the browser
    # only sets to a single - typically the last-clicked - file).
    files: CollectionProperty(type=OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    preset: EnumProperty(
        name="Preset",
        description=(
            "Applies a quick starting combination of the options below - each one stays "
            "individually editable afterwards, this only sets their initial values."
        ),
        items=[
            ("CAR", "Car", "Hierarchy, textures, glass fix and cam/lights on; one combined object per LOD"),
            ("TRACK", "Track", "Hierarchy and textures on, no glass fix or cam/lights; one combined object per LOD"),
            (
                "LIGHT_PAINT", "Light paint",
                "Split by material with textures on, no hierarchy/glass fix/cam-lights - for isolating "
                "and painting the light meshes",
            ),
        ],
        default="CAR",
        update=_apply_import_preset,
    )

    lod_index: EnumProperty(
        name="LOD", description="Which LOD to import for editing (ignored if 'Import all LODs' is on)",
        items=_get_lod_items,
    )
    import_all_lods: BoolProperty(
        name="Import all LODs",
        description=(
            "Import every LOD as its own object instead of just one. Recommended if you plan to "
            "export afterwards: exporting always writes back every LOD currently present in the "
            "scene for this file, re-encoded the same way - importing (and, if needed, re-touching) "
            "all of them keeps them mutually consistent instead of mixing a freshly re-encoded LOD "
            "with others that were never round-tripped through Blender at all."
        ),
        default=False,
    )
    split_mode: EnumProperty(
        name="Split",
        description=(
            "How to break each LOD into separate Blender objects instead of one combined "
            "multi-material mesh. The .mesh format itself only stores one flat vertex/index "
            "buffer per LOD with two independent, orthogonal partition tables - a material-range "
            "table and a per-vertex rigid-part (bone) index - both options below expose one of "
            "those 1:1, nothing is merged that wasn't already merged in the file. Export re-merges "
            "everything back into one buffer automatically either way."
        ),
        items=[
            ("NONE", "Combined", "One object per LOD, all materials on one mesh"),
            ("MATERIAL", "By material", "One object per material range - safest for isolated UV edits"),
            (
                "RIGID_PART", "By rigid part",
                "One object per rigid part (door, mirror, wing, ...), multi-material, parented "
                "directly to its bone Empty when 'Import hierarchy' is also on",
            ),
        ],
        default="NONE",
    )
    import_hierarchy: BoolProperty(
        name="Import hierarchy (bones -> Empties)",
        description=(
            "Also import the rigid-part hierarchy (doors, mirrors, wipers, ...) as a tree of "
            "Empty objects, correctly positioned and parented. The mesh geometry is not parented "
            "to them, and their names/parenting are fixed, but moving/rotating one of these "
            "Empties and exporting DOES write its new transform back to the .mesh file - parts "
            "left untouched are still re-exported byte-for-byte from the original file."
        ),
        default=True,
    )
    link_textures: BoolProperty(
        name="Link textures",
        description=(
            "Resolve and link the real diffuse/normal/opacity textures from the car's content "
            "folder onto each material's shader nodes, instead of leaving plain named slots. "
            "The content folder is found automatically (two levels up from the .mesh file, e.g. "
            ".../meshes/foo.mesh -> .../) - no manual input needed."
        ),
        default=True,
    )
    fix_glass_alpha: BoolProperty(
        name="Fix glass transparency",
        description=(
            "Forces alpha to a near-zero value on the known glass materials (EXT_WINDOWS, "
            "INT_WINDOWS, INT_WINDSHIELD, EXT_LIGHTS_GLASS_FRONT) so windows/windshields render "
            "see-through instead of solid, since our shader setup has no dedicated glass model."
        ),
        default=True,
    )
    add_camera_lights: BoolProperty(
        name="Cam and lights",
        description=(
            "Adds (or repositions, if already present) a ready-to-render 'Camera_Lighting_Rig' - "
            "a 3/4 exterior hero camera, an interior cabin camera (only if a STEER_BASE bone is "
            "present, i.e. the interior LOD was imported), and 3-point area lighting (key/fill/"
            "rim) - sized and aimed from the bounding box of every AC mesh currently in the scene."
        ),
        default=True,
    )
    assets_root: StringProperty(
        name="Assets",
        description=(
            "Optional folder with the same layout (materials/, texture(s)/, editor/, ...) holding "
            "generic materials/textures shared across cars - used whenever a material points to "
            "\"content\\cars\\common_assets\\...\" or \"editor\\textures\\...\" instead of "
            "something inside 'AC content folder'. Leave empty to just skip those (as before)."
        ),
        subtype="DIR_PATH",
    )
    fix_orientation: BoolProperty(
        name="Fix orientation (Y-up -> Z-up)",
        description=(
            "The .mesh format is Y-up (like the AC engine); Blender is Z-up. Without this, the "
            "car imports lying on its side. Rotates a wrapping Empty only - never touches vertex, "
            "normal or bone data - so it has no effect on export whatsoever."
        ),
        default=True,
    )
    mesh_cleaning: BoolProperty(
        name="Mesh cleaning",
        description=(
            "Off by default: every mesh is imported exactly as stored, nothing merged or removed. "
            "Turn this on to run the same coincident-duplicate-triangle removal as 'Clean selected "
            "mesh' on every object right after import (triangles whose 3 corners sit at the same "
            "position as another triangle's, independent of winding or which vertices they use)."
        ),
        default=False,
    )

    def execute(self, context):
        if self.files and len(self.files) > 1:
            paths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        else:
            paths = [self.filepath]

        for o in context.selected_objects:
            o.select_set(False)

        active_obj = None
        summaries = []
        all_warnings = []
        for path in paths:
            result = self._import_one(context, path)
            if result is None:
                continue
            summary, warnings, obj = result
            summaries.append(f"{os.path.basename(path)}: {summary}")
            all_warnings.extend(warnings)
            if obj is not None:
                active_obj = obj

        if self.add_camera_lights:
            _ensure_camera_lighting_rig(context)

        context.view_layer.objects.active = active_obj

        if not summaries:
            return {"CANCELLED"}

        self.report({"INFO"}, "Imported - " + " | ".join(summaries))
        for w in all_warnings:
            self.report({"WARNING"}, w)
        return {"FINISHED"}

    def _import_one(self, context, filepath: str):
        """Imports one .mesh file (per the operator's current settings) into
        the scene. Returns (summary_str, warnings_list, active_obj), or None
        if this file failed/was skipped (already reported via self.report -
        callers should just move on to the next file in a multi-select)."""
        try:
            mf = mesh_codec.load_mesh_file(filepath)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to read {os.path.basename(filepath)}: {exc}")
            return None

        material_resolver = None
        texture_temp_dir = None
        if self.link_textures:
            # Content folder layout is always <root>/meshes/foo.mesh (materials/,
            # texture(s)/ live as siblings of meshes/) - derive it automatically,
            # no manual input needed.
            project_root = os.path.dirname(os.path.dirname(filepath))
            if self.assets_root and not os.path.isdir(self.assets_root):
                self.report({"ERROR"}, "'Assets' is set but is not a valid folder.")
                return None
            texture_temp_dir = tempfile.mkdtemp(prefix="uacec2_bl_tex_")
            index = project_resolver.ProjectIndex(project_root, assets_root=self.assets_root or None)
            texture_converter = project_resolver.TextureConverter(index, texture_temp_dir)
            material_resolver = project_resolver.MaterialResolver(index, texture_converter)

        base_name = os.path.splitext(os.path.basename(filepath))[0]

        if self.import_all_lods:
            lods_to_import = list(enumerate(mf.lods))
        else:
            lod_index = int(self.lod_index)
            if lod_index >= len(mf.lods):
                self.report(
                    {"WARNING"},
                    f"{base_name}: LOD {lod_index} not found ({len(mf.lods)} LOD(s)) - file skipped.",
                )
                return None
            lods_to_import = [(lod_index, mf.lods[lod_index])]

        image_cache: dict[str, bpy.types.Image] = {}
        active_obj = None
        cleaned_faces = 0
        try:
            for lod_index, lod in lods_to_import:
                lod_name = base_name if len(lods_to_import) == 1 else f"{base_name}_LOD{lod_index}"

                axis_root = None
                if self.fix_orientation:
                    axis_root = bpy.data.objects.new(lod_name, None)
                    axis_root.empty_display_type = "ARROWS"
                    axis_root.rotation_euler = _AXIS_FIX_ROTATION.to_euler()
                    axis_root[_PROP_SOURCE_PATH] = filepath
                    axis_root[_PROP_LOD_INDEX] = lod_index
                    context.collection.objects.link(axis_root)
                    geo_name = f"{lod_name}_geo"
                else:
                    geo_name = lod_name

                bone_empties_by_name: dict[str, bpy.types.Object] = {}
                if self.import_hierarchy and lod.bones:
                    hierarchy_root, bone_empties_by_name = _build_bone_hierarchy(
                        context, lod, lod_name, filepath, lod_index,
                    )
                    hierarchy_root.parent = axis_root
                    active_obj = active_obj or hierarchy_root
                    # bone Empties were just (re)parented - matrix_world isn't
                    # guaranteed flushed yet within this same operator call,
                    # and RIGID_PART parenting below needs it to be correct.
                    context.view_layer.update()

                if self.split_mode == "NONE":
                    obj = _build_object_from_lod(
                        geo_name, lod, material_resolver, image_cache, texture_temp_dir, self.fix_glass_alpha,
                    )
                    obj[_PROP_SOURCE_PATH] = filepath
                    obj[_PROP_LOD_INDEX] = lod_index
                    obj.parent = axis_root
                    context.collection.objects.link(obj)
                    obj.select_set(True)
                    active_obj = obj
                    if self.mesh_cleaning:
                        cleaned_faces += _clean_duplicate_faces(obj)
                elif self.split_mode == "MATERIAL":
                    used_ranges = [mr for mr in lod.materials if mr.count > 0]
                    group = bpy.data.objects.new(geo_name, None)
                    group[_PROP_SOURCE_PATH] = filepath
                    group[_PROP_LOD_INDEX] = lod_index
                    group.parent = axis_root
                    context.collection.objects.link(group)

                    for mat_range in used_ranges:
                        sub_lod = _extract_submesh_lod(lod, mat_range)
                        sub_name = f"{lod_name}_{mat_range.name}"
                        obj = _build_object_from_lod(
                            sub_name, sub_lod, material_resolver, image_cache, texture_temp_dir, self.fix_glass_alpha,
                        )
                        obj[_PROP_SOURCE_PATH] = filepath
                        obj[_PROP_LOD_INDEX] = lod_index
                        obj.parent = group
                        context.collection.objects.link(obj)
                        obj.select_set(True)
                        active_obj = obj
                        if self.mesh_cleaning:
                            cleaned_faces += _clean_duplicate_faces(obj)
                else:  # RIGID_PART
                    group = bpy.data.objects.new(geo_name, None)
                    group[_PROP_SOURCE_PATH] = filepath
                    group[_PROP_LOD_INDEX] = lod_index
                    group.parent = axis_root
                    context.collection.objects.link(group)

                    # What matrix_world a freshly-parented, transform-less object
                    # should end up with - same as every other mode: axis_root's
                    # rotation if "Fix orientation" is on, otherwise untransformed.
                    target_world = axis_root.matrix_world.copy() if axis_root is not None else Matrix.Identity(4)

                    for bone_name, sub_lod in _split_lod_by_rigid_part(lod):
                        sub_name = f"{lod_name}_{bone_name}"
                        obj = _build_object_from_lod(
                            sub_name, sub_lod, material_resolver, image_cache, texture_temp_dir, self.fix_glass_alpha,
                        )
                        obj[_PROP_SOURCE_PATH] = filepath
                        obj[_PROP_LOD_INDEX] = lod_index
                        context.collection.objects.link(obj)
                        bone_empty = bone_empties_by_name.get(bone_name)
                        if bone_empty is not None:
                            # Parent to the matching bone Empty *without* moving the
                            # geometry - it's already baked to its correct resting-pose
                            # position, so the parent-inverse must cancel the bone's
                            # own world transform out and reinstate target_world (same
                            # as Ctrl+P "Keep Transform", generalised to a non-identity
                            # target instead of assuming the object started unparented).
                            obj.parent = bone_empty
                            obj.matrix_parent_inverse = bone_empty.matrix_world.inverted() @ target_world
                        else:
                            obj.parent = group
                        obj.select_set(True)
                        active_obj = obj
                        if self.mesh_cleaning:
                            cleaned_faces += _clean_duplicate_faces(obj)

                if axis_root is not None:
                    axis_root.select_set(True)
                    active_obj = axis_root

            if material_resolver is not None:
                # Must happen before the temp dir goes away: the converted
                # textures live there until the images are loaded and packed.
                _collect_base_color_textures(
                    context, lods_to_import, material_resolver, image_cache, texture_temp_dir)
        finally:
            if texture_temp_dir:
                shutil.rmtree(texture_temp_dir, ignore_errors=True)

        summary = ", ".join(f"LOD{i} ({lod.vertex_count}v/{lod.triangle_count}t)" for i, lod in lods_to_import)
        warnings = []
        if cleaned_faces:
            warnings.append(
                f"{base_name}: Mesh cleaning removed {cleaned_faces} coincident duplicate triangle(s) "
                "(same 3 corner positions as another triangle) - no visual difference, but the export "
                "will produce that many fewer triangles for those LOD(s).",
            )
        if material_resolver is not None and (material_resolver.warnings or material_resolver.textures.warnings):
            warnings.extend(material_resolver.warnings + material_resolver.textures.warnings)

        return summary, warnings, active_obj


# Populated at import: material name -> base colour Image. A car can carry
# several light materials, so the panel follows the active object rather than
# pinning one global reference (see _sync_light_slots_to_active).
_BASECOLOR_BY_MATERIAL: dict = {}
# Still used at export time (MESH_OT_ac_apply_ref_material) to decide which
# FunctionMask slot a painted image belongs to - unrelated to auto-import.
_MASK_SLOT_TO_PROP = {"FunctionMask1": "ac_lights_f1_image", "FunctionMask2": "ac_lights_f2_image"}


def _collect_base_color_textures(context, lods_to_import, resolver, image_cache, temp_dir) -> None:
    """Records each material's base colour texture, then points the optic
    reference panel at the active object's one. F1/F2 are no longer
    auto-loaded from existing materials at import - the "Prepare" workflow
    always builds fresh light materials, so there is nothing meaningful to
    detect there anymore."""
    for _lod_index, lod in lods_to_import:
        for mat_range in lod.materials:
            if mat_range.count <= 0 or mat_range.name in _BASECOLOR_BY_MATERIAL:
                continue
            resolved = resolver.resolve(mat_range.name, mat_range.path)
            if resolved.diffuse_texture:
                try:
                    _BASECOLOR_BY_MATERIAL[mat_range.name] = _load_image_cached(
                        resolved.diffuse_texture, "sRGB", image_cache, temp_dir)
                except Exception:  # noqa: BLE001
                    pass
    _sync_light_slots_to_active(context)


def _sync_light_slots_to_active(context) -> bool:
    """Points the optic reference at whatever the active object's material
    uses. Returns True if anything changed."""
    obj = getattr(context, "active_object", None)
    if obj is None or obj.type != "MESH" or not obj.data.materials:
        return False
    mat = obj.active_material or obj.data.materials[0]
    if mat is None:
        return False
    scene = context.scene

    base = _BASECOLOR_BY_MATERIAL.get(mat.name) or _find_base_color_image(mat)
    if base is not None and base.size[0]:
        wanted = f"{_clean_texture_basename(base.name)}_BW"
        current = scene.ac_lights_ref_image
        if current is None or current.name != wanted:
            try:
                scene.ac_lights_ref_image = _make_bw_copy(base)
                return True
            except Exception:  # noqa: BLE001 - never let a UI refresh raise
                pass
    return False


@bpy.app.handlers.persistent
def _ac_selection_changed(scene, depsgraph=None):
    """Keeps the optic reference in sync with the active object's material."""
    global _LAST_ACTIVE_OBJECT
    try:
        context = bpy.context
        obj = getattr(context, "active_object", None)
        name = obj.name if obj is not None else None
        if name == _LAST_ACTIVE_OBJECT:
            return
        _LAST_ACTIVE_OBJECT = name
        if not _BASECOLOR_BY_MATERIAL:
            return
        _sync_light_slots_to_active(context)
    except Exception:  # noqa: BLE001 - a handler must never raise
        pass


_LAST_ACTIVE_OBJECT = None


def _extract_submesh_lod(lod: mesh_codec.Lod, mat_range: mesh_codec.MaterialRange) -> mesh_codec.Lod:
    """Returns a standalone, vertex-compacted (0-based) Lod containing only
    the geometry of one material range - used to import each submesh as its
    own object when "Split by material" is on."""
    tri_indices = lod.indices[mat_range.start:mat_range.start + mat_range.count]
    remap: dict[int, int] = {}
    positions, normals, uv0, bone_weights, bone_indices, extra = [], [], [], [], [], []
    local_indices = []
    for vi in tri_indices:
        li = remap.get(vi)
        if li is None:
            li = len(positions)
            remap[vi] = li
            positions.append(lod.positions[vi])
            normals.append(lod.normals[vi])
            uv0.append(lod.uv0[vi])
            bone_weights.append(lod.bone_weights[vi] if vi < len(lod.bone_weights) else (1.0, 0.0, 0.0, 0.0))
            bone_indices.append(lod.bone_indices[vi] if vi < len(lod.bone_indices) else (0, 0, 0, 0))
            extra.append(lod.extra[vi] if vi < len(lod.extra) else (0.0, 0.0, 0.0, 0.0))
        local_indices.append(li)

    n = len(positions)
    single_mat = mesh_codec.MaterialRange(name=mat_range.name, start=0, count=len(local_indices), path=mat_range.path)
    return mesh_codec.Lod(
        index=lod.index, distance=lod.distance, materials=[single_mat],
        positions=positions, normals=normals, uv0=uv0,
        tangents=[(1.0, 0.0, 0.0, 1.0)] * n, bone_weights=bone_weights, bone_indices=bone_indices,
        indices=local_indices, bones=lod.bones, extra=extra,
        uv1=[(0.0, 0.0)] * n, uv2=[(0.0, 0.0)] * n, uv3=[(0.0, 0.0)] * n,
    )


def _material_by_triangle(lod: mesh_codec.Lod) -> list:
    """Returns a (name, path) pair per triangle, derived from lod.materials -
    which partitions the whole index buffer exhaustively and without overlap
    (verified against every sample file seen), so a single sweep over the
    sorted ranges covers every triangle exactly once."""
    n_tris = len(lod.indices) // 3
    out = [(None, None)] * n_tris
    for mr in sorted((m for m in lod.materials if m.count > 0), key=lambda m: m.start):
        t0, t1 = mr.start // 3, (mr.start + mr.count) // 3
        for t in range(t0, t1):
            out[t] = (mr.name, mr.path)
    return out


def _split_lod_by_rigid_part(lod: mesh_codec.Lod) -> list:
    """Groups the LOD's triangles by rigid part (per-vertex bone_index,
    majority vote across a triangle's 3 corners - real seam triangles
    spanning 2 parts are near-zero in practice, well under 0.1% on every
    sample file checked) instead of by material. A single part commonly
    spans several materials (e.g. a door has paint, glass and trim), so
    each returned sub_lod keeps its own local multi-entry MaterialRange
    table, same as the ungrouped LOD. Returns a list of (bone_name, Lod),
    one per rigid part with at least one triangle."""
    if not lod.bone_indices or not lod.bones:
        return []

    bone_by_vertex = [bi[0] if bi else 0 for bi in lod.bone_indices]
    material_by_tri = _material_by_triangle(lod)

    tris_by_bone: dict[int, list[int]] = {}
    for t in range(len(lod.indices) // 3):
        a, b, c = lod.indices[t * 3:t * 3 + 3]
        ba = bone_by_vertex[a] if a < len(bone_by_vertex) else 0
        bb = bone_by_vertex[b] if b < len(bone_by_vertex) else 0
        bc = bone_by_vertex[c] if c < len(bone_by_vertex) else 0
        # Majority vote, tie-broken on the first vertex when all 3 differ.
        bone_idx = ba if (ba == bb or ba == bc) else (bb if bb == bc else ba)
        tris_by_bone.setdefault(bone_idx, []).append(t)

    bone_name_by_index = {i: b.name for i, b in enumerate(lod.bones)}
    return [
        (bone_name_by_index.get(bone_idx, f"part_{bone_idx}"), _extract_lod_by_triangles(lod, tri_list, material_by_tri))
        for bone_idx, tri_list in sorted(tris_by_bone.items())
    ]


def _extract_lod_by_triangles(lod: mesh_codec.Lod, triangle_indices: list, material_by_triangle: list) -> mesh_codec.Lod:
    """Like _extract_submesh_lod, but for an arbitrary (not necessarily
    contiguous) set of triangles instead of one material range - builds a
    fresh local MaterialRange table by grouping consecutive triangles (in
    the given order) that share the same original material."""
    remap: dict[int, int] = {}
    positions, normals, uv0, bone_weights, bone_indices, extra = [], [], [], [], [], []
    local_indices = []
    for t in triangle_indices:
        for vi in lod.indices[t * 3:t * 3 + 3]:
            li = remap.get(vi)
            if li is None:
                li = len(positions)
                remap[vi] = li
                positions.append(lod.positions[vi])
                normals.append(lod.normals[vi])
                uv0.append(lod.uv0[vi])
                bone_weights.append(lod.bone_weights[vi] if vi < len(lod.bone_weights) else (1.0, 0.0, 0.0, 0.0))
                bone_indices.append(lod.bone_indices[vi] if vi < len(lod.bone_indices) else (0, 0, 0, 0))
                extra.append(lod.extra[vi] if vi < len(lod.extra) else (0.0, 0.0, 0.0, 0.0))
            local_indices.append(li)

    materials = []
    cur_name, cur_path, start, count = None, None, 0, 0
    for i, t in enumerate(triangle_indices):
        name, path = material_by_triangle[t]
        if name != cur_name:
            if cur_name is not None and count:
                materials.append(mesh_codec.MaterialRange(name=cur_name, start=start, count=count, path=cur_path))
            cur_name, cur_path, start, count = name, path, i * 3, 0
        count += 3
    if cur_name is not None and count:
        materials.append(mesh_codec.MaterialRange(name=cur_name, start=start, count=count, path=cur_path))

    n = len(positions)
    return mesh_codec.Lod(
        index=lod.index, distance=lod.distance, materials=materials,
        positions=positions, normals=normals, uv0=uv0,
        tangents=[(1.0, 0.0, 0.0, 1.0)] * n, bone_weights=bone_weights, bone_indices=bone_indices,
        indices=local_indices, bones=lod.bones, extra=extra,
        uv1=[(0.0, 0.0)] * n, uv2=[(0.0, 0.0)] * n, uv3=[(0.0, 0.0)] * n,
    )


def _build_object_from_lod(
    name: str, lod: mesh_codec.Lod, material_resolver=None, image_cache=None, texture_temp_dir=None,
    fix_glass_alpha: bool = False,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    bm_verts = [bm.verts.new(p) for p in lod.positions]
    bm.verts.ensure_lookup_table()
    # Maps each bmesh vert's eventual mesh index -> the lod.* array index to
    # read its UV/bone/extra data from. 1:1 with lod.positions until a
    # duplicate-face vertex copy (below) appends to it.
    vert_source_index = list(range(len(lod.positions)))

    for start_idx in range(0, len(lod.indices), 3):
        a, b, c = lod.indices[start_idx:start_idx + 3]
        try:
            bm.faces.new((bm_verts[a], bm_verts[b], bm_verts[c]))
        except ValueError:
            # Genuinely duplicate triangle (same 3 vertices) in the source data
            # - seen on some decimated low LODs. Blender can't store two faces
            # sharing an identical vertex set, so this one gets its own private
            # vertex copies instead, at the same positions - keeps import
            # itself lossless always; "Mesh cleaning" (on request, see
            # _clean_duplicate_faces) is what decides afterwards whether
            # coincident duplicates like this one should stay or go.
            new_verts = []
            for vi in (a, b, c):
                nv = bm.verts.new(lod.positions[vi])
                vert_source_index.append(vi)
                new_verts.append(nv)
            bm.faces.new(new_verts)

    bm.to_mesh(mesh)
    bm.free()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index, vert_index in zip(poly.loop_indices, poly.vertices):
            src_index = vert_source_index[vert_index] if vert_index < len(vert_source_index) else vert_index
            if src_index < len(lod.uv0):
                u, v = lod.uv0[src_index]
            else:
                u, v = 0.0, 0.0
            # DirectX -> Blender/OpenGL V convention. Confirmed correct against
            # original Kunos content (raw V runs 0..1 there). Note some
            # third-party/converted .mesh files carry V negated instead (raw V
            # -1..0), which lands their UVs one tile up (V 1..2) in Blender -
            # harmless for rendering since textures repeat, and handled by the
            # texture painting code's per-pixel wrapping, but it is a quirk of
            # those files, not of this conversion.
            uv_layer.data[loop_index].uv = (u, 1.0 - v)

    mesh.materials.clear()
    slot_by_name: dict[str, int] = {}
    for mat_range in lod.materials:
        if mat_range.name in slot_by_name:
            # Same material used by >1 (not necessarily contiguous) range in
            # this object - share one slot instead of appending a duplicate.
            continue
        mat = bpy.data.materials.get(mat_range.name) or bpy.data.materials.new(mat_range.name)
        if material_resolver is not None:
            resolved = material_resolver.resolve(mat_range.name, mat_range.path)
            _wire_material_textures(mat, resolved, image_cache if image_cache is not None else {}, texture_temp_dir)
        if fix_glass_alpha and mat_range.name in _GLASS_ALPHA_MATERIALS:
            _apply_glass_alpha(mat, _GLASS_ALPHA_VALUE)
        slot_by_name[mat_range.name] = len(mesh.materials)
        mesh.materials.append(mat)
    # Polygons were created in the exact order of lod.indices triples (one
    # bm.faces.new() per consecutive group of 3 indices), so the Nth polygon
    # directly corresponds to indices[3N:3N+3] - no need to search for it.
    poly_index = 0
    for mat_range in lod.materials:
        slot = slot_by_name[mat_range.name]
        n_tris = mat_range.count // 3
        for _ in range(n_tris):
            if poly_index < len(mesh.polygons):
                mesh.polygons[poly_index].material_index = slot
            poly_index += 1

    for poly in mesh.polygons:
        poly.use_smooth = True

    bone_attr = mesh.attributes.new(_BONE_ATTR, "INT", "POINT")
    for i in range(len(mesh.vertices)):
        src_index = vert_source_index[i] if i < len(vert_source_index) else i
        bi = lod.bone_indices[src_index] if src_index < len(lod.bone_indices) else None
        bone_attr.data[i].value = bi[0] if bi else 0

    extra_attr = mesh.attributes.new(_EXTRA_ATTR, "FLOAT_COLOR", "POINT")
    for i in range(len(mesh.vertices)):
        src_index = vert_source_index[i] if i < len(vert_source_index) else i
        ex = lod.extra[src_index] if src_index < len(lod.extra) else None
        extra_attr.data[i].color = ex if ex is not None and len(ex) == 4 else (0.0, 0.0, 0.0, 0.0)
    # Without this, Blender never renders ac_extra in the viewport (Solid
    # shading's "Attribute"/"Vertex" colour mode, and the default colour
    # attribute node in Material Preview/Rendered, both only ever show the
    # mesh's *active* colour attribute) - the paint data itself would still
    # be perfectly correct and still round-trip into the .mesh file either
    # way, it just wouldn't be visible.
    idx = mesh.color_attributes.find(_EXTRA_ATTR)
    if idx != -1:
        mesh.color_attributes.active_color_index = idx

    mesh.update()
    return bpy.data.objects.new(name, mesh)


def _make_dds_blender_friendly(dds_path: str, temp_dir: str) -> str:
    """See the _SRGB_TO_UNORM_DXGI / _LEGACY_FOURCC_BY_DXGI comment above.
    Returns a path to a rewritten copy (in `temp_dir`) if `dds_path` needed
    one of the two header fixups, or `dds_path` itself unchanged otherwise."""
    try:
        with open(dds_path, "rb") as fh:
            data = bytearray(fh.read())
    except OSError:
        return dds_path
    if len(data) < 132 or bytes(data[84:88]) != b"DX10":
        return dds_path

    dxgi_format = struct.unpack_from("<I", data, 128)[0]
    legacy_fourcc = _LEGACY_FOURCC_BY_DXGI.get(dxgi_format)
    if legacy_fourcc is not None:
        # Drop the 20-byte DX10 extended header entirely and tag as the
        # equivalent pre-DX10 FourCC - same compressed bytes, older/wider
        # supported container.
        header = data[:128]
        struct.pack_into("<4s", header, 84, legacy_fourcc)
        pixel_data = data[148:]
        out = bytes(header) + pixel_data
    else:
        new_format = _SRGB_TO_UNORM_DXGI.get(dxgi_format)
        if new_format is None:
            return dds_path
        struct.pack_into("<I", data, 128, new_format)
        out = bytes(data)

    out_path = os.path.join(temp_dir, "fixed_" + os.path.basename(dds_path))
    with open(out_path, "wb") as fh:
        fh.write(out)
    return out_path


def _load_image_cached(path: str, colorspace: str, image_cache: dict, temp_dir: str | None = None) -> bpy.types.Image:
    key = os.path.abspath(path)
    img = image_cache.get(key)
    if img is not None:
        return img

    load_path = path
    if temp_dir and path.lower().endswith(".dds"):
        load_path = _make_dds_blender_friendly(path, temp_dir)

    img = bpy.data.images.load(load_path)
    # Name the datablock after the real texture, not after the temporary
    # rewritten .dds the DDS fixup may have produced - the image name is what
    # the UI shows and what output filenames are derived from.
    real_name = os.path.splitext(os.path.basename(path))[0]
    if real_name:
        img.name = real_name
    try:
        img.colorspace_settings.name = colorspace
    except Exception:  # noqa: BLE001 - colorspace name missing from this OCIO config; not fatal
        pass
    img.pack()  # embed the pixel data now - the source .dds lives in a temp dir that gets deleted
    image_cache[key] = img
    return img


def _wire_material_textures(mat: bpy.types.Material, resolved, image_cache: dict, temp_dir: str | None) -> None:
    """Builds a basic Principled BSDF node graph wired to the material's
    resolved diffuse/normal/opacity textures (or just its constant paint
    colour if it has no textures at all). Idempotent - a material shared by
    several objects/LODs only gets wired once."""
    if mat.get("ac_textures_wired"):
        return
    mat["ac_textures_wired"] = True

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return

    if resolved.channels:
        _wire_channel_material(mat, resolved.channels, image_cache, temp_dir)
        return

    if not (resolved.diffuse_texture or resolved.normal_texture or resolved.opacity_texture):
        r, g, b = resolved.diffuse_color
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        return

    y = 300
    if resolved.diffuse_texture:
        img = _load_image_cached(resolved.diffuse_texture, "sRGB", image_cache, temp_dir)
        # Remembers this image came from a real, already-existing car texture
        # file - exporters use this to tell "keep the original name/path"
        # apart from "this was created fresh in Blender, needs writing out".
        if resolved.diffuse_raw:
            img[_PROP_TEXTURE_PATH] = resolved.diffuse_raw
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.label = "AC Diffuse"
        tex_node.location = (-400, y)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        y -= 300

    if resolved.normal_texture:
        img = _load_image_cached(resolved.normal_texture, "Non-Color", image_cache, temp_dir)
        if resolved.normal_raw:
            img[_PROP_TEXTURE_PATH] = resolved.normal_raw
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.label = "AC Normal"
        tex_node.location = (-700, y)
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (-400, y)
        links.new(tex_node.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        y -= 300

    if resolved.opacity_texture:
        # Deliberately from the material's own OpacityMap, never the diffuse
        # texture's alpha channel (see project_resolver._pick_opacity_slot).
        img = _load_image_cached(resolved.opacity_texture, "Non-Color", image_cache, temp_dir)
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = img
        tex_node.label = "AC Opacity"
        tex_node.location = (-400, y)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Alpha"])
        for method in ("CLIP", "HASHED", "BLEND"):
            try:
                mat.blend_method = method
                break
            except (TypeError, AttributeError):
                continue


_CHANNEL_GROUP_SOCKETS = [
    # (ChannelMaterial field, group output socket name, colourspace)
    ("base_color_texture", "Base Color", "sRGB"),
    ("normal_texture", "Normal", "Non-Color"),
    ("ao_texture", "Ambient Occlusion", "Non-Color"),
    ("anisotropy_texture", "Anisotropy", "Non-Color"),
    ("metalness_texture", "Metalness", "Non-Color"),
    ("opacity_texture", "Opacity", "Non-Color"),
]


def _build_channel_group(channel, image_cache: dict, temp_dir: str | None) -> bpy.types.NodeTree:
    """One small node group per active Base/Red/Green/Blue channel: samples
    whichever of its 6 maps are actually assigned, through a Mapping node
    scaled by the channel's own UVscale, and exposes each as a Color output.
    A map this channel doesn't have comes out as white - a no-op for the
    multiply chain _wire_channel_material builds across channels - except
    Base Color specifically, which falls back to the channel's own constant
    paint colour (or mid-grey, if it has neither)."""
    group = bpy.data.node_groups.new(f"AC_{channel.prefix}channel", "ShaderNodeTree")
    nodes, links = group.nodes, group.links
    for _field, socket_name, _cs in _CHANNEL_GROUP_SOCKETS:
        group.interface.new_socket(socket_name, in_out="OUTPUT", socket_type="NodeSocketColor")
    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (600, 0)

    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.location = (-900, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-700, 0)
    mapping.inputs["Scale"].default_value = (channel.uv_scale[0], channel.uv_scale[1], 1.0)
    links.new(uv_node.outputs["UV"], mapping.inputs["Vector"])

    y = 300
    for field_name, socket_name, colorspace in _CHANNEL_GROUP_SOCKETS:
        converted_path = getattr(channel, field_name)
        if converted_path:
            img = _load_image_cached(converted_path, colorspace, image_cache, temp_dir)
            engine_path = getattr(channel, field_name.replace("_texture", "_raw"), None)
            if engine_path:
                img[_PROP_TEXTURE_PATH] = engine_path
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.image = img
            tex_node.label = f"{channel.prefix}{socket_name}"
            tex_node.location = (-400, y)
            links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
            links.new(tex_node.outputs["Color"], group_out.inputs[socket_name])
        else:
            const = channel.base_color_const if (socket_name == "Base Color" and channel.base_color_const) else None
            rgb_node = nodes.new("ShaderNodeRGB")
            rgb_node.label = f"{channel.prefix}{socket_name} (default)"
            rgb_node.location = (-400, y)
            rgb_node.outputs[0].default_value = (*const, 1.0) if const else (1.0, 1.0, 1.0, 1.0)
            links.new(rgb_node.outputs[0], group_out.inputs[socket_name])
        y -= 250

    return group


def _wire_channel_material(mat: bpy.types.Material, channels: list, image_cache: dict, temp_dir: str | None) -> None:
    """Wires the layered Base/Red/Green/Blue channel system: one node group
    per active channel (_build_channel_group), then Base x Red x Green x Blue
    multiplied together independently for each of the 6 properties - the
    final Base Color and Ambient Occlusion results are multiplied together
    into the Principled BSDF's Base Color (no dedicated AO input exists
    there), Normal goes through a Normal Map node, Metalness/Anisotropy feed
    the matching BSDF inputs directly, and Opacity feeds Alpha (only wired,
    with the material switched out of opaque rendering, if at least one
    channel actually has an opacity map)."""
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    group_nodes = []
    x = -1400
    for channel in channels:
        group_tree = _build_channel_group(channel, image_cache, temp_dir)
        gnode = nodes.new("ShaderNodeGroup")
        gnode.node_tree = group_tree
        gnode.label = channel.prefix.rstrip("_")
        gnode.location = (x, 300)
        x += 220
        group_nodes.append(gnode)

    def multiply_chain(socket_name: str, y: float):
        current = group_nodes[0].outputs[socket_name]
        mix_x = x + 100
        for gnode in group_nodes[1:]:
            mix = nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.location = (mix_x, y)
            links.new(current, mix.inputs["Color1"])
            links.new(gnode.outputs[socket_name], mix.inputs["Color2"])
            current = mix.outputs["Color"]
            mix_x += 200
        return current

    final_base_color = multiply_chain("Base Color", 400)
    final_normal = multiply_chain("Normal", 100)
    final_ao = multiply_chain("Ambient Occlusion", -200)
    final_anisotropy = multiply_chain("Anisotropy", -500)
    final_metalness = multiply_chain("Metalness", -800)

    ao_mix = nodes.new("ShaderNodeMixRGB")
    ao_mix.blend_type = "MULTIPLY"
    ao_mix.label = "Base Color x AO (no BSDF AO input)"
    ao_mix.inputs["Factor"].default_value = 1.0
    ao_mix.location = (x + 900, 400)
    links.new(final_base_color, ao_mix.inputs["Color1"])
    links.new(final_ao, ao_mix.inputs["Color2"])

    normal_map_node = nodes.new("ShaderNodeNormalMap")
    normal_map_node.location = (x + 900, 100)
    links.new(final_normal, normal_map_node.inputs["Color"])

    links.new(ao_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(final_anisotropy, bsdf.inputs["Anisotropic"])
    links.new(final_metalness, bsdf.inputs["Metallic"])

    # Only wire Alpha (and switch the material out of opaque rendering) if a
    # channel actually carries an opacity map - a channel with none is
    # treated as fully opaque (white, same neutral-multiply convention as
    # the other maps), so leaving Alpha at its default 1.0 unconnected for
    # an all-opaque material avoids needlessly blending/dithering it.
    if any(c.opacity_texture for c in channels):
        final_opacity = multiply_chain("Opacity", -1100)
        links.new(final_opacity, bsdf.inputs["Alpha"])
        for method in ("CLIP", "HASHED", "BLEND"):
            try:
                mat.blend_method = method
                break
            except (TypeError, AttributeError):
                continue


def _apply_glass_alpha(mat: bpy.types.Material, alpha: float) -> None:
    """Forces a near-zero Alpha on a glass material's Principled BSDF,
    overriding (removing) any texture link on that socket first - these
    materials must always end up see-through regardless of whether an
    opacity texture happened to get wired to Alpha above."""
    if mat.get("ac_glass_alpha_applied"):
        return
    mat["ac_glass_alpha_applied"] = True

    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        return
    alpha_input = bsdf.inputs.get("Alpha")
    if alpha_input is None:
        return
    if alpha_input.is_linked:
        for link in list(alpha_input.links):
            mat.node_tree.links.remove(link)
    alpha_input.default_value = alpha
    for method in ("HASHED", "BLEND", "CLIP"):
        try:
            mat.blend_method = method
            break
        except (TypeError, AttributeError):
            continue


def _compute_ac_bbox(context) -> tuple[Vector, Vector] | tuple[None, None]:
    """World-space bounding box over every mesh object tagged as coming from
    this addon (not just the LOD(s) from the current import call) - so the
    camera/lighting rig always reflects everything currently in the scene,
    e.g. after importing both an interior and an exterior LOD in turn."""
    min_co = Vector((float("inf"),) * 3)
    max_co = Vector((float("-inf"),) * 3)
    found = False
    for obj in bpy.data.objects:
        if obj.type != "MESH" or _PROP_SOURCE_PATH not in obj:
            continue
        found = True
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            min_co.x, max_co.x = min(min_co.x, wc.x), max(max_co.x, wc.x)
            min_co.y, max_co.y = min(min_co.y, wc.y), max(max_co.y, wc.y)
            min_co.z, max_co.z = min(min_co.z, wc.z), max(max_co.z, wc.z)
    if not found:
        return None, None
    return min_co, max_co


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    if direction.length < 1e-6:
        return
    # up_axis is always 'Y' here - Blender's camera-local "up" convention,
    # independent of the scene's own up-axis (Z here, after fix_orientation).
    # Using 'Z' produces a dutch-angle roll on every shot.
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _ensure_camera(rig_coll: bpy.types.Collection, name: str, lens: float) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "CAMERA":
        obj = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    if rig_coll not in obj.users_collection:
        rig_coll.objects.link(obj)
    obj.data.lens = lens
    return obj


def _ensure_area_light(
    rig_coll: bpy.types.Collection, name: str, energy: float, size: float, color: tuple[float, float, float],
) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "LIGHT":
        obj = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
    if rig_coll not in obj.users_collection:
        rig_coll.objects.link(obj)
    obj.data.energy = energy
    obj.data.size = size
    obj.data.color = color
    return obj


def _ensure_camera_lighting_rig(context) -> None:
    """Builds (or repositions, if it already exists) a "Camera_Lighting_Rig"
    collection with a 3/4 exterior hero camera, an interior cabin camera
    (only if a STEER_BASE bone Empty is present - i.e. an interior LOD with
    hierarchy import is in the scene), and 3-point area lighting (key/fill/
    rim) - sized and aimed from the bounding box of every AC mesh currently
    in the scene, so importing exterior then interior (or vice-versa) grows
    the rig to fit both instead of leaving two separate rigs behind."""
    # matrix_world of objects parented moments ago (bone-hierarchy Empties,
    # axis-fix root) isn't guaranteed to be flushed yet within the same
    # operator call - force it before reading any bounding box/world position.
    context.view_layer.update()
    min_co, max_co = _compute_ac_bbox(context)
    if min_co is None:
        return
    center = (min_co + max_co) / 2
    size = max_co - min_co
    diag = size.length
    if diag < 1e-6:
        return

    rig_coll = bpy.data.collections.get(_RIG_COLLECTION_NAME) or bpy.data.collections.new(_RIG_COLLECTION_NAME)
    if rig_coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(rig_coll)

    distance = diag * 1.25
    theta = math.radians(35)
    cam_ext_loc = Vector((
        center.x - distance * math.sin(theta),
        center.y - distance * math.cos(theta),
        center.z + size.z * 0.9,
    ))
    cam_ext_target = Vector((center.x, center.y, min_co.z + size.z * 0.30))

    cam_ext = _ensure_camera(rig_coll, "CAM_Exterior_34", lens=50)
    cam_ext.location = cam_ext_loc
    _look_at(cam_ext, cam_ext_target)
    if context.scene.camera is None:
        context.scene.camera = cam_ext

    steer = bpy.data.objects.get("STEER_BASE")
    if steer is not None:
        steer_pos = steer.matrix_world.translation
        cam_int_loc = Vector((-steer_pos.x - 0.15, steer_pos.y + 0.10, steer_pos.z + 0.35))
        cam_int_target = Vector((steer_pos.x, steer_pos.y - 0.30, steer_pos.z - 0.05))
        cam_int = _ensure_camera(rig_coll, "CAM_Interior", lens=24)
        cam_int.location = cam_int_loc
        _look_at(cam_int, cam_int_target)

    key_loc = Vector((cam_ext_loc.x * 1.15, cam_ext_loc.y * 1.35, cam_ext_loc.z * 0.75))
    fill_loc = Vector((-cam_ext_loc.x * 1.0, cam_ext_loc.y * 0.85, cam_ext_loc.z * 0.55))
    rim_loc = Vector((center.x * 0.3, max_co.y + distance * 0.5, center.z + size.z * 1.4))

    key = _ensure_area_light(rig_coll, "Light_Key", energy=3500, size=2.2, color=(1.0, 1.0, 1.0))
    key.location = key_loc
    _look_at(key, center)

    fill = _ensure_area_light(rig_coll, "Light_Fill", energy=1200, size=2.5, color=(0.85, 0.9, 1.0))
    fill.location = fill_loc
    _look_at(fill, center)

    rim = _ensure_area_light(rig_coll, "Light_Rim", energy=1800, size=1.5, color=(1.0, 1.0, 1.0))
    rim.location = rim_loc
    _look_at(rim, center)


def _build_bone_hierarchy(context, lod: mesh_codec.Lod, lod_name: str, source_path: str, lod_index: int):
    """Imports lod.bones (the rigid-part hierarchy - doors, mirrors, wipers,
    ...) as a tree of Empty objects under one root Empty named
    "<lod_name>_hierarchy", correctly positioned and parented. Names/
    parenting are read-only, but each Empty's transform is re-read on
    export (see _collect_bone_updates) - moving one and exporting does
    write its new position back to the .mesh file. Returns
    (hierarchy_root, {bone_name: empty}) - the dict is used by "Split by
    rigid part" imports to parent each part's geometry to its own bone."""
    objs_by_name: dict[str, bpy.types.Object] = {}
    for bone in lod.bones:
        empty = bpy.data.objects.new(bone.name, None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.05
        # AC matrices are row-major with translation in row 4 (DirectX/row-vector
        # convention); mathutils.Matrix expects translation in column 4
        # (OpenGL/column-vector convention) - transpose to convert between them.
        empty.matrix_local = Matrix(bone.matrix).transposed()
        empty[_PROP_SOURCE_PATH] = source_path
        empty[_PROP_LOD_INDEX] = lod_index
        # The original bone name, kept separate from empty.name - which Blender
        # silently suffixes ("DOOR_L.001") if the same name is already taken,
        # e.g. re-importing another LOD with an identically-named bone.
        empty[_PROP_BONE_NAME] = bone.name
        objs_by_name[bone.name] = empty

    hierarchy_root = bpy.data.objects.new(f"{lod_name}_hierarchy", None)
    hierarchy_root.empty_display_type = "PLAIN_AXES"
    hierarchy_root[_PROP_SOURCE_PATH] = source_path
    hierarchy_root[_PROP_LOD_INDEX] = lod_index
    context.collection.objects.link(hierarchy_root)

    for bone in lod.bones:
        empty = objs_by_name[bone.name]
        parent = objs_by_name.get(bone.parent) if bone.parent else None
        empty.parent = parent if parent is not None else hierarchy_root
        context.collection.objects.link(empty)

    return hierarchy_root, objs_by_name


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class EXPORT_OT_ac_mesh(Operator, ExportHelper):
    bl_idname = "export_scene.ac_mesh"
    bl_label = "Export AC Mesh (.mesh)"
    bl_options = {"REGISTER"}

    filename_ext = ".mesh"
    filter_glob: StringProperty(default="*.mesh", options={"HIDDEN"})

    def invoke(self, context, event):
        obj = context.active_object
        if obj and _PROP_SOURCE_PATH in obj:
            self.filepath = obj[_PROP_SOURCE_PATH]
        return super().invoke(context, event)

    def execute(self, context):
        # The active object just needs to point at the right source file -
        # it can be a mesh (combined or one split-by-material submesh) or
        # the parent "LOD group" Empty created in split mode; either way,
        # the actual per-LOD data below always comes from scanning *all*
        # tagged mesh objects in the scene, not from this one specifically.
        obj = context.active_object
        if obj is None or _PROP_SOURCE_PATH not in obj:
            self.report(
                {"ERROR"},
                "Select an object imported by this addon (or one of its children) - "
                f"property '{_PROP_SOURCE_PATH}' is missing, so there is no way to tell which "
                ".mesh file to write back to.",
            )
            return {"CANCELLED"}

        source_path = obj[_PROP_SOURCE_PATH]
        if not os.path.isfile(source_path):
            self.report({"ERROR"}, f"Original source file not found: {source_path}")
            return {"CANCELLED"}

        # A mesh in Edit mode keeps its live state in the BMesh: the Mesh
        # datablock still holds the state from when Edit mode was entered, and
        # its attribute collections read as empty. Exporting straight from
        # Edit mode would therefore silently write out stale geometry and drop
        # the bone-index/vertex-colour attributes - flush back to Object mode
        # first (restored afterwards so the user stays where they were).
        restore_edit_mode = obj.mode == "EDIT"
        if restore_edit_mode:
            bpy.ops.object.mode_set(mode="OBJECT")
        try:
            return self._export(context, obj, source_path)
        finally:
            if restore_edit_mode:
                bpy.ops.object.mode_set(mode="EDIT")

    def _export(self, context, obj, source_path):
        try:
            mf = mesh_codec.load_mesh_file(source_path)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to re-read the original file: {exc}")
            return {"CANCELLED"}

        # Export is "all or nothing" per file: every object in the scene
        # tagged with this same source path gets re-encoded fresh from its
        # current Blender state; any LOD with no matching object is copied
        # verbatim from the original. This avoids ending up with some LODs
        # freshly round-tripped through Blender and others not, which can
        # leave the file in a state the game doesn't like (mismatched LOD
        # data) if only a single LOD is ever touched.
        objects_by_lod: dict[int, list] = {}
        for scene_obj in bpy.data.objects:
            if scene_obj.type != "MESH":
                continue
            if scene_obj.get(_PROP_SOURCE_PATH) != source_path:
                continue
            if _PROP_LOD_INDEX not in scene_obj:
                continue
            objects_by_lod.setdefault(int(scene_obj[_PROP_LOD_INDEX]), []).append(scene_obj)

        material_path_for = _make_material_path_resolver(mf, source_path)
        synthesized: set = set()

        updated, preserved = [], []
        for lod_index, original_lod in enumerate(mf.lods):
            objs = objects_by_lod.get(lod_index)
            if not objs:
                preserved.append(lod_index)
                continue
            try:
                # One or several objects (e.g. "Split by material" import,
                # or manually split further) - convert each independently,
                # then concatenate them back into the single combined
                # vertex/index buffer the .mesh format itself expects.
                sub_lods = [
                    _build_lod_from_object(o, lod_index, original_lod, material_path_for, synthesized)
                    for o in objs
                ]
                mf.lods[lod_index] = _merge_lods(sub_lods, lod_index, original_lod)
                mf.lods[lod_index].bones = _collect_bone_updates(source_path, lod_index, original_lod)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"Failed to convert the Blender mesh (LOD {lod_index}): {exc}")
                return {"CANCELLED"}
            updated.append(lod_index)

        try:
            mesh_codec.save_mesh_file(mf, self.filepath)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Failed to write the .mesh: {exc}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Wrote {self.filepath} - LOD(s) updated from Blender: {updated or 'none'}; "
            f"LOD(s) preserved as-is: {preserved or 'none'}",
        )
        if synthesized:
            car_root = os.path.dirname(os.path.dirname(source_path))
            missing = sorted(
                name for name in synthesized
                if not os.path.isfile(os.path.join(car_root, "materials", f"{name}.material"))
            )
            self.report(
                {"WARNING"},
                f".material path rebuilt for {len(synthesized)} material(s) missing from the original file: "
                f"{', '.join(sorted(synthesized))}.",
            )
            if missing:
                # Deliberately a WARNING, not an ERROR: the .mesh itself was
                # written correctly, and reporting ERROR on an operator that
                # returns FINISHED makes scripted bpy.ops calls raise.
                self.report(
                    {"WARNING"},
                    f"WARNING - the matching .material file(s) do NOT exist in "
                    f"{os.path.join(car_root, 'materials')}: {', '.join(missing)}. The game will crash on "
                    "load until they are created in the AC editor.",
                )
        return {"FINISHED"}


def _matrices_close(a, b, eps: float = _BONE_MATRIX_EPS) -> bool:
    return all(abs(a[r][c] - b[r][c]) <= eps for r in range(4) for c in range(4))


def _collect_bone_updates(source_path: str, lod_index: int, original_lod: mesh_codec.Lod) -> list:
    """Re-reads the current transform of every bone Empty imported by
    _build_bone_hierarchy for this LOD, and returns a bones list with a
    freshly-rebuilt Bone (fresh matrix, no `raw`) for whichever ones moved -
    bones with no matching Empty in the scene (hierarchy not imported, or
    the Empty got deleted) or whose Empty didn't move keep the original
    Bone untouched, byte-identical on export."""
    empties_by_bone_name: dict[str, bpy.types.Object] = {}
    for obj in bpy.data.objects:
        if obj.type != "EMPTY":
            continue
        if obj.get(_PROP_SOURCE_PATH) != source_path or obj.get(_PROP_LOD_INDEX) != lod_index:
            continue
        bone_name = obj.get(_PROP_BONE_NAME)
        if bone_name:
            empties_by_bone_name[bone_name] = obj

    updated_bones = []
    for bone in original_lod.bones:
        empty = empties_by_bone_name.get(bone.name)
        if empty is None:
            updated_bones.append(bone)
            continue
        # Undo the transpose applied on import (empty.matrix_local =
        # Matrix(bone.matrix).transposed()) to get back a plain row-major
        # 4x4 list of lists.
        m = empty.matrix_local.transposed()
        new_matrix = [[m[r][c] for c in range(4)] for r in range(4)]
        if _matrices_close(new_matrix, bone.matrix):
            updated_bones.append(bone)
        else:
            updated_bones.append(mesh_codec.Bone(name=bone.name, parent=bone.parent, matrix=new_matrix, raw=None))
    return updated_bones


def _make_material_path_resolver(mf: mesh_codec.MeshFile, source_path: str):
    """Returns name -> engine-relative ".material" path, used for materials
    that exist in Blender but not in the original file (renamed, or split off
    after having been merged away earlier). Writing a MaterialRange with no
    path at all makes the game abort while loading the car with "Trying to
    load a message with an empty path", so a plausible path must always be
    emitted - the .material file itself is authored separately in the AC
    editor and simply has to exist under the car's materials/ folder.

    The folder is taken from whichever material range already carries a path
    (they all live in the same folder), falling back to rebuilding it from
    the .mesh file's own location."""
    for lod in mf.lods:
        for mr in lod.materials:
            if mr.path and "\\" in mr.path:
                folder = mr.path.rsplit("\\", 1)[0]
                return lambda name: f"{folder}\\{name}.material"

    # Fallback: <...>/content/cars/<car>/meshes/foo.mesh -> content\cars\<car>\materials
    parts = os.path.normpath(source_path).split(os.sep)
    lowered = [p.lower() for p in parts]
    if "content" in lowered and len(parts) >= 3:
        start = len(lowered) - 1 - lowered[::-1].index("content")
        car_parts = parts[start:-2]  # drop the "meshes" folder and the filename
        if car_parts:
            folder = "\\".join(car_parts + ["materials"])
            return lambda name: f"{folder}\\{name}.material"
    return lambda name: f"materials\\{name}.material"


def _build_lod_from_object(
    obj: bpy.types.Object, lod_index: int, original_lod: mesh_codec.Lod,
    material_path_for=None, synthesized: set | None = None,
) -> mesh_codec.Lod:
    mesh = obj.data
    mesh.calc_loop_triangles()
    if hasattr(mesh, "calc_normals_split"):
        # Removed in Blender 4.1+ (loop normals are always available there
        # without an explicit call) - only needed on older versions.
        mesh.calc_normals_split()
    try:
        mesh.calc_tangents()
        has_tangents = True
    except Exception:  # noqa: BLE001 - typically: no UV layer to derive tangents from
        has_tangents = False

    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    bone_attr = mesh.attributes.get(_BONE_ATTR)
    extra_attr = mesh.attributes.get(_EXTRA_ATTR)

    # A rigid part re-parented (in the Outliner) onto a *different* bone
    # Empty than the one it was imported under should rigidly follow that
    # bone instead - the per-vertex ac_bone_index attribute alone can't
    # express that since it's only ever written once, at import. Whenever
    # the object's direct parent is a recognised bone Empty, every vertex in
    # this object is forced onto that one bone; otherwise (parented to the
    # LOD's own hierarchy root/group, or nothing) the per-vertex attribute
    # from the original file still applies unchanged, matching non-rigid or
    # already-multi-bone meshes ("split by material" imports, etc).
    forced_bone_index = None
    if obj.parent is not None and obj.parent.type == "EMPTY":
        parent_bone_name = obj.parent.get(_PROP_BONE_NAME)
        if parent_bone_name is not None:
            bone_index_by_name = {b.name: i for i, b in enumerate(original_lod.bones)}
            forced_bone_index = bone_index_by_name.get(parent_bone_name)

    key_to_index: dict[tuple, int] = {}
    positions, normals, uv0, tangents, bone_weights, bone_indices, extra = [], [], [], [], [], [], []
    triangles: list[tuple[int, int, int]] = []

    for tri in mesh.loop_triangles:
        corner_ids = []
        for loop_index in tri.loops:
            loop = mesh.loops[loop_index]
            vi = loop.vertex_index

            normal = tuple(loop.normal)
            if uv_layer is not None:
                u, v = uv_layer[loop_index].uv
                uv_raw = (u, 1.0 - v)  # undo the Blender/OpenGL <-> DirectX V flip applied on import
            else:
                uv_raw = (0.0, 0.0)

            key = (vi, normal, uv_raw)
            idx = key_to_index.get(key)
            if idx is None:
                idx = len(positions)
                key_to_index[key] = idx
                positions.append(tuple(mesh.vertices[vi].co))
                normals.append(normal)
                uv0.append(uv_raw)
                if has_tangents:
                    t = loop.tangent
                    tangents.append((t[0], t[1], t[2], loop.bitangent_sign))
                else:
                    tangents.append((1.0, 0.0, 0.0, 1.0))
                bone_weights.append((1.0, 0.0, 0.0, 0.0))
                if forced_bone_index is not None:
                    bidx = forced_bone_index
                else:
                    bidx = bone_attr.data[vi].value if bone_attr else 0
                bone_indices.append((bidx, 0, 0, 0))
                ex = tuple(extra_attr.data[vi].color) if extra_attr else (0.0, 0.0, 0.0, 0.0)
                extra.append(ex)
            corner_ids.append(idx)
        triangles.append(tuple(corner_ids))

    num_slots = max(len(mesh.materials), 1)
    tris_by_material: list[list[int]] = [[] for _ in range(num_slots)]
    for tri, tri_info in zip(triangles, mesh.loop_triangles):
        slot = tri_info.material_index if tri_info.material_index < num_slots else 0
        tris_by_material[slot].extend(tri)

    original_by_name = {mr.name: mr for mr in original_lod.materials}
    materials = []
    indices: list[int] = []
    offset = 0
    for slot, mat in enumerate(mesh.materials or [None]):
        mat_name = mat.name if mat is not None else "Default"
        tri_indices = tris_by_material[slot]
        count = len(tri_indices)
        orig = original_by_name.get(mat_name)
        path = orig.path if orig else None
        if not path and material_path_for is not None:
            path = material_path_for(mat_name)
            if synthesized is not None:
                synthesized.add(mat_name)
        materials.append(mesh_codec.MaterialRange(name=mat_name, start=offset, count=count, path=path))
        indices.extend(tri_indices)
        offset += count

    n = len(positions)
    return mesh_codec.Lod(
        index=lod_index,
        distance=original_lod.distance,
        materials=materials,
        positions=positions,
        normals=normals,
        uv0=uv0,
        tangents=tangents,
        bone_weights=bone_weights,
        bone_indices=bone_indices,
        indices=indices,
        bones=original_lod.bones,
        extra=extra,
        uv1=[(0.0, 0.0)] * n,
        uv2=[(0.0, 0.0)] * n,
        uv3=[(0.0, 0.0)] * n,
    )


def _merge_lods(sub_lods: list, lod_index: int, original_lod: mesh_codec.Lod) -> mesh_codec.Lod:
    """Concatenates one or more (already vertex-compacted, 0-based-indexed)
    Lods - one per Blender object - into the single combined buffer the
    .mesh format stores per LOD. Works just as well for a single combined
    multi-material object (one sub-Lod, offset 0 - a no-op merge) as for
    several single-material objects from a "Split by material" import."""
    positions, normals, uv0, tangents, bone_weights, bone_indices, extra = [], [], [], [], [], [], []
    indices: list[int] = []
    materials = []
    offset = 0

    for sub in sub_lods:
        vertex_offset = len(positions)
        positions.extend(sub.positions)
        normals.extend(sub.normals)
        uv0.extend(sub.uv0)
        tangents.extend(sub.tangents)
        bone_weights.extend(sub.bone_weights)
        bone_indices.extend(sub.bone_indices)
        extra.extend(sub.extra)

        for mr in sub.materials:
            local_indices = sub.indices[mr.start:mr.start + mr.count]
            global_indices = [i + vertex_offset for i in local_indices]
            materials.append(mesh_codec.MaterialRange(name=mr.name, start=offset, count=len(global_indices), path=mr.path))
            indices.extend(global_indices)
            offset += len(global_indices)

    n = len(positions)
    return mesh_codec.Lod(
        index=lod_index, distance=original_lod.distance, materials=materials,
        positions=positions, normals=normals, uv0=uv0, tangents=tangents,
        bone_weights=bone_weights, bone_indices=bone_indices, indices=indices,
        bones=original_lod.bones, extra=extra,
        uv1=[(0.0, 0.0)] * n, uv2=[(0.0, 0.0)] * n, uv3=[(0.0, 0.0)] * n,
    )


# ---------------------------------------------------------------------------
# vertex / texture paint (AC EVO SDK "UV AND VERTEX COLOURS" + lights function
# textures - see sdk.pdf pages 11 and 14-15)
# ---------------------------------------------------------------------------

# Per-vertex RGB used on light meshes (solid and glass) to mark their
# position, decoded/encoded losslessly through the existing `extra` Lod field
# (mesh_codec field 13) <-> `ac_extra` FLOAT_COLOR attribute already wired up
# on import/export - painting this attribute is all that's needed, no format
# change required. "Rear Left"/"Rear Right" are Pink/Teal in the SDK's colour
# swatches, which only matches R+B / G+B (magenta/cyan) - the SDK text itself
# says "R+G" for Rear Left, seemingly a typo since that's identical to Yellow
# (Centre Front)'s formula right below it; using R+B here instead.
_LIGHT_POSITIONS = [
    ("FRONT_LEFT", "Front Left (Red)", (1.0, 0.0, 0.0)),
    ("FRONT_RIGHT", "Front Right (Green)", (0.0, 1.0, 0.0)),
    ("REAR_LEFT", "Rear Left (Pink)", (1.0, 0.0, 1.0)),
    ("REAR_RIGHT", "Rear Right (Teal)", (0.0, 1.0, 1.0)),
    ("CENTRE_FRONT", "Centre Front (Yellow)", (1.0, 1.0, 0.0)),
    ("CENTRE_REAR", "Centre Rear (White)", (1.0, 1.0, 1.0)),
]
_DISC_POSITIONS = [
    ("DISC_FRONT", "Disc Front (Red)", (1.0, 0.0, 0.0)),
    ("DISC_REAR", "Disc Rear (Green)", (0.0, 1.0, 0.0)),
]
_POSITION_COLORS = {key: color for key, _label, color in _LIGHT_POSITIONS + _DISC_POSITIONS}
_POSITION_ENUM_ITEMS = [(key, label, "") for key, label, _color in _LIGHT_POSITIONS + _DISC_POSITIONS]

# Lights function textures (sdk.pdf p.14-15) - unlike vertex colour position,
# these are ordinary UV-space textures the addon doesn't otherwise touch;
# painting them means rasterizing the selected faces' UV footprint into a
# user-picked Image datablock, one RGB channel at a time.
#
# Each F_1 channel carries a different meaning depending on whether the light
# sits at the front or the rear of the car - the engine tells the two apart
# from the front/rear vertex colour painted above, not from the texture. So
# "Front > Daylight" and "Rear > Light" deliberately write the very same
# channel; they are split here only to match how the SDK lists them.
#
# Each entry is (channel, button label, Scene property holding its intensity).
# The intensity is simply the value written into that channel: in AC EVO the
# colour intensity drives the light's brightness, so a low beam painted at 0.5
# does not flare like a high beam painted at 0.8. 1.0 = 255, 0.0 = off.
_LIGHTS_F1_FRONT = [
    ("R", "Daylight", "ac_intensity_daylight"),
    ("G", "Low beam", "ac_intensity_lowbeam"),
    ("B", "High beam", "ac_intensity_highbeam"),
]
_LIGHTS_F1_REAR = [
    ("R", "Light", "ac_intensity_rear_light"),
    ("G", "Brake", "ac_intensity_brake"),
    ("B", "Rain / Fog", "ac_intensity_rainfog"),
]
_LIGHTS_F2_CHANNELS = [
    ("R", "Indicator", "ac_intensity_indicator"),
    ("G", "Reverse", "ac_intensity_reverse"),
    ("B", "Special", "ac_intensity_special"),
]
_CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}

# Registered on Scene in register(); defaults chosen so the relative brightness
# of the functions is sane out of the box.
_INTENSITY_PROPS = [
    ("ac_intensity_daylight", "Daylight", 0.5),
    ("ac_intensity_lowbeam", "Low beam", 0.7),
    ("ac_intensity_highbeam", "High beam", 1.0),
    ("ac_intensity_rear_light", "Rear light", 0.5),
    ("ac_intensity_brake", "Brake", 0.8),
    ("ac_intensity_rainfog", "Rain / Fog", 0.8),
    ("ac_intensity_indicator", "Indicator", 0.7),
    ("ac_intensity_reverse", "Reverse", 0.7),
    ("ac_intensity_special", "Special", 0.7),
]


def _meshes_in_edit_mode(context) -> list:
    """Every mesh currently in Edit mode, not only the active one. Blender's
    multi-object edit mode puts *all* selected meshes into Edit at once, and
    the usual light-painting workflow is exactly that: isolate a handful of
    light meshes, select everything, paint in one go. Operating on
    context.active_object alone would silently paint just the first one."""
    objs = getattr(context, "objects_in_mode_unique_data", None)
    if objs:
        return [o for o in objs if o.type == "MESH"]
    obj = context.active_object
    if obj is not None and obj.type == "MESH" and obj.mode == "EDIT":
        return [obj]
    return []


def _selected_face_groups(objects) -> list:
    """[(bm, uv_layer, [faces]), ...] for every edit-mode mesh with a UV layer
    and a face selection. The BMesh is kept in the tuple on purpose: dropping
    it would let Python garbage-collect the wrapper, which invalidates every
    face/layer reference taken from it ("BMesh data of type BMFace has been
    removed") as soon as more than one object is involved."""
    groups = []
    for obj in objects:
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            continue
        faces = [f for f in bm.faces if f.select]
        if faces:
            groups.append((bm, uv_layer, faces))
    return groups


def _get_or_create_extra_bm_layer(bm: bmesh.types.BMesh):
    layer = bm.verts.layers.float_color.get(_EXTRA_ATTR)
    if layer is None:
        layer = bm.verts.layers.float_color.new(_EXTRA_ATTR)
    return layer


def _rasterize_uv_triangle(pixels, width: int, height: int, uv0, uv1, uv2, channel_writes, painted_pixels: set) -> None:
    """Rasterizes one UV triangle with repeat (GL_REPEAT) semantics: the
    triangle is scan-converted in its own *unwrapped* pixel space, and each
    covered pixel is wrapped back into the image with a modulo. UVs on real
    content routinely sit far outside the 0..1 tile (repeating trim strips
    seen spanning u=-19..20), and individual triangles both straddle tile
    boundaries and span more than a whole tile - so neither clamping nor
    re-wrapping the triangle as a block works; only per-pixel wrapping keeps
    the shape intact and matches how the texture is actually sampled."""
    # No V flip here: Blender's image.pixels buffer is stored bottom-row
    # first, which already matches Blender's UV space (v=0 at the bottom).
    # Flipping V would paint the texture upside-down relative to the UVs.
    xs = (uv0.x * width, uv1.x * width, uv2.x * width)
    ys = (uv0.y * height, uv1.y * height, uv2.y * height)
    # Bounds stay in unwrapped space (no clamping to the image) - wrapping
    # happens per pixel at write time instead.
    min_x, max_x = int(math.floor(min(xs))), int(math.ceil(max(xs)))
    min_y, max_y = int(math.floor(min(ys))), int(math.ceil(max(ys)))
    x0, y0, x1, y1, x2, y2 = xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return
    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            cx, cy = px + 0.5, py + 0.5
            a = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / denom
            b = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / denom
            c = 1.0 - a - b
            if a >= -1e-6 and b >= -1e-6 and c >= -1e-6:
                wx, wy = px % width, py % height
                for channel_index, value in channel_writes:
                    pixels[wy, wx, channel_index] = value
                painted_pixels.add((wx, wy))


def _paint_faces_to_image(image: bpy.types.Image, face_groups, channel_writes: list) -> int:
    """Rasterizes the UV footprint of every selected face (fan-triangulated
    for n-gons) into `image`, writing every (channel_index, value) pair in
    `channel_writes` to each covered pixel. `face_groups` is a list of
    (uv_layer, faces) so several meshes in multi-object edit mode all land in
    the same single read-modify-write pass. Returns the distinct pixel count."""
    width, height = image.size
    if width == 0 or height == 0:
        return 0
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    painted_pixels: set = set()
    for _bm, uv_layer, faces in face_groups:
        for f in faces:
            loops = list(f.loops)
            if len(loops) < 3:
                continue
            uv0 = loops[0][uv_layer].uv
            for i in range(1, len(loops) - 1):
                uv1 = loops[i][uv_layer].uv
                uv2 = loops[i + 1][uv_layer].uv
                _rasterize_uv_triangle(pixels, width, height, uv0, uv1, uv2, channel_writes, painted_pixels)
    if painted_pixels:
        image.pixels = pixels.reshape(-1).tolist()
        image.update()
    return len(painted_pixels)


def _selection_uv_tile_span(face_groups) -> tuple:
    """Returns how many whole UV tiles the selection covers in U and V.
    Anything meaningfully above 1 means the mesh reuses a repeating texture
    strip across several tiles, so any paint applied here wraps around and
    smears across the image instead of staying a localised mask - real
    content does this on trim/chrome strips (spans of 30+ tiles seen)."""
    us, vs = [], []
    for _bm, uv_layer, faces in face_groups:
        for f in faces:
            for loop in f.loops:
                uv = loop[uv_layer].uv
                us.append(uv.x)
                vs.append(uv.y)
    if not us:
        return 0.0, 0.0
    return max(us) - min(us), max(vs) - min(vs)


def _image_to_array(image: bpy.types.Image):
    w, h = image.size
    return np.array(image.pixels[:], dtype=np.float32).reshape(h, w, 4)


def _luminance_of(image: bpy.types.Image):
    """Rec.709 luma of an image, as an (h, w) array. Pixel values are used
    exactly as Blender stores them (verified: writing 0.5 reads back 128/255,
    i.e. no colour-management round-trip), so multiplying by this matches what
    a "Multiply" layer does on 8-bit data in Photoshop/Krita rather than
    producing a different, linear-space result."""
    px = _image_to_array(image)
    return 0.2126 * px[:, :, 0] + 0.7152 * px[:, :, 1] + 0.0722 * px[:, :, 2]


def _resample_nearest(plane, width: int, height: int):
    """Nearest-neighbour resize so a reference image of a different resolution
    than the target still lines up (both share the mesh's UV space)."""
    src_h, src_w = plane.shape
    if (src_h, src_w) == (height, width):
        return plane
    yi = np.clip((np.arange(height) * src_h) // height, 0, src_h - 1)
    xi = np.clip((np.arange(width) * src_w) // width, 0, src_w - 1)
    return plane[yi][:, xi]


# Blender's own undo does NOT cover image pixels written from Python: verified
# by pushing an undo step, writing pixels, then undoing - the pixels stay
# modified. So paint/erase snapshot the image here first and a dedicated
# "Undo paint" button restores the last snapshot.
_PAINT_HISTORY: dict = {}
_PAINT_HISTORY_LIMIT = 12


def _snapshot_image(image: bpy.types.Image):
    px = np.array(image.pixels[:], dtype=np.float32)
    if not image.is_float:
        # Byte images: uint8 is lossless here and a quarter of the memory.
        return ("u8", np.clip(px * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8))
    return ("f32", px)


def _push_paint_history(image: bpy.types.Image) -> None:
    hist = _PAINT_HISTORY.setdefault(image.name, [])
    hist.append(_snapshot_image(image))
    if len(hist) > _PAINT_HISTORY_LIMIT:
        hist.pop(0)


def _pop_paint_history(image: bpy.types.Image) -> bool:
    hist = _PAINT_HISTORY.get(image.name)
    if not hist:
        return False
    kind, data = hist.pop()
    px = (data.astype(np.float32) / 255.0) if kind == "u8" else data
    image.pixels = px.tolist()
    image.update()
    return True


def _paint_history_depth(image) -> int:
    if image is None:
        return 0
    return len(_PAINT_HISTORY.get(image.name, ()))


def _fill_image(image: bpy.types.Image, rgb) -> None:
    width, height = image.size
    if width == 0 or height == 0:
        return
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    pixels[:, :, 0] = rgb[0]
    pixels[:, :, 1] = rgb[1]
    pixels[:, :, 2] = rgb[2]
    image.pixels = pixels.reshape(-1).tolist()
    image.update()


class MESH_OT_ac_paint_position(Operator):
    """Paints the AC EVO light/disc position colour (see sdk.pdf p.11) onto
    the selected vertices' `ac_extra` colour attribute - round-trips through
    the .mesh file's field 13 automatically via the existing import/export
    wiring, no format change needed."""
    bl_idname = "mesh.ac_paint_position"
    bl_label = "Paint AC Vertex Color Position"
    bl_options = {"REGISTER", "UNDO"}

    position: EnumProperty(items=_POSITION_ENUM_ITEMS)

    def execute(self, context):
        objects = _meshes_in_edit_mode(context)
        if not objects:
            self.report({"ERROR"}, "Select a mesh and switch to Edit mode.")
            return {"CANCELLED"}
        r, g, b = _POSITION_COLORS[self.position]
        count, touched = 0, 0
        for obj in objects:
            mesh = obj.data
            bm = bmesh.from_edit_mesh(mesh)
            layer = _get_or_create_extra_bm_layer(bm)
            n = 0
            for v in bm.verts:
                if v.select:
                    v[layer] = (r, g, b, 1.0)
                    n += 1
            if n:
                bmesh.update_edit_mesh(mesh)
                touched += 1
                count += n
            # ac_extra can exist on a mesh (e.g. one produced by Prepare /
            # Is plastic-glass-chrome, which round-trips attributes through
            # bmesh) without ever having been the *active* colour attribute -
            # Blender only ever renders that one in the viewport, so the
            # paint above would otherwise apply correctly and still be
            # invisible. Cheap enough to just re-set on every paint.
            idx = mesh.color_attributes.find(_EXTRA_ATTR)
            if idx != -1:
                mesh.color_attributes.active_color_index = idx
        if count == 0:
            self.report({"WARNING"}, "No vertices selected.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"{self.position}: {count} vertex(es) painted across {touched} mesh(es).")
        return {"FINISHED"}


class MESH_OT_ac_erase_vcolor(Operator):
    """Resets the `ac_extra` colour attribute to black - either just the
    current selection, or (with a confirmation prompt) the whole mesh."""
    bl_idname = "mesh.ac_erase_vcolor"
    bl_label = "Erase AC Vertex Color"
    bl_options = {"REGISTER", "UNDO"}

    whole_mesh: BoolProperty(default=False)

    def invoke(self, context, event):
        if self.whole_mesh:
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        objects = _meshes_in_edit_mode(context)
        if not objects:
            self.report({"ERROR"}, "Select a mesh and switch to Edit mode.")
            return {"CANCELLED"}
        count = 0
        for obj in objects:
            bm = bmesh.from_edit_mesh(obj.data)
            layer = _get_or_create_extra_bm_layer(bm)
            n = 0
            for v in bm.verts:
                if self.whole_mesh or v.select:
                    v[layer] = (0.0, 0.0, 0.0, 1.0)
                    n += 1
            if n:
                bmesh.update_edit_mesh(obj.data)
            count += n
        scope = "whole mesh(es)" if self.whole_mesh else "selection"
        self.report({"INFO"}, f"{count} vertex(es) reset to black ({scope}, {len(objects)} mesh(es)).")
        return {"FINISHED"}


class MESH_OT_ac_erase_all_vcolor(Operator):
    """Full reset: fills the `ac_extra` colour attribute with black on EVERY
    AC-imported mesh in the scene, whole mesh, regardless of selection or
    which object is active - `ac_extra` is the only vertex colour that
    round-trips into the .mesh file, so clearing it clears everything the
    exporter would write. Split-by-material imports produce dozens of
    objects, so a per-object erase alone would silently leave most of the
    car still painted."""
    bl_idname = "mesh.ac_erase_all_vcolor"
    bl_label = "Erase ALL AC vertex colours (black)"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        targets = [
            o for o in bpy.data.objects
            if o.type == "MESH" and _PROP_SOURCE_PATH in o
        ]
        if not targets:
            self.report({"WARNING"}, "No mesh imported by this addon in the scene.")
            return {"CANCELLED"}

        objects_done = 0
        verts_done = 0
        for obj in targets:
            mesh = obj.data
            # Blender supports multi-object edit mode, and a split-by-material
            # import leaves every sub-object selected - so entering Edit mode
            # puts them ALL in it, not just the active one. A mesh in Edit mode
            # keeps its data in the BMesh and exposes an empty
            # mesh.attributes[...].data, so writing through the attribute API
            # there silently does nothing: dispatch per object on its own mode.
            if obj.mode == "EDIT":
                bm = bmesh.from_edit_mesh(mesh)
                layer = _get_or_create_extra_bm_layer(bm)
                for v in bm.verts:
                    v[layer] = (0.0, 0.0, 0.0, 1.0)
                    verts_done += 1
                bmesh.update_edit_mesh(mesh)
            else:
                attr = mesh.attributes.get(_EXTRA_ATTR)
                if attr is None:
                    attr = mesh.attributes.new(_EXTRA_ATTR, "FLOAT_COLOR", "POINT")
                for item in attr.data:
                    item.color = (0.0, 0.0, 0.0, 1.0)
                    verts_done += 1
                mesh.update()
            idx = mesh.color_attributes.find(_EXTRA_ATTR)
            if idx != -1:
                mesh.color_attributes.active_color_index = idx
            objects_done += 1

        self.report({"INFO"}, f"Vertex colours reset to black: {objects_done} mesh(es), {verts_done} vertex(es).")
        return {"FINISHED"}


def _clean_texture_basename(name: str) -> str:
    """Turns a Blender image name back into the plain texture base name:
    strips the "blenderfriendly_" prefix the DDS fixup adds, the "_BW" suffix
    of the greyscale copy, and any file extension."""
    base = name
    if base.startswith("blenderfriendly_"):
        base = base[len("blenderfriendly_"):]
    if base.endswith("_BW"):
        base = base[:-3]
    for ext in (".dds", ".png", ".tga", ".jpg", ".jpeg", ".texture"):
        if base.lower().endswith(ext):
            base = base[:-len(ext)]
            break
    return base


class MESH_OT_ac_generate_normal_map(Operator):
    """Generates a tangent-space normal map from the reference (optic) image.

    Uses the algorithm of NormalMapGenerator by Mehdi-Antoine (MIT licence),
    https://github.com/Mehdi-Antoine/NormalMapGenerator - reimplemented on
    plain numpy in normal_map.py since Blender ships no scipy."""
    bl_idname = "mesh.ac_generate_normal_map"
    bl_label = "Generate normal map"
    bl_options = {"REGISTER", "UNDO"}

    smooth: FloatProperty(
        name="Smooth", default=0.0, min=0.0, max=16.0,
        description="Gaussian blur applied before the gradient - raise it to calm down noisy sources",
    )
    intensity: FloatProperty(
        name="Intensity", default=1.0, min=0.05, max=10.0,
        description="Relief strength: higher pushes the normals further away from flat",
    )

    def invoke(self, context, event):
        if context.scene.ac_lights_ref_image is None:
            self.report({"ERROR"}, "Pick a reference image first ('Use actual').")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        ref = context.scene.ac_lights_ref_image
        if ref is None or ref.size[0] == 0:
            self.report({"ERROR"}, "Pick a reference image first ('Use actual').")
            return {"CANCELLED"}
        try:
            image = _generate_normal_map_from_ref(ref, self.smooth, self.intensity)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Normal map generation failed: {exc}")
            return {"CANCELLED"}
        context.scene.ac_normal_map_image = image
        self.report(
            {"INFO"},
            f"'{image.name}' ({image.size[0]}x{image.size[1]}) generated from '{ref.name}' "
            f"(smooth={self.smooth:g}, intensity={self.intensity:g}).",
        )
        return {"FINISHED"}


class MESH_OT_ac_create_light_texture(Operator):
    """Creates a new, blank black Image datablock (no AO/content baked in -
    a plain starting point to paint the light function channels onto) and
    assigns it to the chosen texture slot."""
    bl_idname = "mesh.ac_create_light_texture"
    bl_label = "Create AC Light Texture"
    bl_options = {"REGISTER", "UNDO"}

    texture_slot: EnumProperty(items=[("F1", "EXT_Lights_F_1", ""), ("F2", "EXT_Lights_F_2", "")])
    width: IntProperty(name="Width", default=1024, min=4, max=8192)
    height: IntProperty(name="Height", default=1024, min=4, max=8192)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        name = "EXT_Lights_F_1" if self.texture_slot == "F1" else "EXT_Lights_F_2"
        image = bpy.data.images.new(name, width=self.width, height=self.height, alpha=True)
        pixels = np.zeros((self.height, self.width, 4), dtype=np.float32)
        pixels[:, :, 3] = 1.0
        image.pixels = pixels.reshape(-1).tolist()
        image.update()
        if self.texture_slot == "F1":
            context.scene.ac_lights_f1_image = image
        else:
            context.scene.ac_lights_f2_image = image
        self.report({"INFO"}, f"New texture '{image.name}' created ({self.width}x{self.height}), black.")
        return {"FINISHED"}


def _find_linked_image(mat: bpy.types.Material, socket_name: str):
    """Walks back from the material's Principled BSDF `socket_name` input and
    returns the first Image Texture feeding it - directly, or through any
    intermediate nodes (mix, gamma, a Normal Map node, ...)."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None or bsdf.type != "BSDF_PRINCIPLED":
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None
    socket = bsdf.inputs.get(socket_name)
    if socket is None or not socket.is_linked:
        return None

    # Each queue entry also carries the specific *output socket* it was
    # reached through - needed once a GROUP node is involved, since one
    # group bundles several independent pipelines (Base Color, Normal, AO,
    # ...) behind one Group Output; without tracking which socket the outer
    # link actually used, descending into the group would just as happily
    # return an image feeding a *different* one of its outputs.
    seen = set()
    queue = [(link.from_node, link.from_socket) for link in socket.links]
    while queue:
        node, from_socket = queue.pop(0)
        key = (node.name, from_socket.identifier if from_socket else None)
        if key in seen:
            continue
        seen.add(key)
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node.image
        if node.type == "GROUP" and node.node_tree is not None:
            # A node group's own `.inputs` are its *external* sockets (fed
            # from outside) - the actual wiring is inside node.node_tree, so
            # continue the search from its Group Output node instead of
            # treating the group as a dead end. Needed for the layered
            # Base/Red/Green/Blue channel materials (_wire_material_textures),
            # which hide their image textures one level down inside a group
            # per channel - only the internal input matching the specific
            # output socket used to reach this group is followed, the
            # group's other, unrelated pipelines are left alone.
            group_output = next((n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT"), None)
            if group_output is not None and from_socket is not None:
                inner_socket = group_output.inputs.get(from_socket.name)
                if inner_socket is not None:
                    queue.extend((link.from_node, link.from_socket) for link in inner_socket.links)
            continue
        for inp in node.inputs:
            queue.extend((link.from_node, link.from_socket) for link in inp.links)
    return None


def _find_base_color_image(mat: bpy.types.Material):
    return _find_linked_image(mat, "Base Color")


def _find_normal_map_image(mat: bpy.types.Material):
    return _find_linked_image(mat, "Normal")


def _make_bw_copy(image: bpy.types.Image) -> bpy.types.Image:
    """Returns a black & white (Rec.709 luma) copy of `image`, reusing the
    "<name>_BW" datablock when one of the right size already exists so that
    repeatedly pressing the button doesn't pile up duplicates."""
    w, h = image.size
    name = f"{image.name}_BW"
    target = bpy.data.images.get(name)
    if target is None or tuple(target.size) != (w, h):
        target = bpy.data.images.new(name, width=w, height=h, alpha=True)
    lum = _luminance_of(image)
    px = np.zeros((h, w, 4), dtype=np.float32)
    px[:, :, 0] = px[:, :, 1] = px[:, :, 2] = lum
    px[:, :, 3] = 1.0
    target.pixels = px.reshape(-1).tolist()
    target.update()
    return target


def _generate_normal_map_from_ref(ref: bpy.types.Image, smooth: float = 0.0, intensity: float = 1.0) -> bpy.types.Image:
    """Shared by the "Generate normal map" button and Prepare's auto-step:
    runs the NormalMapGenerator algorithm on `ref` and returns the resulting
    "<ref>_nm" Image datablock (reused if one of the right size already
    exists, same rule as _make_bw_copy)."""
    w, h = ref.size
    src = _image_to_array(ref)[:, :, :3]
    normal = normal_map.generate_normal_map(src, smooth, intensity)
    name = f"{_clean_texture_basename(ref.name)}_nm"
    image = bpy.data.images.get(name)
    if image is None or tuple(image.size) != (w, h):
        image = bpy.data.images.new(name, width=w, height=h, alpha=True)
    # Colorspace FIRST: switching it afterwards discards the buffer that
    # was just written (verified - a pixel written as 0.5 reads back as 0
    # once the colorspace changes), which silently produced a black map.
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:  # noqa: BLE001 - colorspace name absent from this OCIO config
        pass
    out = np.ones((h, w, 4), dtype=np.float32)
    out[:, :, :3] = normal
    image.pixels = out.reshape(-1).tolist()
    image.update()
    return image


def _image_to_array(img: bpy.types.Image) -> np.ndarray:
    w, h = img.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


def _array_to_image(name: str, arr: np.ndarray, colorspace: str) -> bpy.types.Image:
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    try:
        img.colorspace_settings.name = colorspace
    except Exception:  # noqa: BLE001 - colorspace name missing from this OCIO config; not fatal
        pass
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))
    img.pack()
    img.update()
    return img


def _sample_bilinear_wrapped(src: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear-samples `src` (shape (h, w, 4)) at UV coordinates `u`/`v`
    (any shape, same shape as each other), wrapping with GL_REPEAT semantics
    - real AC content routinely tiles a texture (chrome trim strips seen
    spanning dozens of tiles), so this has to wrap, not clamp."""
    h, w = src.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0i, x1i = x0 % w, (x0 + 1) % w
    y0i, y1i = y0 % h, (y0 + 1) % h
    top = src[y0i, x0i] * (1 - fx) + src[y0i, x1i] * fx
    bot = src[y1i, x0i] * (1 - fx) + src[y1i, x1i] * fx
    return top * (1 - fy) + bot * fy


def _bake_triangle_into(dst: np.ndarray, dst_uv: np.ndarray, src: np.ndarray, src_uv: np.ndarray) -> None:
    """Bakes a triangle's worth of `src` pixels into `dst`.

    `dst_uv` and `src_uv` are each a (3, 2) array of the *same* triangle's
    corners, one in each image's own UV space - they don't need to be the
    same shape or size, only the same winding/vertex order. The triangle is
    scan-converted in `dst`'s raster space (from `dst_uv`); for every pixel
    it covers, the matching position in `src` is found via that pixel's
    barycentric weights applied to `src_uv`, then bilinear-sampled with
    wrap. This is the actual "sample from the old UV, write to the new UV"
    transfer a UV repack needs - plain vertical stacking can't do this since
    it only ever remaps V uniformly, it can't un-overlap two islands that
    already share the same UV space."""
    dst_h, dst_w = dst.shape[:2]
    xs = dst_uv[:, 0] * dst_w
    ys = dst_uv[:, 1] * dst_h
    min_x = max(0, int(math.floor(xs.min())))
    max_x = min(dst_w - 1, int(math.ceil(xs.max())))
    min_y = max(0, int(math.floor(ys.min())))
    max_y = min(dst_h - 1, int(math.ceil(ys.max())))
    if max_x < min_x or max_y < min_y:
        return
    x0, y0, x1, y1, x2, y2 = xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return
    px = np.arange(min_x, max_x + 1) + 0.5
    py = np.arange(min_y, max_y + 1) + 0.5
    cx, cy = np.meshgrid(px, py)
    a = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / denom
    b = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / denom
    c = 1.0 - a - b
    mask = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
    if not mask.any():
        return
    su = a * src_uv[0, 0] + b * src_uv[1, 0] + c * src_uv[2, 0]
    sv = a * src_uv[0, 1] + b * src_uv[1, 1] + c * src_uv[2, 1]
    sampled = _sample_bilinear_wrapped(src, su, sv)
    region = dst[min_y:max_y + 1, min_x:max_x + 1]
    region[mask] = sampled[mask]


def _flat_placeholder_array(size: int = 64) -> np.ndarray:
    arr = np.zeros((size, size, 4), dtype=np.float32)
    arr[:, :, :3] = 0.5
    arr[:, :, 3] = 1.0
    return arr


def _neutral_normal_array(w: int, h: int) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.float32)
    arr[:, :, 0] = 0.5
    arr[:, :, 1] = 0.5
    arr[:, :, 2] = 1.0
    arr[:, :, 3] = 1.0
    return arr


def _connected_components(bm: bmesh.types.BMesh) -> list:
    """The mesh's disconnected 3D parts, as lists of BMFace - two faces are
    in the same part iff there's a chain of shared edges between them.
    Mirror-pair detection operates on these, not on UV islands: what we're
    asking is "does this 3D chunk of geometry have a mirror twin elsewhere
    in the merged mesh", and disconnected mesh topology is what defines a
    chunk, not however its UVs happen to be laid out."""
    # ensure_lookup_table() only makes bm.verts[i]/bm.faces[i] indexing work;
    # it does NOT refresh the .index attribute on each element - that needs
    # index_update() separately, and _find_mirror_pairs keys its vertex_map
    # by BMVert.index (freshly-added verts from repeated bmesh.from_mesh()
    # calls all read back as .index == 0 otherwise, since it's never been
    # assigned yet).
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    visited: set = set()
    components = []
    for seed in bm.faces:
        if seed.index in visited:
            continue
        stack = [seed]
        visited.add(seed.index)
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        visited.add(lf.index)
                        stack.append(lf)
        components.append(comp)
    return components


def _match_mirror_vertices(coords_a: np.ndarray, coords_b_mirrored: np.ndarray, tol: float):
    """Greedy nearest-neighbour bijective matching between two equal-length
    point sets (closest pairs claimed first, globally, by sorting every
    candidate pair by distance) - returns {b_index: a_index} only if *every*
    point in both sets found a partner within `tol`, None otherwise. Good
    enough for genuine mirror duplicates (matches land at ~0 distance, far
    below any unrelated point), no need for real Hungarian-algorithm rigour."""
    n = len(coords_a)
    if n == 0 or n != len(coords_b_mirrored):
        return None
    dists = np.linalg.norm(coords_a[:, None, :] - coords_b_mirrored[None, :, :], axis=2)
    flat_order = np.argsort(dists, axis=None)
    used_a, used_b, mapping = set(), set(), {}
    for flat in flat_order:
        ai, bi = divmod(int(flat), n)
        if ai in used_a or bi in used_b:
            continue
        if dists[ai, bi] > tol:
            break
        used_a.add(ai)
        used_b.add(bi)
        mapping[bi] = ai
        if len(mapping) == n:
            break
    return mapping if len(mapping) == n else None


_PREPARE_MIRROR_COLOR_TOL = 0.06  # max per-channel mean-colour gap (0..1)
# still trusted as "the same texture" for mirror-pair purposes - loose enough
# to absorb sampling/compression noise, tight enough that a red-toned fog
# light next to a white-toned reverse light (same geometry, deliberately
# different colour) never passes.


def _component_uv_mean_color(comp, uv_layer, img, cache: dict):
    """Mean colour of `img` sampled *only* under this component's own UV
    footprint - deliberately not the whole image's mean. Real AC1-era light
    textures routinely pack a black housing, chrome trim and several
    differently-coloured lenses into one shared file; comparing whole-image
    averages made a red lens and an orange indicator (both diluted by the
    same surrounding black/chrome) read as "close enough", which is exactly
    how a fog light ended up silently merged with an unrelated part sharing
    its rough shape. Sampling only the pixels this component actually reads
    reflects what it really looks like."""
    if img is None or uv_layer is None or not comp:
        return None
    arr = cache.get(img.name)
    if arr is None:
        arr = _image_to_array(img)[:, :, :3]
        cache[img.name] = arr
    h, w = arr.shape[:2]
    us, vs = [], []
    for f in comp:
        for loop in f.loops:
            u, v = loop[uv_layer].uv
            us.append(u % 1.0)
            vs.append(v % 1.0)
    xs = (np.array(us) * w).astype(np.int64) % w
    ys = ((1.0 - np.array(vs)) * h).astype(np.int64) % h
    return arr[ys, xs].mean(axis=0)


def _find_mirror_pairs(components: list, tol: float, face_sources: list = None, uv_layer=None) -> tuple:
    """Compares every pair of connected components for an X=0 mirror match
    *in the mesh's own object space, with no recentring* - two components
    only match if one, X-flipped, actually lands where the other really is.
    That's deliberate: recentring each component to its own local origin
    before comparing (an earlier version of this did) makes the test purely
    about *shape*, so any two same-shaped-but-unrelated parts anywhere on
    the car (two identical simple panels, say) would false-positive as a
    "mirror pair" - car meshes are modelled with X as the left/right axis
    and the origin on the centreline, so comparing raw positions is what
    actually asks "is this genuinely the other side of the same part".

    Geometry alone isn't enough either: a symmetric pair of light housings
    can be genuinely different parts wearing the same shape - e.g. a rear
    fog light (red-toned lens) mirrored against a reverse light (white-toned
    lens), or two unrelated small quads (few enough vertices that shape alone
    barely constrains the match) that just happen to sit at mirrored
    positions. Forcing those to share one atlas patch would make every
    function mask painted on one bleed onto the other. When `face_sources`
    and `uv_layer` are given, a geometric match is only accepted if both
    components also look the same where they actually sample their source
    texture - see _component_uv_mean_color.

    Returns (excluded_faces: set[int], vertex_map: {excluded_vert_idx:
    kept_vert_idx}) - the second component of any matched pair is the one
    excluded; its vertices map onto their mirror partner's for UV copying
    after packing."""
    color_cache: dict = {}
    info = []
    for comp in components:
        verts = list({v for f in comp for v in f.verts})
        coords = np.array([v.co for v in verts], dtype=np.float64)
        base_img = None
        mean_color = None
        if face_sources is not None and comp:
            base_img = face_sources[comp[0].index][0]
            mean_color = _component_uv_mean_color(comp, uv_layer, base_img, color_cache)
        info.append({
            "faces": comp, "verts": verts, "coords": coords,
            "base_img": base_img, "mean_color": mean_color,
        })

    def _same_texture(mean_a, mean_b):
        if face_sources is None:
            return True
        if mean_a is None and mean_b is None:
            return True
        if mean_a is None or mean_b is None:
            return False
        return bool(np.all(np.abs(mean_a - mean_b) <= _PREPARE_MIRROR_COLOR_TOL))

    excluded_faces: set = set()
    vertex_map: dict = {}
    claimed: set = set()
    for i in range(len(info)):
        if i in claimed:
            continue
        for j in range(i + 1, len(info)):
            if j in claimed:
                continue
            a, b = info[i], info[j]
            if len(a["verts"]) != len(b["verts"]) or len(a["faces"]) != len(b["faces"]):
                continue
            if not _same_texture(a["mean_color"], b["mean_color"]):
                continue
            mirrored_b = b["coords"].copy()
            mirrored_b[:, 0] *= -1
            mapping = _match_mirror_vertices(a["coords"], mirrored_b, tol)
            if mapping is None:
                continue
            claimed.add(i)
            claimed.add(j)
            for bi, ai in mapping.items():
                vertex_map[b["verts"][bi].index] = a["verts"][ai].index
            excluded_faces.update(f.index for f in b["faces"])
            break
    return excluded_faces, vertex_map


_LIGHT_KIND_MESH_NAMES = {
    "PLASTIC": "EXT_LIGHTS_PLASTIC_FIXED",
    "GLASS": "EXT_LIGHTS_GLASS_FIXED",
    "CHROME": "EXT_LIGHTS_CHROME_FIXED",
}

_NORMAL_TOGGLE_PROP = {
    "PLASTIC": "ac_export_plastic_normal",
    "GLASS": "ac_export_glass_normal",
    "CHROME": "ac_export_chrome_normal",
}


_PREPARE_UV_MARGIN = 0.015
_PREPARE_MIRROR_TOL = 1e-3
_CLEAN_MESH_POS_NDIGITS = 5  # rounding precision (metres) for "same point"


def _clean_duplicate_faces(obj: bpy.types.Object) -> int:
    """Removes triangles whose 3 corners sit at the same *positions* as
    another triangle already kept, independent of winding or which actual
    vertex objects they use. This is the post-import counterpart of the
    exact-index duplicate check done at import time: "Disable mesh cleaning"
    (on import) gives a coincident duplicate triangle its own private vertex
    copies specifically so it CAN coexist in the same bmesh (import can't
    otherwise represent it at all) - this is what lets it be found and
    removed again afterwards, once it's no longer needed, without having to
    decide at import time. Also drops any vertex left with zero faces as a
    result. Returns the number of faces removed."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    def key_of(face):
        return frozenset(
            (round(v.co.x, _CLEAN_MESH_POS_NDIGITS),
             round(v.co.y, _CLEAN_MESH_POS_NDIGITS),
             round(v.co.z, _CLEAN_MESH_POS_NDIGITS))
            for v in face.verts
        )

    seen: set = set()
    dup_faces = []
    for f in bm.faces:
        key = key_of(f)
        if key in seen:
            dup_faces.append(f)
        else:
            seen.add(key)

    if dup_faces:
        bmesh.ops.delete(bm, geom=dup_faces, context="FACES")
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()
    return len(dup_faces)


class MESH_OT_ac_clean_selected_mesh(Operator):
    """Runs _clean_duplicate_faces on every currently selected mesh object -
    the general-purpose version of 'Clean lights mesh', for anything else
    imported with 'Disable mesh cleaning' on."""
    bl_idname = "mesh.ac_clean_selected_mesh"
    bl_label = "Clean selected mesh"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        targets = [o for o in context.selected_objects if o.type == "MESH"]
        if not targets:
            self.report({"ERROR"}, "Select at least one mesh in the Outliner/viewport.")
            return {"CANCELLED"}
        total, touched = 0, 0
        for obj in targets:
            removed = _clean_duplicate_faces(obj)
            total += removed
            touched += 1 if removed else 0
        self.report({"INFO"}, f"Removed {total} coincident duplicate face(s) across {touched} mesh(es).")
        return {"FINISHED"}


class MESH_OT_ac_prepare_lights(Operator):
    """Merges every object whose name contains "EXT_LIGHTS" into one mesh,
    repacks its UVs into non-overlapping islands (Blender's own Pack Islands,
    run on the whole merged mesh so islands from different source objects are
    considered together), then bakes each face's own original texture -
    sampled at its *original* UV - into a single new EXT_LIGHTS_PLASTIC_FIXED
    x _c.texture, at its *new*, non-overlapping UV position.

    This exists because real AC1-era light meshes routinely reused the same
    patch of texture across several unrelated UV islands (no per-pixel
    function masking back then, so it didn't matter) - painting a function
    mask over one island now would bleed into every other island that used
    to share that same patch. Repacking + baking gives every island its own,
    unshared pixels before any mask gets painted onto them; a straight
    resize-and-stack of whole textures (the previous approach here) can't
    fix an overlap that exists *within* a single source texture."""
    bl_idname = "mesh.ac_prepare_lights"
    bl_label = "Prepare"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        candidates = [
            o for o in context.scene.objects
            if o.type == "MESH" and "EXT_LIGHTS" in o.name.upper()
        ]
        if not candidates:
            self.report({"ERROR"}, "No object with 'EXT_LIGHTS' in its name found.")
            return {"CANCELLED"}

        cleaned_faces = 0
        if context.scene.ac_clean_lights_before_prepare:
            for obj in candidates:
                cleaned_faces += _clean_duplicate_faces(obj)

        old_material_names = {
            obj.data.materials[0].name for obj in candidates if obj.data.materials
        }

        # -- merge every object's geometry, remembering which (base, normal)
        #    image pair each contributed face should sample from at bake time,
        #    and which vertex range came from which source material (-> a
        #    vertex group per material below, so a part stays selectable
        #    after the merge without having to hand-pick faces again) --
        bm_final = bmesh.new()
        face_sources = []  # parallel to the merged mesh's eventual polygon order
        vert_group_ranges: dict[str, list] = {}
        for obj in candidates:
            mat = obj.data.materials[0] if obj.data.materials else None
            group_name = mat.name if mat is not None else obj.name
            base_img = _find_base_color_image(mat) if mat is not None else None
            normal_img = _find_normal_map_image(mat) if mat is not None else None
            before_faces = len(bm_final.faces)
            before_verts = len(bm_final.verts)
            bm_final.from_mesh(obj.data)
            face_sources.extend([(base_img, normal_img)] * (len(bm_final.faces) - before_faces))
            vert_group_ranges.setdefault(group_name, []).append((before_verts, len(bm_final.verts)))

        # -- detect bilateral (X-mirror) duplicate parts (a symmetric car's
        #    left/right light sharing the same texture patch is normal, and
        #    packing both separately would waste half the atlas for nothing)
        #    before the bmesh is thrown away - excluded_faces are left out of
        #    the packing pass entirely; vertex_map lets their UV be copied
        #    from their kept mirror partner afterwards. --
        excluded_faces, mirror_vertex_map = _find_mirror_pairs(
            _connected_components(bm_final), _PREPARE_MIRROR_TOL, face_sources,
            bm_final.loops.layers.uv.active)

        target_name = _LIGHT_KIND_MESH_NAMES["PLASTIC"]
        final_mesh = bpy.data.meshes.new(target_name)
        bm_final.to_mesh(final_mesh)
        bm_final.free()
        for poly in final_mesh.polygons:
            poly.use_smooth = True
        final_mesh.update()

        anchor = candidates[0]
        final_obj = bpy.data.objects.new(target_name, final_mesh)
        context.collection.objects.link(final_obj)
        final_obj.parent = anchor.parent
        src_path = anchor.get(_PROP_SOURCE_PATH)
        lod_index = anchor.get(_PROP_LOD_INDEX)
        if src_path is not None:
            final_obj[_PROP_SOURCE_PATH] = src_path
        if lod_index is not None:
            final_obj[_PROP_LOD_INDEX] = lod_index

        for group_name, ranges in vert_group_ranges.items():
            vg = final_obj.vertex_groups.new(name=group_name)
            for start, end in ranges:
                vg.add(list(range(start, end)), 1.0, "REPLACE")

        # -- duplicate the original UV into a second layer, then repack that
        #    copy so every island (regardless of which source object it came
        #    from) gets its own, non-overlapping patch of the new atlas --
        n_loops = len(final_mesh.loops)
        orig_uv_flat = np.empty(n_loops * 2, dtype=np.float32)
        final_mesh.uv_layers["UVMap"].data.foreach_get("uv", orig_uv_flat)
        packed_layer = final_mesh.uv_layers.new(name="UVMap_packed")
        packed_layer.data.foreach_set("uv", orig_uv_flat)
        final_mesh.uv_layers.active = packed_layer

        for o in context.selected_objects:
            o.select_set(False)
        final_obj.select_set(True)
        context.view_layer.objects.active = final_obj
        bpy.ops.object.mode_set(mode="EDIT")
        bm_edit = bmesh.from_edit_mesh(final_mesh)
        bm_edit.faces.ensure_lookup_table()
        # Excluded (mirror-duplicate) faces are deliberately left unselected
        # so Pack Islands never touches their UV at all - not calling
        # uv.select_all afterwards matters here, it would select every UV in
        # the mesh regardless of this face selection and defeat the exclusion.
        for f in bm_edit.faces:
            f.select = f.index not in excluded_faces
        bmesh.update_edit_mesh(final_mesh)
        bpy.ops.uv.pack_islands(margin=_PREPARE_UV_MARGIN)
        bpy.ops.object.mode_set(mode="OBJECT")

        packed_uv_flat = np.empty(n_loops * 2, dtype=np.float32)
        final_mesh.uv_layers["UVMap_packed"].data.foreach_get("uv", packed_uv_flat)
        orig_uv = orig_uv_flat.reshape(n_loops, 2)
        packed_uv = packed_uv_flat.reshape(n_loops, 2)

        if mirror_vertex_map:
            # Excluded faces were never packed - their loops still hold
            # whatever the *original* UV was (copied verbatim into the
            # packed layer up front). Give each one its mirror partner's
            # actual packed position instead, so both halves end up
            # sampling the exact same atlas patch.
            vert_to_packed_uv: dict = {}
            for poly in final_mesh.polygons:
                if poly.index in excluded_faces:
                    continue
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    vi = final_mesh.loops[li].vertex_index
                    vert_to_packed_uv.setdefault(vi, packed_uv[li])
            for poly in final_mesh.polygons:
                if poly.index not in excluded_faces:
                    continue
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    vi = final_mesh.loops[li].vertex_index
                    partner_vi = mirror_vertex_map.get(vi)
                    if partner_vi is None:
                        continue
                    partner_uv = vert_to_packed_uv.get(partner_vi)
                    if partner_uv is not None:
                        packed_uv[li] = partner_uv
            # Re-fetch rather than reuse the `packed_layer` reference taken
            # before entering/exiting Edit mode - it's stale by this point
            # and writing through it segfaults (mesh sub-data references
            # don't survive an edit-mode round trip).
            final_mesh.uv_layers["UVMap_packed"].data.foreach_set("uv", packed_uv.reshape(-1))

        # -- bake: for every triangle, sample its source texture at the
        #    *original* UV and write it at the *packed* UV's raster position --
        size = int(context.scene.ac_prepare_atlas_size)
        diffuse_canvas = _flat_placeholder_array(size)
        normal_canvas = _neutral_normal_array(size, size)
        image_array_cache: dict = {}
        placeholder_base = _flat_placeholder_array(1)
        placeholder_normal = _neutral_normal_array(1, 1)
        any_normal = any(n is not None for _b, n in face_sources)

        def array_for(img, placeholder):
            if img is None:
                return placeholder
            arr = image_array_cache.get(img.name)
            if arr is None:
                arr = _image_to_array(img)
                image_array_cache[img.name] = arr
            return arr

        for poly, (base_img, normal_img) in zip(final_mesh.polygons, face_sources):
            loop_idx = list(range(poly.loop_start, poly.loop_start + poly.loop_total))
            if len(loop_idx) < 3:
                continue
            base_arr = array_for(base_img, placeholder_base)
            normal_arr = array_for(normal_img, placeholder_normal) if any_normal else None
            for i in range(1, len(loop_idx) - 1):
                tri = [loop_idx[0], loop_idx[i], loop_idx[i + 1]]
                dst_tri = packed_uv[tri]
                src_tri = orig_uv[tri]
                _bake_triangle_into(diffuse_canvas, dst_tri, base_arr, src_tri)
                if any_normal:
                    _bake_triangle_into(normal_canvas, dst_tri, normal_arr, src_tri)

        atlas_base_img = _array_to_image(f"{target_name}_c", diffuse_canvas, "sRGB")
        atlas_normal_img = _array_to_image(f"{target_name}_nm", normal_canvas, "Non-Color") if any_normal else None

        # -- auto-populate the Texture Paint tab's Reference/Normal map
        #    fields from the atlas that was just built, so painting F1/F2
        #    right after Prepare bakes against the new, non-overlapping
        #    texture instead of a stale or manually-picked one --
        ref_bw = _make_bw_copy(atlas_base_img)
        context.scene.ac_lights_ref_image = ref_bw
        generated_nm = None
        try:
            generated_nm = _generate_normal_map_from_ref(ref_bw)
            context.scene.ac_normal_map_image = generated_nm
        except Exception as exc:  # noqa: BLE001
            self.report({"WARNING"}, f"Auto normal map generation failed: {exc}")

        # -- the packed layer is the one that matters from here on; drop the
        #    original (overlapping) one and put the packed data under its name --
        final_mesh.uv_layers.remove(final_mesh.uv_layers["UVMap"])
        final_mesh.uv_layers["UVMap_packed"].name = "UVMap"
        final_mesh.uv_layers.active = final_mesh.uv_layers["UVMap"]

        new_mat = bpy.data.materials.new(target_name)
        new_mat.use_nodes = True
        final_mesh.materials.append(new_mat)
        nodes, links = new_mat.node_tree.nodes, new_mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        if bsdf is not None:
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.image = atlas_base_img
            tex_node.label = "AC Diffuse (atlas)"
            tex_node.location = (-400, 300)
            links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            if atlas_normal_img is not None:
                n_tex = nodes.new("ShaderNodeTexImage")
                n_tex.image = atlas_normal_img
                n_tex.label = "AC Normal (atlas)"
                n_tex.location = (-700, 0)
                nm_node = nodes.new("ShaderNodeNormalMap")
                nm_node.location = (-400, 0)
                links.new(n_tex.outputs["Color"], nm_node.inputs["Color"])
                links.new(nm_node.outputs["Normal"], bsdf.inputs["Normal"])

        for obj in candidates:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Removing an object doesn't drop its mesh datablock's own material
        # reference (the mesh just becomes orphaned) - a recursive purge is
        # needed to actually cascade the old materials down to 0 users.
        bpy.data.orphans_purge(do_recursive=True)
        removed_materials = sorted(n for n in old_material_names if bpy.data.materials.get(n) is None)

        self.report(
            {"INFO"},
            f"Prepare: merged {len(candidates)} object(s), repacked and rebaked into "
            f"'{target_name}' ({size}x{size} atlas), {len(removed_materials)} unused material(s) removed"
            + (f", {len(excluded_faces)} mirror-duplicate face(s) sharing texture with their twin." if excluded_faces else ".")
            + (f" Reference '{ref_bw.name}' and normal map '{generated_nm.name}' auto-generated." if generated_nm else "")
            + (f" {cleaned_faces} coincident duplicate face(s) cleaned first." if cleaned_faces else ""),
        )
        return {"FINISHED"}


class MESH_OT_ac_split_light_kind(Operator):
    """Edit-mode only: cuts the selected faces (across every AC mesh
    currently in Edit mode) out of their current mesh and moves them into the
    single canonical EXT_LIGHTS_<kind>_FIXED mesh - created fresh (with its
    own <kind>-named material, carrying over the source's Base Color texture)
    the first time this kind is used, or simply appended into on every call
    after that. The source mesh(es) keep whatever geometry wasn't selected."""
    bl_idname = "mesh.ac_split_light_kind"
    bl_label = "Split by kind"
    bl_options = {"REGISTER", "UNDO"}

    kind: EnumProperty(items=[
        ("PLASTIC", "Is plastic", ""),
        ("GLASS", "Is glass", ""),
        ("CHROME", "Is chrome", ""),
    ])

    @classmethod
    def poll(cls, context):
        return bool(_meshes_in_edit_mode(context))

    def execute(self, context):
        target_name = _LIGHT_KIND_MESH_NAMES[self.kind]
        extracted_meshes = []
        source_mat = None
        source_obj = None

        for obj in _meshes_in_edit_mode(context):
            bm_src = bmesh.from_edit_mesh(obj.data)
            bm_src.faces.ensure_lookup_table()
            faces = [f for f in bm_src.faces if f.select]
            if not faces:
                continue
            selected_indices = {f.index for f in faces}

            if source_obj is None:
                source_obj = obj
                mats = obj.data.materials
                if mats:
                    source_mat = mats[min(faces[0].material_index, len(mats) - 1)]

            # bmesh.ops.duplicate's `dest` isn't implemented in this Blender
            # build ("keyword dest type 4 not working yet"), so cross-bmesh
            # extraction goes through a full Mesh datablock copy instead -
            # from_mesh()/to_mesh() already round-trips every custom layer
            # correctly (same mechanism MESH_OT_ac_prepare_lights relies on).
            # Mesh.polygons[].select is stale while the object is still in
            # Edit mode (it reflects edit-mode-entry state, not the live
            # BMesh selection) - matching by face *index* instead of trusting
            # the copy's own select flags is what actually works here.
            bmesh.update_edit_mesh(obj.data)
            tmp_mesh = obj.data.copy()
            bm_piece = bmesh.new()
            bm_piece.from_mesh(tmp_mesh)
            bm_piece.faces.ensure_lookup_table()
            unselected = [f for f in bm_piece.faces if f.index not in selected_indices]
            bmesh.ops.delete(bm_piece, geom=unselected, context="FACES")
            loose = [v for v in bm_piece.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm_piece, geom=loose, context="VERTS")
            bm_piece.to_mesh(tmp_mesh)
            bm_piece.free()
            extracted_meshes.append(tmp_mesh)

            bmesh.ops.delete(bm_src, geom=faces, context="FACES")
            loose_verts = [v for v in bm_src.verts if not v.link_faces]
            if loose_verts:
                bmesh.ops.delete(bm_src, geom=loose_verts, context="VERTS")
            bmesh.update_edit_mesh(obj.data)

        if not extracted_meshes:
            self.report({"ERROR"}, "No faces selected in any mesh currently in Edit mode.")
            return {"CANCELLED"}

        target = bpy.data.objects.get(target_name)
        created = target is None
        if target is None:
            mesh = bpy.data.meshes.new(target_name)
            bm_target = bmesh.new()
            for tm in extracted_meshes:
                bm_target.from_mesh(tm)
            bm_target.to_mesh(mesh)
            bm_target.free()
            for poly in mesh.polygons:
                poly.use_smooth = True

            target = bpy.data.objects.new(target_name, mesh)
            context.collection.objects.link(target)
            target.parent = source_obj.parent
            src_path = source_obj.get(_PROP_SOURCE_PATH)
            lod_index = source_obj.get(_PROP_LOD_INDEX)
            if src_path is not None:
                target[_PROP_SOURCE_PATH] = src_path
            if lod_index is not None:
                target[_PROP_LOD_INDEX] = lod_index

            new_mat = bpy.data.materials.new(target_name)
            new_mat.use_nodes = True
            mesh.materials.append(new_mat)
            base_img = _find_base_color_image(source_mat) if source_mat is not None else None
            if base_img is not None:
                bsdf = new_mat.node_tree.nodes.get("Principled BSDF")
                if bsdf is not None:
                    tex_node = new_mat.node_tree.nodes.new("ShaderNodeTexImage")
                    tex_node.image = base_img
                    tex_node.label = "AC Diffuse"
                    tex_node.location = (-400, 300)
                    new_mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            bm_target = bmesh.new()
            bm_target.from_mesh(target.data)
            for tm in extracted_meshes:
                bm_target.from_mesh(tm)
            bm_target.to_mesh(target.data)
            bm_target.free()
            for poly in target.data.polygons:
                poly.use_smooth = True
            target.data.update()

        for tm in extracted_meshes:
            bpy.data.meshes.remove(tm)
        bpy.data.orphans_purge(do_recursive=True)

        self.report(
            {"INFO"},
            f"{'Created' if created else 'Appended into'} '{target_name}' "
            f"({len(extracted_meshes)} source mesh(es) contributed geometry).",
        )
        return {"FINISHED"}


class MESH_OT_ac_ref_from_material(Operator):
    """Picks up the texture wired to the active material's Base Color, makes a
    black & white copy of it and sets that as the reference - the usual case,
    since the optic's own colour map is exactly the shading that keeps the
    light from reading flat in-game."""
    bl_idname = "mesh.ac_ref_from_material"
    bl_label = "Use actual"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh.")
            return {"CANCELLED"}
        mat = obj.active_material
        if mat is None:
            self.report({"ERROR"}, f"'{obj.name}' has no active material.")
            return {"CANCELLED"}

        image = _find_base_color_image(mat)
        if image is None:
            self.report(
                {"ERROR"},
                f"No texture wired to the Base Color of '{mat.name}'. Re-import the mesh with "
                "'Link textures' on, or wire an Image Texture into Base Color.",
            )
            return {"CANCELLED"}
        if image.size[0] == 0 or image.size[1] == 0:
            self.report({"ERROR"}, f"Image '{image.name}' has no loaded pixels.")
            return {"CANCELLED"}

        bw = _make_bw_copy(image)
        context.scene.ac_lights_ref_image = bw
        self.report(
            {"INFO"},
            f"Reference = '{bw.name}' ({bw.size[0]}x{bw.size[1]}), B&W copy of '{image.name}' "
            f"(Base Color of '{mat.name}').",
        )
        return {"FINISHED"}


class MESH_OT_ac_paint_light_texture(Operator):
    """Paints one RGB channel of EXT_Lights_F_1/F_2 (sdk.pdf p.14-15) at the
    UV footprint of the currently selected faces, leaving every other pixel
    and channel untouched."""
    bl_idname = "mesh.ac_paint_light_texture"
    bl_label = "Paint AC Light Function Texture"
    bl_options = {"REGISTER", "UNDO"}

    texture_slot: EnumProperty(items=[("F1", "EXT_Lights_F_1", ""), ("F2", "EXT_Lights_F_2", "")])
    channel: EnumProperty(items=[("R", "R", ""), ("G", "G", ""), ("B", "B", "")])
    value: FloatProperty(default=1.0)

    def execute(self, context):
        objects = _meshes_in_edit_mode(context)
        if not objects:
            self.report({"ERROR"}, "Select a mesh and switch to Edit mode.")
            return {"CANCELLED"}
        image = context.scene.ac_lights_f1_image if self.texture_slot == "F1" else context.scene.ac_lights_f2_image
        if image is None:
            self.report({"ERROR"}, f"Pick an image for {self.texture_slot} first.")
            return {"CANCELLED"}
        groups = _selected_face_groups(objects)
        if not groups:
            self.report({"WARNING"}, "No faces selected (or no active UV map).")
            return {"CANCELLED"}
        span_u, span_v = _selection_uv_tile_span(groups)
        _push_paint_history(image)
        painted = _paint_faces_to_image(image, groups, [(_CHANNEL_INDEX[self.channel], self.value)])
        self.report(
            {"INFO"},
            f"{painted} pixel(s) painted on {self.texture_slot} ({self.channel}={self.value:g}) "
            f"from {len(groups)} mesh(es).",
        )
        if span_u > 1.05 or span_v > 1.05:
            self.report(
                {"INFO"},
                f"Note: UVs span {span_u:.1f} x {span_v:.1f} tiles (deliberate texture tiling) - the paint "
                "therefore wraps across the image and is shared by every area landing on the same spot "
                "within the tile.",
            )
        return {"FINISHED"}


class MESH_OT_ac_erase_light_texture(Operator):
    """Resets EXT_Lights_F_1/F_2 (all 3 channels) to black - either just the
    UV footprint of the current selection, or (with a confirmation prompt)
    the whole image."""
    bl_idname = "mesh.ac_erase_light_texture"
    bl_label = "Erase AC Light Function Texture"
    bl_options = {"REGISTER", "UNDO"}

    texture_slot: EnumProperty(items=[("F1", "EXT_Lights_F_1", ""), ("F2", "EXT_Lights_F_2", "")])
    whole_image: BoolProperty(default=False)

    def invoke(self, context, event):
        if self.whole_image:
            return context.window_manager.invoke_confirm(self, event)
        return self.execute(context)

    def execute(self, context):
        image = context.scene.ac_lights_f1_image if self.texture_slot == "F1" else context.scene.ac_lights_f2_image
        if image is None:
            self.report({"ERROR"}, f"Pick an image for {self.texture_slot} first.")
            return {"CANCELLED"}
        if self.whole_image:
            _push_paint_history(image)
            _fill_image(image, (0.0, 0.0, 0.0))
            self.report({"INFO"}, f"{self.texture_slot} fully reset to black.")
            return {"FINISHED"}
        objects = _meshes_in_edit_mode(context)
        if not objects:
            self.report({"ERROR"}, "Select a mesh and switch to Edit mode.")
            return {"CANCELLED"}
        groups = _selected_face_groups(objects)
        if not groups:
            self.report({"WARNING"}, "No faces selected (or no active UV map).")
            return {"CANCELLED"}
        _push_paint_history(image)
        painted = _paint_faces_to_image(image, groups, [(0, 0.0), (1, 0.0), (2, 0.0)])
        self.report(
            {"INFO"},
            f"{painted} pixel(s) reset to black on {self.texture_slot} from {len(groups)} mesh(es).",
        )
        return {"FINISHED"}


class MESH_OT_ac_undo_paint(Operator):
    """Steps back through this addon's own paint history. Needed because
    Blender's undo does not restore image pixels written from a script (only
    the operator call itself is undone, the pixels stay as painted)."""
    bl_idname = "mesh.ac_undo_paint"
    bl_label = "Undo last paint"
    bl_options = {"REGISTER"}

    texture_slot: EnumProperty(items=[("F1", "EXT_Lights_F_1", ""), ("F2", "EXT_Lights_F_2", "")])

    def execute(self, context):
        image = context.scene.ac_lights_f1_image if self.texture_slot == "F1" else context.scene.ac_lights_f2_image
        if image is None:
            self.report({"ERROR"}, f"No image assigned to slot {self.texture_slot}.")
            return {"CANCELLED"}
        if not _pop_paint_history(image):
            self.report({"WARNING"}, f"Nothing left to undo on {self.texture_slot}.")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Undone on {self.texture_slot} - {_paint_history_depth(image)} step(s) left.",
        )
        return {"FINISHED"}


class MESH_OT_ac_bake_lights(Operator):
    """Bakes the painted function texture together with the black & white
    reference into a final, export-ready image: final = painted x luma(ref).
    Multiplying keeps each channel's ratio intact (a pixel painted 77,0,255
    stays proportionally 77:0:255), so the optic's shading comes through
    without muddying the function colours. The painted source image is left
    untouched so it stays editable."""
    bl_idname = "mesh.ac_bake_lights"
    bl_label = "Bake with reference"
    bl_options = {"REGISTER", "UNDO"}

    texture_slot: EnumProperty(items=[("F1", "EXT_Lights_F_1", ""), ("F2", "EXT_Lights_F_2", "")])

    def execute(self, context):
        scene = context.scene
        painted = scene.ac_lights_f1_image if self.texture_slot == "F1" else scene.ac_lights_f2_image
        ref = scene.ac_lights_ref_image
        if painted is None:
            self.report({"ERROR"}, f"No painted image for {self.texture_slot}.")
            return {"CANCELLED"}
        if ref is None or ref.size[0] == 0:
            self.report({"ERROR"}, "Pick a reference image first ('Use actual').")
            return {"CANCELLED"}
        w, h = painted.size
        if w == 0 or h == 0:
            self.report({"ERROR"}, "The painted image is empty.")
            return {"CANCELLED"}

        lum = _resample_nearest(_luminance_of(ref), w, h)
        out = _image_to_array(painted).copy()
        out[:, :, 0] *= lum
        out[:, :, 1] *= lum
        out[:, :, 2] *= lum
        out[:, :, 3] = 1.0

        name = f"{painted.name}_BAKED"
        target = bpy.data.images.get(name)
        if target is None or tuple(target.size) != (w, h):
            target = bpy.data.images.new(name, width=w, height=h, alpha=True)
        target.pixels = out.reshape(-1).tolist()
        target.update()
        self.report(
            {"INFO"},
            f"'{target.name}' ({w}x{h}) generated from '{painted.name}' x '{ref.name}' - "
            "the painted image is left untouched.",
        )
        return {"FINISHED"}


def _car_texture_folder():
    """The car's texture folder, sitting next to meshes/ - real content uses
    "texture" (singular) but "textures" turns up too, so both are accepted.
    Located from any AC-imported object's own source path."""
    for obj in bpy.data.objects:
        src = obj.get(_PROP_SOURCE_PATH)
        if not src:
            continue
        car_root = os.path.dirname(os.path.dirname(src))
        for name in ("texture", "textures"):
            candidate = os.path.join(car_root, name)
            if os.path.isdir(candidate):
                return candidate
    return None


def _image_existing_path(image):
    """The engine-relative path this image was imported from, if any - absent
    for images created fresh in Blender (a Prepare atlas, a new mask, ...),
    which is exactly the signal exporters use to decide a texture needs
    writing out under a new name rather than reusing an existing file."""
    if image is None:
        return None
    path = image.get(_PROP_TEXTURE_PATH)
    return path or None


def _mask_output_name(context, texture_slot: str, painted) -> str:
    """File base name for a light mask.

    Two cases, deliberately different:
      - the mask came from the material -> keep its original name, so an
        existing texture shared by other materials keeps working and the
        material's FunctionMask entries need no edit at all;
      - it was created here -> name it after the optic reference with an
        _f1/_f2 suffix, and the material gets pointed at it.
    """
    existing = _image_existing_path(painted)
    if existing:
        return os.path.splitext(os.path.basename(existing.replace("/", "\\")))[0]
    suffix = "_f1" if texture_slot == "F1" else "_f2"
    ref = context.scene.ac_lights_ref_image
    if ref is not None:
        return f"{_clean_texture_basename(ref.name)}{suffix}"
    return "EXT_Lights_F_1" if texture_slot == "F1" else "EXT_Lights_F_2"


class MESH_OT_ac_save_ac_texture(Operator):
    """Writes the baked image straight out as the game's own .texture +
    .texturemips pair into the car's texture folder.

    Stored uncompressed RGBA8, single mip - deliberately NOT block-compressed.
    A function mask's channel value *is* the data (it selects a light function
    and drives its intensity), so BC1's 5:6:5 quantisation visibly wrecks it.
    Real content agrees: in shipped cars EXT_Lights_F_1/F_2 are uncompressed
    single-mip while the colour/AO maps next to them are BC7 with full mip
    chains."""
    bl_idname = "mesh.ac_save_ac_texture"
    bl_label = "Save as .texture"
    bl_options = {"REGISTER"}

    texture_slot: EnumProperty(items=[("F1", "EXT_Lights_F_1", ""), ("F2", "EXT_Lights_F_2", "")])

    def _resolve(self, context):
        painted = (context.scene.ac_lights_f1_image if self.texture_slot == "F1"
                   else context.scene.ac_lights_f2_image)
        image = bpy.data.images.get(f"{painted.name}_BAKED") if painted is not None else None
        return _mask_output_name(context, self.texture_slot, painted), painted, image

    def invoke(self, context, event):
        base, painted, image = self._resolve(context)
        folder = _car_texture_folder()
        if painted is None:
            self.report({"ERROR"}, f"No image assigned to slot {self.texture_slot}.")
            return {"CANCELLED"}
        if image is None:
            self.report({"ERROR"}, f"No baked image '{painted.name}_BAKED' - run Bake first.")
            return {"CANCELLED"}
        if folder is None:
            self.report({"ERROR"}, "No texture/ folder found next to the car's meshes/ folder.")
            return {"CANCELLED"}
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        base, painted, image = self._resolve(context)
        folder = _car_texture_folder()
        if image is None or folder is None:
            self.report({"ERROR"}, "Bake the texture first, and make sure a car is imported.")
            return {"CANCELLED"}
        w, h = image.size
        if w == 0 or h == 0:
            self.report({"ERROR"}, "The baked image is empty.")
            return {"CANCELLED"}

        px = _image_to_array(image)
        # Blender's pixel buffer starts at the BOTTOM row; .texturemips stores
        # the top row first, so the image has to be flipped on the way out.
        # Channel order is R,G,B,A - confirmed against a reference file written
        # by another tool, whose stored bytes peak at (179,179,0) exactly like
        # the source image's R,G,B (BGRA would have given (0,179,179)).
        rgba = np.clip(px[::-1] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 3] = 255

        try:
            if context.scene.ac_mask_export_format == "RGBA8":
                texture_bytes, mips_bytes = ace_texture.image_to_texture(rgba.tobytes(), w, h)
            else:
                texture_bytes, mips_bytes = ace_texture.rgba_to_bc7_texture(rgba, w, h, srgb=True)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Texture encoding failed: {exc}")
            return {"CANCELLED"}

        tex_path = os.path.join(folder, f"{base}.texture")
        mip_path = os.path.join(folder, f"{base}.texturemips")
        try:
            with open(tex_path, "wb") as fh:
                fh.write(texture_bytes)
            with open(mip_path, "wb") as fh:
                fh.write(mips_bytes)
        except OSError as exc:
            self.report({"ERROR"}, f"Could not write to {folder}: {exc}")
            return {"CANCELLED"}

        meta = ace_texture.decode_metadata(texture_bytes)
        fmt = ace_texture.FORMATS_BY_KUNOS.get(meta.kunos_format)
        self.report(
            {"INFO"},
            f"Wrote {base}.texture + .texturemips to {folder} "
            f"({w}x{h}, {meta.mipcount} mip, {fmt.name if fmt else meta.kunos_format}, "
            f"{len(mips_bytes) // 1024} KiB).",
        )
        return {"FINISHED"}


# --------------------------------------------------------------------------
# .material authoring
#
# Blend mode is stored TWICE and both have to agree, see
# champ_cache_blendmode.pdf: the named "blendMode" property drives the
# shader's blend equation, while an undocumented root varint field 2 drives
# the render pass (opaque vs transparent queue). Real content is perfectly
# consistent: opaque -> property 0.0 and field 2 absent entirely; transparent
# -> property 1.0 and field 2 == 2. Writing only one of the two leaves the
# object classified transparent by the pipeline but shaded opaque (or the
# reverse), which shows up as random-looking colour/light dropouts.
_MATERIAL_BLEND_ROOT_FIELD = 2
_MATERIAL_BLEND_ROOT_TRANSPARENT = 2

# Values written when the normal map is wired in (see the "add/replace normal
# map" checkbox). A property left at KIND_UNSET means "not set" in this
# format, so enabling one means giving it a real scalar value.
_NORMAL_MAP_PROPS = {
    "Base_HasNormalMap": 1.0,
    "Base_NormalScale": 0.5,
    # Same name as the texture slot below but a distinct item in the file -
    # the slot carries the path, this property is the enable flag.
    "Base_NormalMap": 1.0,
}
_NORMAL_MAP_SLOT = "Base_NormalMap"
_BASECOLOR_SLOT = "Base_BaseColorMap"


def _reference_texture_path(context, car_prefix: str):
    """Engine path of the optic reference texture, used as the base colour
    when a material has no existing file to carry one over from."""
    ref = context.scene.ac_lights_ref_image
    if ref is None or not car_prefix:
        return None
    return f"{car_prefix}\\texture\\{_clean_texture_basename(ref.name)}.texture"


def _bundled_refmat_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "refmat.material")


_LIGHT_KIND_PRESET_FILES = {
    "PLASTIC": "refmat_plastic.material",
    "GLASS": "refmat_glass.material",
    "CHROME": "refmat_chrome.material",
}


def _bundled_kind_preset_path(kind: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), _LIGHT_KIND_PRESET_FILES[kind])


def _car_root_from_scene():
    for obj in bpy.data.objects:
        src = obj.get(_PROP_SOURCE_PATH)
        if src:
            return os.path.dirname(os.path.dirname(src))
    return None


def _car_content_prefix(car_root: str):
    """Engine-relative "content\\cars\\<car>" prefix for a car folder on disk."""
    parts = os.path.normpath(car_root).split(os.sep)
    lowered = [p.lower() for p in parts]
    if "content" in lowered:
        start = len(lowered) - 1 - lowered[::-1].index("content")
        return "\\".join(parts[start:])
    return None


def _retarget_texture_path(path, car_prefix: str):
    """Repoints a "content\\cars\\<other car>\\..." path at the current car.
    Shared content (common_assets) and anything not under content\\cars is
    left exactly as-is - those are meant to be referenced across cars."""
    if not path or not car_prefix:
        return path
    parts = path.replace("/", "\\").split("\\")
    lowered = [p.lower() for p in parts]
    if len(parts) >= 4 and lowered[0] == "content" and lowered[1] == "cars":
        if lowered[2] == "common_assets":
            return path
        return "\\".join(car_prefix.split("\\") + parts[3:])
    return path


def _set_material_blend(mf, transparent: bool) -> None:
    for prop in mf.properties:
        if prop.name == "blendMode":
            prop.kind = material_codec.KIND_SCALAR
            prop.components = {1: 1.0 if transparent else 0.0}
            break
    mf.items = [
        it for it in mf.items
        if not (isinstance(it, material_codec.RawField) and it.field_no == _MATERIAL_BLEND_ROOT_FIELD)
    ]
    if transparent:
        # Real files carry it immediately after the shader name, before the
        # property list - keep the same ordering.
        mf.items.insert(0, material_codec.RawField(
            _MATERIAL_BLEND_ROOT_FIELD, "varint", _MATERIAL_BLEND_ROOT_TRANSPARENT))


def _selected_material_names(context) -> list:
    """Every distinct material on the selected meshes - a multi-object
    selection with different materials must have all of them processed."""
    names, seen = [], set()
    objects = list(context.selected_objects) or (
        [context.active_object] if context.active_object else [])
    for obj in objects:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if mat is not None and mat.name not in seen:
                seen.add(mat.name)
                names.append(mat.name)
    return names


class MESH_OT_ac_tag_for_export(Operator):
    """Stamps the selected mesh object(s) with the same source .mesh file and
    LOD index as whatever's already tagged elsewhere in the scene, so a
    newly added object (built from scratch, duplicated and detached, ...)
    gets picked up by Export without having to be joined into an existing
    one first - Export only ever looks at objects carrying these two
    properties, never at what's simply present or selected in the scene."""
    bl_idname = "mesh.ac_tag_for_export"
    bl_label = "Tag for export"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source_path, lod_index = None, None
        active = context.active_object
        if active is not None and _PROP_SOURCE_PATH in active and _PROP_LOD_INDEX in active:
            source_path = active[_PROP_SOURCE_PATH]
            lod_index = int(active[_PROP_LOD_INDEX])
        else:
            for obj in bpy.data.objects:
                if obj.type == "MESH" and _PROP_SOURCE_PATH in obj and _PROP_LOD_INDEX in obj:
                    source_path = obj[_PROP_SOURCE_PATH]
                    lod_index = int(obj[_PROP_LOD_INDEX])
                    break
        if source_path is None:
            self.report({"ERROR"}, "No AC-imported mesh in the scene to copy the source/LOD from.")
            return {"CANCELLED"}

        targets = [o for o in context.selected_objects if o.type == "MESH"]
        if not targets:
            self.report({"ERROR"}, "Select at least one mesh.")
            return {"CANCELLED"}

        for obj in targets:
            obj[_PROP_SOURCE_PATH] = source_path
            obj[_PROP_LOD_INDEX] = lod_index

        self.report(
            {"INFO"},
            f"{len(targets)} object(s) tagged for export (LOD {lod_index}, {os.path.basename(source_path)}).",
        )
        return {"FINISHED"}


class MESH_OT_ac_open_actor(Operator):
    """Opens this car's .actor file in the bundled Lite Editor - a standalone
    tkinter tool (not part of Blender) for directly editing .actor /
    .carlightingsystem files. Launched as a separate OS process against the
    system's own Python (matching lite_editor.bat's own launch method), since
    Blender's bundled Python doesn't reliably ship tkinter."""
    bl_idname = "mesh.ac_open_actor"
    bl_label = "Open .actor"
    bl_options = {"REGISTER"}

    def execute(self, context):
        car_root = _car_root_from_scene()
        if car_root is None:
            self.report({"ERROR"}, "No AC-imported mesh in the scene.")
            return {"CANCELLED"}
        try:
            actor_names = sorted(f for f in os.listdir(car_root) if f.lower().endswith(".actor"))
        except OSError as exc:
            self.report({"ERROR"}, f"Could not list {car_root}: {exc}")
            return {"CANCELLED"}
        if not actor_names:
            self.report({"ERROR"}, f"No .actor file found in {car_root}.")
            return {"CANCELLED"}
        actor_path = os.path.join(car_root, actor_names[0])

        bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lite_editor", "lite_editor.bat")
        if not os.path.isfile(bat_path):
            self.report({"ERROR"}, f"Lite Editor not found: {bat_path}")
            return {"CANCELLED"}

        try:
            subprocess.Popen([bat_path, actor_path], shell=True)
        except OSError as exc:
            self.report({"ERROR"}, f"Could not launch the Lite Editor: {exc}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Opening {actor_names[0]} in the Lite Editor.")
        return {"FINISHED"}


class MESH_OT_ac_apply_ref_material(Operator):
    """Rewrites the .material file of every selected object's material, either
    patching only the two light function-mask texture slots or replacing the
    whole thing with the bundled reference material. Existing files are backed
    up to <name>.material.bak before being overwritten."""
    bl_idname = "mesh.ac_apply_ref_material"
    bl_label = "Apply reference material"
    bl_options = {"REGISTER"}

    mode: EnumProperty(items=[
        ("TEXTURE_ONLY", "Texture only", "Only repoint FunctionMask1/2 at this car's light masks"),
        ("WHOLE", "Whole material", "Replace the whole material with the reference one"),
    ])

    def invoke(self, context, event):
        if not _selected_material_names(context):
            self.report({"ERROR"}, "Select at least one mesh with a material.")
            return {"CANCELLED"}
        if _car_root_from_scene() is None:
            self.report({"ERROR"}, "No AC-imported mesh in the scene - cannot locate the car folder.")
            return {"CANCELLED"}
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        car_root = _car_root_from_scene()
        if car_root is None:
            self.report({"ERROR"}, "No AC-imported mesh in the scene.")
            return {"CANCELLED"}
        car_prefix = _car_content_prefix(car_root)
        if car_prefix is None:
            self.report({"ERROR"}, f"Could not derive a content\\cars\\... prefix from {car_root}.")
            return {"CANCELLED"}
        materials_dir = os.path.join(car_root, "materials")
        if not os.path.isdir(materials_dir):
            self.report({"ERROR"}, f"No materials/ folder in {car_root}.")
            return {"CANCELLED"}

        names = _selected_material_names(context)
        if not names:
            self.report({"ERROR"}, "Select at least one mesh with a material.")
            return {"CANCELLED"}

        scene = context.scene
        transparent = scene.ac_material_transparent

        # Only repoint a FunctionMask slot when the mask was created here. If
        # it was imported from the material it keeps its own name and path, so
        # touching the slot would only risk breaking a texture that other
        # materials may share.
        mask_paths = {}
        for slot, prop_name in _MASK_SLOT_TO_PROP.items():
            image = getattr(scene, prop_name)
            if image is None or _image_existing_path(image):
                continue
            texture_slot = "F1" if slot == "FunctionMask1" else "F2"
            base = _mask_output_name(context, texture_slot, image)
            mask_paths[slot] = f"{car_prefix}\\texture\\{base}.texture"

        # Optional: write the generated normal map next to the masks and wire
        # it into every material we touch.
        normal_path = None
        if scene.ac_apply_normal_map:
            nm_image = scene.ac_normal_map_image
            if nm_image is None or nm_image.size[0] == 0:
                self.report({"ERROR"}, "'Add/replace normal map' is on but no normal map was generated.")
                return {"CANCELLED"}
            tex_folder = _car_texture_folder()
            if tex_folder is None:
                self.report({"ERROR"}, "No texture/ folder found next to meshes/.")
                return {"CANCELLED"}
            nm_base = _clean_texture_basename(nm_image.name)
            nw, nh = nm_image.size
            # Normal maps go out as BC6H_UF16 with a full mip chain: that is
            # what every normal map in shipped car content uses (73/73).
            rgb = _image_to_array(nm_image)[::-1, :, :3]
            try:
                texture_bytes, mips_bytes = ace_texture.rgb_to_bc6h_texture(rgb, nw, nh)
                with open(os.path.join(tex_folder, f"{nm_base}.texture"), "wb") as fh:
                    fh.write(texture_bytes)
                with open(os.path.join(tex_folder, f"{nm_base}.texturemips"), "wb") as fh:
                    fh.write(mips_bytes)
            except OSError as exc:
                self.report({"ERROR"}, f"Could not write the normal map: {exc}")
                return {"CANCELLED"}
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"BC6H encoding failed: {exc}")
                return {"CANCELLED"}
            nmeta = ace_texture.decode_metadata(texture_bytes)
            normal_path = f"{car_prefix}\\texture\\{nm_base}.texture"
            self.report(
                {"INFO"},
                f"Wrote {nm_base}.texture ({nw}x{nh}, {nmeta.mipcount} mips, BC6H_UF16, "
                f"{len(mips_bytes) // 1024} KiB).",
            )

        if self.mode == "WHOLE":
            ref_path = _bundled_refmat_path()
            if not os.path.isfile(ref_path):
                self.report({"ERROR"}, f"Reference material not found: {ref_path}")
                return {"CANCELLED"}
            try:
                material_codec.decode_material(ref_path)
            except Exception as exc:  # noqa: BLE001
                self.report({"ERROR"}, f"Could not read the reference material: {exc}")
                return {"CANCELLED"}

        done, skipped = [], []
        for name in names:
            target = os.path.join(materials_dir, f"{name}.material")
            try:
                if self.mode == "WHOLE":
                    # Carry the material's own base colour across before the
                    # reference overwrites everything: retargeting alone would
                    # only fix the car folder and leave the reference car's
                    # texture *name*, pointing at a file this car doesn't have.
                    base_color = None
                    if os.path.isfile(target):
                        try:
                            existing = material_codec.decode_material(target)
                            base_color = next(
                                (t.path for t in existing.textures
                                 if t.name == _BASECOLOR_SLOT and t.path), None)
                        except Exception:  # noqa: BLE001 - unreadable original, fall through
                            base_color = None
                    if base_color is None:
                        base_color = _reference_texture_path(context, car_prefix)

                    mf = material_codec.decode_material(_bundled_refmat_path())
                    for tex in mf.textures:
                        tex.path = _retarget_texture_path(tex.path, car_prefix)
                    if base_color:
                        for tex in mf.textures:
                            if tex.name == _BASECOLOR_SLOT:
                                tex.path = base_color
                else:
                    if not os.path.isfile(target):
                        skipped.append(f"{name} (no existing .material)")
                        continue
                    mf = material_codec.decode_material(target)

                for tex in mf.textures:
                    if tex.name in mask_paths:
                        tex.path = mask_paths[tex.name]
                if normal_path is not None:
                    for tex in mf.textures:
                        if tex.name == _NORMAL_MAP_SLOT:
                            tex.path = normal_path
                    for prop in mf.properties:
                        if prop.name in _NORMAL_MAP_PROPS:
                            prop.kind = material_codec.KIND_SCALAR
                            prop.components = {1: _NORMAL_MAP_PROPS[prop.name]}
                _set_material_blend(mf, transparent)

                if os.path.isfile(target):
                    backup = target + ".bak"
                    if not os.path.exists(backup):
                        shutil.copy(target, backup)
                with open(target, "wb") as fh:
                    fh.write(material_codec.encode_material(mf))
                done.append(name)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{name} ({exc})")

        mode_label = "whole material" if self.mode == "WHOLE" else "mask textures only"
        blend_label = "transparent (blendMode=1, root field=2)" if transparent else "opaque (blendMode=0, no root field)"
        self.report({"INFO"}, f"{len(done)} material(s) written [{mode_label}, {blend_label}]: {', '.join(done) or 'none'}")
        if skipped:
            self.report({"WARNING"}, f"Skipped: {'; '.join(skipped)}")
        return {"FINISHED"}


class MESH_OT_ac_export_fixed_materials(Operator):
    """Writes the .material file for every EXT_LIGHTS_{PLASTIC,GLASS,CHROME}_FIXED
    mesh currently in the scene - whichever of the three don't exist are just
    skipped, no selection needed. Each one uses its matching bundled preset
    (refmat_plastic/glass/chrome.material); blend mode comes from that preset
    as-is (see the Glass preset for how transparency is handled) - nothing is
    toggled here.

    Every texture slot that matters is repointed at real content: Base Color
    -> Base_BaseColorMap, Normal Map -> Base_NormalMap, and the two painted
    function masks (scene.ac_lights_f1/f2_image, same as the generic
    Texture-only flow) -> FunctionMask1/2. Any of those that isn't already an
    existing car texture (a Prepare atlas, a freshly painted mask, ...) gets
    written out as a new .texture/.texturemips pair; one that already points
    at a real file is left untouched. A slot with nothing to put in it is
    explicitly cleared rather than left at the bundled preset's own leftover
    placeholder path, which would point at a file this car doesn't have."""
    bl_idname = "mesh.ac_export_fixed_materials"
    bl_label = "Export FIXED materials"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        if not any(bpy.data.objects.get(name) for name in _LIGHT_KIND_MESH_NAMES.values()):
            self.report({"ERROR"}, "No EXT_LIGHTS_*_FIXED mesh in the scene - run Prepare first.")
            return {"CANCELLED"}
        if _car_root_from_scene() is None:
            self.report({"ERROR"}, "No AC-imported mesh in the scene - cannot locate the car folder.")
            return {"CANCELLED"}
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        car_root = _car_root_from_scene()
        if car_root is None:
            self.report({"ERROR"}, "No AC-imported mesh in the scene.")
            return {"CANCELLED"}
        car_prefix = _car_content_prefix(car_root)
        if car_prefix is None:
            self.report({"ERROR"}, f"Could not derive a content\\cars\\... prefix from {car_root}.")
            return {"CANCELLED"}
        materials_dir = os.path.join(car_root, "materials")
        if not os.path.isdir(materials_dir):
            self.report({"ERROR"}, f"No materials/ folder in {car_root}.")
            return {"CANCELLED"}
        tex_folder = _car_texture_folder()

        def _export_image(image, fmt: str, basename: str | None = None) -> str:
            """Writes `image` to the car's texture folder if it isn't already
            an existing car texture, returning the engine-relative path to
            use in the material either way. `fmt` is "BC6H" (normal map),
            "BC7" (base colour, always compressed) or "MASK" (F1/F2 - format
            follows the scene's own toggle, RGBA8 vs BC7, since that one's
            meant to be user-chosen rather than fixed)."""
            existing = _image_existing_path(image)
            if existing:
                return existing
            if tex_folder is None:
                raise RuntimeError("no texture/ folder found next to meshes/")
            base_name = basename or _clean_texture_basename(image.name)
            w, h = image.size
            effective_fmt = fmt
            if effective_fmt == "MASK":
                effective_fmt = context.scene.ac_mask_export_format
            if effective_fmt == "BC6H":
                rgb = _image_to_array(image)[::-1, :, :3]
                texture_bytes, mips_bytes = ace_texture.rgb_to_bc6h_texture(rgb, w, h)
                fmt_label = "BC6H_UF16"
            else:
                rgba = np.clip(_image_to_array(image)[::-1] * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
                rgba[:, :, 3] = 255
                if effective_fmt == "RGBA8":
                    texture_bytes, mips_bytes = ace_texture.image_to_texture(rgba.tobytes(), w, h)
                    fmt_label = "RGBA8"
                else:
                    texture_bytes, mips_bytes = ace_texture.rgba_to_bc7_texture(rgba, w, h, srgb=True)
                    fmt_label = "BC7_UNORM_SRGB"
            with open(os.path.join(tex_folder, f"{base_name}.texture"), "wb") as fh:
                fh.write(texture_bytes)
            with open(os.path.join(tex_folder, f"{base_name}.texturemips"), "wb") as fh:
                fh.write(mips_bytes)
            written_textures.append(f"{base_name}.texture ({w}x{h}, {fmt_label})")
            return f"{car_prefix}\\texture\\{base_name}.texture"

        done, skipped, written_textures = [], [], []

        # Same painted-mask mechanism as the generic Texture-only flow, and
        # shared identically across all three FIXED materials - there is only
        # ever one F1/F2 pair for the whole car, painted once. Export the
        # baked (painted x B&W reference) result when one exists - same rule
        # "Save F_1"/"Save F_2" already follow - never the raw painted layer,
        # which is only ever an intermediate editing state.
        mask_paths = {}
        for slot, prop_name in _MASK_SLOT_TO_PROP.items():
            painted = getattr(context.scene, prop_name)
            if painted is None:
                continue
            baked = bpy.data.images.get(f"{painted.name}_BAKED")
            export_source = baked if baked is not None else painted
            texture_slot = "F1" if slot == "FunctionMask1" else "F2"
            base_name = _mask_output_name(context, texture_slot, painted)
            try:
                mask_paths[slot] = _export_image(export_source, fmt="MASK", basename=base_name)
            except Exception as exc:  # noqa: BLE001
                self.report({"WARNING"}, f"Could not export {slot}: {exc}")

        any_normal_toggle_on = any(getattr(context.scene, prop) for prop in _NORMAL_TOGGLE_PROP.values())
        nm_scene_image = context.scene.ac_normal_map_image
        if any_normal_toggle_on and (nm_scene_image is None or nm_scene_image.size[0] == 0):
            self.report(
                {"WARNING"},
                "A normal map checkbox is on but no normal map has been generated "
                "('Generate normal map') - skipped for every kind.",
            )

        for kind, target_name in _LIGHT_KIND_MESH_NAMES.items():
            obj = bpy.data.objects.get(target_name)
            if obj is None:
                continue
            mats = obj.data.materials
            if not mats or mats[0] is None:
                skipped.append(f"{target_name} (no material)")
                continue
            mat = mats[0]

            try:
                base_img = _find_base_color_image(mat)
                base_color_path = _export_image(base_img, fmt="BC7") if base_img is not None else None
                # The generated normal map (scene.ac_normal_map_image, from
                # "Generate normal map") is the one and only normal source
                # here - each kind's checkbox just decides whether to apply
                # that same map to it, not whether the material happens to
                # already have something wired in its own node graph (a
                # freshly split-off Glass/Chrome material never does).
                normal_wanted = getattr(context.scene, _NORMAL_TOGGLE_PROP[kind])
                normal_source = context.scene.ac_normal_map_image
                normal_img = normal_source if (normal_wanted and normal_source is not None
                                                and normal_source.size[0] > 0) else None
                normal_path = _export_image(normal_img, fmt="BC6H") if normal_img is not None else None

                mf = material_codec.decode_material(_bundled_kind_preset_path(kind))
                for tex in mf.textures:
                    tex.path = _retarget_texture_path(tex.path, car_prefix)
                for tex in mf.textures:
                    if tex.name == _BASECOLOR_SLOT and base_color_path:
                        tex.path = base_color_path
                    elif tex.name == _NORMAL_MAP_SLOT and normal_path:
                        tex.path = normal_path
                    elif tex.name in mask_paths:
                        tex.path = mask_paths[tex.name]
                    elif tex.name in _MASK_SLOT_TO_PROP:
                        # No mask painted for this slot - clear the preset's
                        # own leftover path rather than ship a reference to a
                        # file this car doesn't have.
                        tex.path = None
                if normal_path:
                    for prop in mf.properties:
                        if prop.name in _NORMAL_MAP_PROPS:
                            prop.kind = material_codec.KIND_SCALAR
                            prop.components = {1: _NORMAL_MAP_PROPS[prop.name]}

                target = os.path.join(materials_dir, f"{target_name}.material")
                if os.path.isfile(target):
                    backup = target + ".bak"
                    if not os.path.exists(backup):
                        shutil.copy(target, backup)
                with open(target, "wb") as fh:
                    fh.write(material_codec.encode_material(mf))
                done.append(target_name)
            except Exception as exc:  # noqa: BLE001
                skipped.append(f"{target_name} ({exc})")

        if not done and not skipped:
            self.report({"ERROR"}, "No EXT_LIGHTS_*_FIXED mesh in the scene - run Prepare first.")
            return {"CANCELLED"}

        msg = f"{len(done)} material(s) written: {', '.join(done) or 'none'}"
        if written_textures:
            msg += f" | textures: {', '.join(written_textures)}"
        self.report({"INFO"}, msg)
        if skipped:
            self.report({"WARNING"}, f"Skipped: {'; '.join(skipped)}")
        return {"FINISHED"}


class VIEW3D_PT_ac_input(Panel):
    """Turns the merged EXT_LIGHTS_* geometry into one clean, non-overlapping
    atlas - the starting point every other tab (texture paint, output) reads
    from."""
    bl_label = "1 - Input"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AC Mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Prepare (join EXT_LIGHTS_* by texture)")
        box.operator("mesh.ac_prepare_lights", text="Prepare", icon="MOD_BOOLEAN")
        box.prop(scene, "ac_prepare_atlas_size", text="Atlas size")
        box.prop(scene, "ac_clean_lights_before_prepare")
        box.operator("mesh.ac_clean_selected_mesh", text="Clean selected mesh", icon="TRASH")


class VIEW3D_PT_ac_vertex_paint(Panel):
    bl_label = "2 - Vertex painting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AC Mesh"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        layout = self.layout
        in_edit = bool(_meshes_in_edit_mode(context))

        row = layout.row()
        row.scale_y = 1.4
        row.alert = True
        row.operator("mesh.ac_erase_all_vcolor", text="ERASE ALL (every AC mesh)", icon="TRASH")

        if not in_edit:
            layout.separator()
            layout.label(text="Switch to Edit mode to paint", icon="INFO")
            return

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Lights - position")
        grid = col.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
        for key, label, _color in _LIGHT_POSITIONS:
            grid.operator("mesh.ac_paint_position", text=label).position = key

        col = layout.column(align=True)
        col.label(text="Brake disc - position")
        row = col.row(align=True)
        for key, label, _color in _DISC_POSITIONS:
            row.operator("mesh.ac_paint_position", text=label).position = key

        row = layout.row(align=True)
        row.operator("mesh.ac_erase_vcolor", text="Erase selection").whole_mesh = False
        row.operator("mesh.ac_erase_vcolor", text="Erase these meshes").whole_mesh = True


class VIEW3D_PT_ac_texture_paint(Panel):
    bl_label = "3 - Texture paint"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AC Mesh"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        in_edit = bool(_meshes_in_edit_mode(context))

        box = layout.box()
        box.label(text="Light kind (which FIXED material this belongs to)")
        row = box.row(align=True)
        row.operator("mesh.ac_split_light_kind", text="Is plastic").kind = "PLASTIC"
        row.operator("mesh.ac_split_light_kind", text="Is glass").kind = "GLASS"
        row.operator("mesh.ac_split_light_kind", text="Is chrome").kind = "CHROME"

        box = layout.box()
        box.label(text="Reference (optic) - B&W base for the bake")
        box.prop(scene, "ac_lights_ref_image", text="Reference")
        box.operator("mesh.ac_ref_from_material", text="Use actual (Base Color -> B&W)", icon="NODE_TEXTURE")
        row = box.row(align=True)
        row.enabled = scene.ac_lights_ref_image is not None
        row.operator("mesh.ac_generate_normal_map", text="Generate normal map", icon="NORMALS_FACE")
        box.prop(scene, "ac_normal_map_image", text="Normal map")

        for slot, title, front, rear in (
            ("F1", "EXT_Lights_F_1", _LIGHTS_F1_FRONT, _LIGHTS_F1_REAR),
            ("F2", "EXT_Lights_F_2", _LIGHTS_F2_CHANNELS, None),
        ):
            box = layout.box()
            box.label(text=title)
            current_image = getattr(scene, f"ac_lights_{slot.lower()}_image")
            row = box.row(align=True)
            row.prop(scene, f"ac_lights_{slot.lower()}_image", text="Image")
            row.operator("mesh.ac_create_light_texture", text="", icon="ADD").texture_slot = slot
            if _image_existing_path(current_image):
                box.label(text=f"From material: {current_image.name}", icon="LINKED")

            if rear is None:
                col = box.column(align=True)
                col.enabled = in_edit
                row = col.row(align=True)
                for ch, label, prop_name in front:
                    op = row.operator("mesh.ac_paint_light_texture", text=label)
                    op.texture_slot, op.channel = slot, ch
                    op.value = 1.0 if scene.ac_use_data_intensity else getattr(scene, prop_name)
            else:
                for group_label, entries in (("Front", front), ("Rear", rear)):
                    col = box.column(align=True)
                    col.enabled = in_edit
                    col.label(text=group_label)
                    row = col.row(align=True)
                    for ch, label, prop_name in entries:
                        op = row.operator("mesh.ac_paint_light_texture", text=label)
                        op.texture_slot, op.channel = slot, ch
                        op.value = 1.0 if scene.ac_use_data_intensity else getattr(scene, prop_name)

            row = box.row(align=True)
            sub = row.row(align=True)
            sub.enabled = in_edit
            sub.operator("mesh.ac_erase_light_texture", text="Erase selection").texture_slot = slot
            op = row.operator("mesh.ac_erase_light_texture", text="Erase ALL")
            op.texture_slot, op.whole_image = slot, True

            image = scene.ac_lights_f1_image if slot == "F1" else scene.ac_lights_f2_image
            depth = _paint_history_depth(image)
            row = box.row(align=True)
            row.enabled = depth > 0
            row.operator(
                "mesh.ac_undo_paint", text=f"Undo paint ({depth})", icon="LOOP_BACK",
            ).texture_slot = slot



_SHOW_REFERENCE_MATERIAL_UI = False


class VIEW3D_PT_ac_export(Panel):
    """Everything that writes to disk: baking, the game-format textures and
    the .material files."""
    bl_label = "4 - Output"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AC Mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Bake (paint x B&W reference)")
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("mesh.ac_bake_lights", text="Bake F_1", icon="RENDERLAYERS").texture_slot = "F1"
        row.operator("mesh.ac_bake_lights", text="Bake F_2", icon="RENDERLAYERS").texture_slot = "F2"

        folder = _car_texture_folder()
        box = layout.box()
        box.label(text="Textures -> game format")
        box.prop(scene, "ac_mask_export_format", text="Format")
        row = box.row(align=True)
        row.enabled = folder is not None
        row.operator("mesh.ac_save_ac_texture", text="Save F_1", icon="EXPORT").texture_slot = "F1"
        row.operator("mesh.ac_save_ac_texture", text="Save F_2", icon="EXPORT").texture_slot = "F2"
        box.label(text=(os.path.basename(folder) + "/  (next to meshes/)") if folder
                  else "No texture/ folder found", icon="FILE_FOLDER" if folder else "ERROR")

        # Hidden for now (kept working, just not exposed) - the FIXED
        # workflow below has fully replaced this generic "apply a reference
        # material to whatever's selected" path for day-to-day use.
        if _SHOW_REFERENCE_MATERIAL_UI:
            names = _selected_material_names(context)
            box = layout.box()
            box.label(text="Materials -> reference material")
            # "Is transparent" is hidden for now - the FIXED workflow below picks
            # transparency up from the Glass preset itself instead of a manual
            # toggle. The property and _set_material_blend() are still there,
            # still used by this generic path, just not exposed here anymore.
            row = box.row()
            row.prop(scene, "ac_apply_normal_map")
            if scene.ac_apply_normal_map and scene.ac_normal_map_image is None:
                box.label(text="No normal map generated yet", icon="ERROR")
            col = box.column(align=True)
            col.enabled = bool(names)
            col.operator(
                "mesh.ac_apply_ref_material", text="Texture only", icon="TEXTURE",
            ).mode = "TEXTURE_ONLY"
            col.operator(
                "mesh.ac_apply_ref_material", text="Whole material", icon="MATERIAL",
            ).mode = "WHOLE"
            if names:
                shown = ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "")
                box.label(text=f"{len(names)} material(s): {shown}", icon="CHECKMARK")
            else:
                box.label(text="Select meshes to target their materials", icon="ERROR")

        box = layout.box()
        box.label(text="FIXED light materials (plastic/glass/chrome presets)")
        present = [kind for kind, name in _LIGHT_KIND_MESH_NAMES.items() if bpy.data.objects.get(name)]
        row = box.row(align=True)
        row.prop(scene, "ac_export_plastic_normal", text="Plastic normal map")
        row.prop(scene, "ac_export_glass_normal", text="Glass normal map")
        row.prop(scene, "ac_export_chrome_normal", text="Chrome normal map")
        row = box.row(align=True)
        row.scale_y = 1.3
        row.enabled = bool(present)
        row.operator("mesh.ac_export_fixed_materials", text="Export FIXED materials", icon="EXPORT")
        if present:
            box.label(text=f"Found: {', '.join(k.title() for k in present)}", icon="CHECKMARK")
        else:
            box.label(text="No EXT_LIGHTS_*_FIXED mesh - run Prepare / Is plastic/glass/chrome first", icon="ERROR")

        box = layout.box()
        box.label(text="Actor file")
        box.operator("mesh.ac_open_actor", text="Open actor", icon="FILE_TEXT")


class VIEW3D_PT_ac_intensity(Panel):
    """Per-function intensity used by the Texture Paint buttons."""
    bl_label = "3.1 - Intensity"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AC Mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "ac_use_data_intensity")
        layout.label(text="Value written into the channel (1.0 = 255)")
        col = layout.column()
        col.enabled = not scene.ac_use_data_intensity
        if scene.ac_use_data_intensity:
            col.label(text="Ignored - every channel paints at 1.0 while this is on", icon="INFO")
        for group_label, entries in (
            ("F_1 Front", _LIGHTS_F1_FRONT),
            ("F_1 Rear", _LIGHTS_F1_REAR),
            ("F_2", _LIGHTS_F2_CHANNELS),
        ):
            sub = col.column(align=True)
            sub.label(text=group_label)
            for _ch, _label, prop_name in entries:
                sub.prop(scene, prop_name, slider=True)


class VIEW3D_PT_ac_debug(Panel):
    """Low-level / occasional-use tools."""
    bl_label = "5 - Other"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AC Mesh"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="New object -> not picked up by Export until tagged")
        box.operator("mesh.ac_tag_for_export", text="Tag selected for export", icon="LINKED")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def _menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_ac_mesh.bl_idname, text="Assetto Corsa Mesh (.mesh)")


def _menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_ac_mesh.bl_idname, text="Assetto Corsa Mesh (.mesh)")


_classes = (
    IMPORT_OT_ac_mesh, EXPORT_OT_ac_mesh,
    MESH_OT_ac_paint_position, MESH_OT_ac_erase_vcolor, MESH_OT_ac_erase_all_vcolor,
    MESH_OT_ac_clean_selected_mesh,
    MESH_OT_ac_prepare_lights, MESH_OT_ac_split_light_kind,
    MESH_OT_ac_create_light_texture, MESH_OT_ac_ref_from_material, MESH_OT_ac_generate_normal_map,
    MESH_OT_ac_undo_paint, MESH_OT_ac_bake_lights, MESH_OT_ac_save_ac_texture,
    MESH_OT_ac_paint_light_texture, MESH_OT_ac_erase_light_texture,
    MESH_OT_ac_apply_ref_material, MESH_OT_ac_export_fixed_materials,
    MESH_OT_ac_tag_for_export, MESH_OT_ac_open_actor,
    VIEW3D_PT_ac_input, VIEW3D_PT_ac_vertex_paint, VIEW3D_PT_ac_texture_paint,
    VIEW3D_PT_ac_intensity, VIEW3D_PT_ac_export, VIEW3D_PT_ac_debug,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(_menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(_menu_func_export)
    bpy.types.Scene.ac_lights_f1_image = PointerProperty(type=bpy.types.Image, name="EXT_Lights_F_1")
    bpy.types.Scene.ac_lights_f2_image = PointerProperty(type=bpy.types.Image, name="EXT_Lights_F_2")
    bpy.types.Scene.ac_use_data_intensity = BoolProperty(
        name="Use data intensity",
        description="Ignore the per-function intensity sliders below and paint every channel at 1.0 instead",
        default=True,
    )
    bpy.types.Scene.ac_prepare_atlas_size = EnumProperty(
        name="Atlas size",
        description="Resolution of the Prepare atlas texture (always square)",
        items=[("1024", "1024", ""), ("2048", "2048", ""), ("4096", "4096", "")],
        default="2048",
    )
    bpy.types.Scene.ac_clean_lights_before_prepare = BoolProperty(
        name="Clean lights mesh first",
        description=(
            "Before merging, remove coincident duplicate triangles (same 3 corner positions as "
            "another triangle) from every EXT_LIGHTS_* object - the post-import counterpart of "
            "'Disable mesh cleaning' on import. On by default; only matters if that import option "
            "was used, otherwise there's nothing to clean"
        ),
        default=True,
    )
    bpy.types.Scene.ac_mask_export_format = EnumProperty(
        name="F1/F2 format",
        description=(
            "Texture format for Save F_1/F_2 (and the F1/F2 masks written by "
            "Export FIXED materials) - the base colour atlas and normal map "
            "are unaffected, always BC7/BC6H respectively"
        ),
        items=[
            ("BC7", "BC7", "Compressed, matches shipped car content"),
            ("RGBA8", "RGBA8", "Uncompressed - different look in-game, larger files"),
        ],
        default="RGBA8",
    )
    bpy.types.Scene.ac_export_plastic_normal = BoolProperty(
        name="Plastic normal map",
        description="Populate Base_NormalMap and its enable properties on the Plastic FIXED material",
        default=True,
    )
    bpy.types.Scene.ac_export_glass_normal = BoolProperty(
        name="Glass normal map",
        description="Populate Base_NormalMap and its enable properties on the Glass FIXED material",
        default=False,
    )
    bpy.types.Scene.ac_export_chrome_normal = BoolProperty(
        name="Chrome normal map",
        description="Populate Base_NormalMap and its enable properties on the Chrome FIXED material",
        default=True,
    )
    bpy.types.Scene.ac_material_transparent = BoolProperty(
        name="Is transparent",
        description=(
            "Transparent: blendMode=1 plus the hidden root blend field set to AlphaBlend. "
            "Opaque: blendMode=0 and the hidden field removed. Both are always written "
            "together - setting only one leaves the material in an inconsistent state"
        ),
        default=False,
    )
    bpy.types.Scene.ac_normal_map_image = PointerProperty(
        type=bpy.types.Image, name="Normal map",
        description="Normal map generated from the reference image, written out with the materials",
    )
    bpy.types.Scene.ac_apply_normal_map = BoolProperty(
        name="Add/replace normal map",
        description=(
            "Write the generated normal map into the car's texture folder and wire it into every "
            "material being written: Base_NormalMap path, Base_HasNormalMap=1, Base_NormalScale=0.5"
        ),
        default=False,
    )
    bpy.types.Scene.ac_lights_ref_image = PointerProperty(
        type=bpy.types.Image, name="Reference",
        description=(
            "The optic's own colour/AO texture, used as a black & white base: its shading is what "
            "keeps the light from reading flat in-game"
        ),
    )
    if _ac_selection_changed not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_ac_selection_changed)
    for prop_name, label, default in _INTENSITY_PROPS:
        setattr(bpy.types.Scene, prop_name, FloatProperty(
            name=label, default=default, min=0.0, max=1.0, subtype="FACTOR",
            description=(
                "Value written into this function's channel. In-game the colour intensity drives the "
                "light intensity: 1.0 = channel at full (255), 0.0 = off"
            ),
        ))


def unregister():
    if _ac_selection_changed in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_ac_selection_changed)
    _BASECOLOR_BY_MATERIAL.clear()
    for prop_name, _label, _default in _INTENSITY_PROPS:
        delattr(bpy.types.Scene, prop_name)
    del bpy.types.Scene.ac_apply_normal_map
    del bpy.types.Scene.ac_normal_map_image
    del bpy.types.Scene.ac_material_transparent
    del bpy.types.Scene.ac_export_chrome_normal
    del bpy.types.Scene.ac_export_glass_normal
    del bpy.types.Scene.ac_export_plastic_normal
    del bpy.types.Scene.ac_prepare_atlas_size
    del bpy.types.Scene.ac_use_data_intensity
    del bpy.types.Scene.ac_lights_ref_image
    del bpy.types.Scene.ac_lights_f2_image
    del bpy.types.Scene.ac_lights_f1_image
    bpy.types.TOPBAR_MT_file_import.remove(_menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(_menu_func_export)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
