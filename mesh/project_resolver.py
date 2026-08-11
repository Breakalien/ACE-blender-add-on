"""
Resolves the cross-references found inside a standard Assetto Corsa content
folder (the kind produced by extracting a .kn5-based mod: meshes/, materials/,
texture/, parts/, skins/, displays/, collider/, ...).

A .mesh file's MaterialRange.path and a .material file's TextureSlot.path both
look like:

    content\\cars\\ks_ks_merce_w\\materials\\EXT_WINDOWS.material
    content\\cars\\ks_ks_merce_w\\texture\\damage_dirt\\Damage_Areas.texture
    content\\cars\\common_assets\\parts\\tyres\\materials\\EXT_TYRE.material

i.e. "content\\cars\\<car id>\\<path relative to the car's own content root>".
The project root the user points us at (e.g. "merc/") corresponds to that
"<car id>" folder, so resolution strips the "content\\cars\\<car id>\\" prefix
and joins the rest onto the project root. References into shared/global
content ("content\\cars\\common_assets\\...", or "editor\\..." placeholders)
aren't part of a single car's own export - if the caller supplies a second
`assets_root` (a folder with the same layout, containing whatever generic
materials/textures the car's own folder doesn't ship), those get resolved
there instead; without one they simply won't resolve, and callers must
tolerate that.

Image assets referenced by a material aren't always sitting where the
reference says, and aren't always in the packed .texture/.texturemips
format either - alternate skins, display/instrument textures etc. are
sometimes plain .dds/.png/.jpg placed in a completely different folder
(skins/, displays/, generated/, ...). `find_image_asset` looks in the
referenced folder first (trying every known image extension, not just the
one in the reference), then falls back to a project-wide index keyed by
filename stem alone.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass

from .ace_texture import texture_to_dds
from .material_codec import decode_material, MaterialFile, KIND_VEC2, KIND_VEC3, KIND_VEC4

_SPLIT_RE = re.compile(r"[\\/]+")

# Order matters: preferred format wins when several exist for the same stem.
IMAGE_EXTENSIONS = [".texture", ".dds", ".png", ".jpg", ".jpeg", ".tga", ".bmp"]


def _is_placeholder_path(raw_path: str | None) -> bool:
    """The AC EVO material editor auto-fills unassigned texture/material
    slots with a sentinel under "editor\\..." (e.g.
    "editor\\textures\\default_material\\albedo2.texture",
    "editor\\default.material") rather than leaving them empty. That's not a
    missing asset, it's the editor's own way of saying "nothing assigned
    here" - many materials deliberately have no diffuse texture at all and
    rely on a constant paint colour instead (see ResolvedMaterial.diffuse_color)."""
    if not raw_path:
        return False
    parts = [p for p in _SPLIT_RE.split(raw_path) if p]
    return bool(parts) and parts[0].lower() == "editor"


class _RootIndex:
    """Basename/stem index over a single folder tree - see ProjectIndex,
    which holds one of these per root it searches."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.by_basename_lower: dict[str, list[str]] = {}
        self.images_by_stem_lower: dict[str, list[str]] = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                self.by_basename_lower.setdefault(fn.lower(), []).append(full)
                stem, ext = os.path.splitext(fn)
                if ext.lower() in IMAGE_EXTENSIONS:
                    self.images_by_stem_lower.setdefault(stem.lower(), []).append(full)


class ProjectIndex:
    def __init__(self, root: str, assets_root: str | None = None):
        self._roots = [_RootIndex(root)]
        if assets_root:
            self._roots.append(_RootIndex(assets_root))

    @property
    def root(self) -> str:
        """The car's own content root (first root searched)."""
        return self._roots[0].root

    @property
    def assets_root(self) -> str | None:
        """The shared/generic assets root (common_assets, editor/...), if any."""
        return self._roots[1].root if len(self._roots) > 1 else None

    def resolve(self, raw_path: str) -> str | None:
        """Best-effort resolution of a content-relative path to a real file -
        tries the car's own root first, then the assets root if one was
        supplied (used for .material lookups, where the extension is
        authoritative). Returns None if unresolved anywhere."""
        if not raw_path:
            return None
        rest = self.relative_rest(raw_path)

        if rest:
            for ri in self._roots:
                candidate = os.path.join(ri.root, *rest)
                if os.path.isfile(candidate):
                    return candidate

        basename = rest[-1].lower() if rest else None
        if basename:
            for ri in self._roots:
                matches = ri.by_basename_lower.get(basename)
                if matches:
                    return matches[0]
        return None

    def relative_rest(self, raw_path: str) -> list[str]:
        """The path segments after "content\\cars\\<car id>\\" (for mirroring
        output layout), regardless of whether resolution actually succeeded."""
        parts = [p for p in _SPLIT_RE.split(raw_path) if p]
        if "cars" in parts:
            idx = parts.index("cars")
            return parts[idx + 2:]
        return parts

    def find_image_asset(self, raw_path: str) -> str | None:
        """Format-agnostic texture resolution: same folder, trying every
        known image extension (not just the one in the reference) before
        falling back to a stem-only search - handles skins/alternate
        textures that got authored/exported as a plain .dds/.png instead of
        the packed .texture format, or that live in a folder the
        reference's own path doesn't point at. Tries the car's own root
        first, then the assets root if one was supplied."""
        if not raw_path:
            return None
        rest = self.relative_rest(raw_path)
        if not rest:
            return None

        stem, orig_ext = os.path.splitext(rest[-1])
        ext_order = [orig_ext.lower()] + [e for e in IMAGE_EXTENSIONS if e != orig_ext.lower()]

        for ri in self._roots:
            directory = os.path.join(ri.root, *rest[:-1])
            for ext in ext_order:
                candidate = os.path.join(directory, stem + ext)
                if os.path.isfile(candidate):
                    return candidate

        for ri in self._roots:
            matches = ri.images_by_stem_lower.get(stem.lower())
            if not matches:
                continue
            for ext in ext_order:
                for m in matches:
                    if m.lower().endswith(ext):
                        return m
            return matches[0]
        return None

    def relpath_from_root(self, resolved_path: str) -> str:
        """Path relative to whichever of this index's roots actually
        contains `resolved_path`, prefixed with "assets/" when it came from
        the assets root - keeps shared-asset outputs organised separately
        from (and never colliding with) the car's own texture tree."""
        resolved_abs = os.path.abspath(resolved_path)
        for i, ri in enumerate(self._roots):
            try:
                rel = os.path.relpath(resolved_abs, ri.root)
            except ValueError:
                continue  # different drive on Windows - can't be this root
            if not rel.startswith(".."):
                return os.path.join("assets", rel) if i > 0 else rel
        return os.path.basename(resolved_path)


@dataclass
class ResolvedMaterial:
    name: str
    shader_name: str
    diffuse_texture: str | None = None  # absolute path, converted/copied under the output folder
    normal_texture: str | None = None
    opacity_texture: str | None = None  # from the material's *OpacityMap slot - never the diffuse's own alpha
    uv_scale: tuple = (1.0, 1.0)  # (U, V) tiling factor for the diffuse/normal maps
    diffuse_color: tuple = (0.8, 0.8, 0.8)  # constant paint colour, used as-is when there's no diffuse texture
    diffuse_raw: str | None = None  # engine-relative Base_BaseColorMap path
    normal_raw: str | None = None  # engine-relative Base_NormalMap path


class TextureConverter:
    """Resolves + materializes referenced textures under <output_root>/textures/,
    once per unique source file (cached across the whole project run).

    .texture/.texturemips pairs are converted to .dds via ace_texture; any
    other already-standard image format (.dds/.png/.jpg/.jpeg/.tga/.bmp) is
    simply copied as-is - FBX importers handle those natively, no conversion
    needed."""

    def __init__(self, index: ProjectIndex, output_root: str):
        self.index = index
        self.output_root = output_root
        self._cache: dict[str, str | None] = {}  # resolved source path -> absolute output path (or None if failed)
        self.warnings: list[str] = []

    def convert(self, raw_texture_path: str | None) -> str | None:
        """Returns the *absolute* path to the materialized texture (converted
        or copied), or None if it couldn't be found/processed."""
        if not raw_texture_path:
            return None

        resolved = self.index.find_image_asset(raw_texture_path)
        if resolved is None:
            # "editor\\..." placeholders are expected to be missing when no
            # assets_root was supplied at all - don't spam warnings for
            # those specifically; anything else (including a placeholder
            # that's *still* unresolved despite an assets_root) is worth
            # flagging.
            if not (_is_placeholder_path(raw_texture_path) and not self.index.assets_root):
                self.warnings.append(f"texture introuvable (aucun format connu): {raw_texture_path}")
            return None

        if resolved in self._cache:
            return self._cache[resolved]

        rel_from_root = self.index.relpath_from_root(resolved)
        ext = os.path.splitext(resolved)[1].lower()

        if ext == ".texture":
            out_path = self._convert_packed_texture(resolved, rel_from_root)
        else:
            out_path = self._copy_as_is(resolved, rel_from_root)

        self._cache[resolved] = out_path
        return out_path

    def _convert_packed_texture(self, resolved: str, rel_from_root: str) -> str | None:
        mips_path = os.path.splitext(resolved)[0] + ".texturemips"
        if not os.path.isfile(mips_path):
            self.warnings.append(f".texturemips manquant pour {rel_from_root}")
            return None

        rel_out = os.path.join("textures", os.path.splitext(rel_from_root)[0] + ".dds")
        out_path = os.path.abspath(os.path.join(self.output_root, rel_out))
        try:
            dds_bytes = texture_to_dds(open(resolved, "rb").read(), open(mips_path, "rb").read())
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(dds_bytes)
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"echec conversion texture {rel_from_root}: {exc}")
            return None
        return out_path

    def _copy_as_is(self, resolved: str, rel_from_root: str) -> str | None:
        rel_out = os.path.join("textures", rel_from_root)
        out_path = os.path.abspath(os.path.join(self.output_root, rel_out))
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if os.path.abspath(resolved) != out_path:
                shutil.copyfile(resolved, out_path)
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"echec copie texture {rel_from_root}: {exc}")
            return None
        return out_path


# Preferred texture-slot names for the channels we wire into the FBX material,
# in priority order (first one with an actual assigned path wins).
DIFFUSE_SLOT_CANDIDATES = ["txDiffuse", "Base_BaseColorMap", "BaseColorMap"]
NORMAL_SLOT_CANDIDATES = ["txNormal", "Base_NormalMap", "NormalMap"]

# UberVehicleMaterial-family shaders split several channels (paint layers
# blended by vertex colour) each with their own prefixed property/slot names.
CHANNEL_PREFIXES = ("Base_", "Red_", "Green_", "Blue_")


def _channel_prefix(slot_name: str | None) -> str:
    if slot_name is None:
        return ""
    for prefix in CHANNEL_PREFIXES:
        if slot_name.startswith(prefix):
            return prefix
    return ""


def _pick_opacity_slot(tex_by_name: dict, diffuse_slot: str | None) -> str | None:
    """Opacity must always come from the material's dedicated *OpacityMap
    slot - never from the diffuse texture's own alpha channel, which this
    shader family doesn't use for transparency at all. Unlike diffuse,
    there's no constant-colour equivalent for opacity, so an "editor\\..."
    placeholder path is still worth trying (it'll resolve to the shared
    default mask if an assets_root was supplied, or just fail quietly
    otherwise - see TextureConverter.convert)."""
    prefix = _channel_prefix(diffuse_slot)
    candidates = ([f"{prefix}OpacityMap"] if prefix else []) + ["OpacityMap"]
    for slot in candidates:
        path = tex_by_name.get(slot)
        if path:
            return path
    for name, path in tex_by_name.items():
        if path and name.endswith("OpacityMap"):
            return path
    return None


# Constant paint-colour properties, tried in this order when no diffuse
# texture is actually assigned (a real, common case: liveries/paint jobs
# driven entirely by a flat colour, e.g. AC EVO's in-game colour picker).
BASECOLOR_PROPERTY_CANDIDATES = ["Base_Basecolor", "ksBaseColor", "Basecolor"]


def _constant_diffuse_color(mf: MaterialFile) -> tuple | None:
    by_name = {p.name: p for p in mf.properties}
    for name in BASECOLOR_PROPERTY_CANDIDATES:
        p = by_name.get(name)
        if p is not None and p.kind in (KIND_VEC3, KIND_VEC4):
            return (p.get(1, 0.0), p.get(2, 0.0), p.get(3, 0.0))
    return None


def _uv_scale_for_slot(mf: MaterialFile, slot_name: str | None) -> tuple:
    """Looks up the "<Channel>_UVscale" property (VEC2) matching the texture
    slot that ended up being used, so tiled textures (e.g. rim/checker
    patterns with UVscale 4x4) repeat correctly once baked into the FBX UVs.

    UberVehicleMaterial-family shaders prefix per-channel properties with
    Base_/Red_/Green_/Blue_ (e.g. "Base_UVscale" for Base_BaseColorMap);
    simple legacy slots (txDiffuse, txNormal...) share a single unprefixed
    "UVscale" property instead.
    """
    prefix = _channel_prefix(slot_name)
    prop_name = f"{prefix}UVscale" if prefix else "UVscale"
    for p in mf.properties:
        if p.name == prop_name and p.kind == KIND_VEC2:
            return (p.get(1, 1.0), p.get(2, 1.0))
    return (1.0, 1.0)


class MaterialResolver:
    """Decodes .material files (by resolved path) and converts their diffuse
    / normal / opacity textures, with caching so a material referenced by
    many meshes is only decoded/converted once per project run."""

    def __init__(self, index: ProjectIndex, texture_converter: TextureConverter):
        self.index = index
        self.textures = texture_converter
        self._cache: dict[str, ResolvedMaterial | None] = {}
        self.warnings: list[str] = []

    def resolve(self, material_name: str, raw_material_path: str | None) -> ResolvedMaterial:
        key = raw_material_path or material_name
        if key in self._cache and self._cache[key] is not None:
            return self._cache[key]

        is_placeholder_material = bool(raw_material_path) and _is_placeholder_path(raw_material_path)
        if is_placeholder_material and not self.index.assets_root:
            # "editor/default.material" - deliberately unassigned (e.g.
            # collider meshes) and no assets_root to actually look it up in.
            result = ResolvedMaterial(name=material_name, shader_name="")
            self._cache[key] = result
            return result

        resolved_path = self.index.resolve(raw_material_path) if raw_material_path else None
        if resolved_path is None:
            if not is_placeholder_material:
                self.warnings.append(f"materiau introuvable: {material_name} ({raw_material_path})")
            result = ResolvedMaterial(name=material_name, shader_name="")
            self._cache[key] = result
            return result

        try:
            mf: MaterialFile = decode_material(resolved_path)
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"echec lecture materiau {material_name}: {exc}")
            result = ResolvedMaterial(name=material_name, shader_name="")
            self._cache[key] = result
            return result

        tex_by_name = {t.name: t.path for t in mf.textures}

        def pick(candidates, skip_placeholder):
            for slot in candidates:
                path = tex_by_name.get(slot)
                if not path:
                    continue
                if skip_placeholder and _is_placeholder_path(path):
                    continue
                return slot, path
            return None, None

        # Diffuse specifically always prefers the material's constant paint
        # colour over an "editor\\..." placeholder texture (verified: for
        # this shader family, a placeholder txDiffuse/Base_BaseColorMap
        # reliably means "flat colour, see Base_Basecolor" - not "look up
        # the generic default bitmap"). Other channels have no such
        # constant-colour equivalent, so their placeholders are worth
        # trying against assets_root instead of being skipped outright.
        diffuse_slot, diffuse_raw = pick(DIFFUSE_SLOT_CANDIDATES, skip_placeholder=True)
        _normal_slot, normal_raw = pick(NORMAL_SLOT_CANDIDATES, skip_placeholder=False)
        opacity_raw = _pick_opacity_slot(tex_by_name, diffuse_slot)

        diffuse_texture = self.textures.convert(diffuse_raw)
        diffuse_color = (0.8, 0.8, 0.8)
        if diffuse_texture is None:
            # No diffuse texture actually resolved (either none was assigned
            # at all, or one was but couldn't be found - e.g. it lives in
            # common_assets and no assets_root was supplied) - fall back to
            # the material's constant paint colour instead of a flat grey.
            diffuse_color = _constant_diffuse_color(mf) or diffuse_color

        result = ResolvedMaterial(
            name=material_name,
            shader_name=mf.shader_name,
            diffuse_texture=diffuse_texture,
            normal_texture=self.textures.convert(normal_raw),
            opacity_texture=self.textures.convert(opacity_raw),
            uv_scale=_uv_scale_for_slot(mf, diffuse_slot),
            diffuse_color=diffuse_color,
            diffuse_raw=diffuse_raw,
            normal_raw=normal_raw,
        )
        self._cache[key] = result
        return result
