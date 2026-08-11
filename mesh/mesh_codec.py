"""
Full read/write codec for the reverse-engineered Assetto Corsa ".mesh" format
(plain Protocol Buffers, no shipped .proto - see the field-by-field notes
below, recovered the same way as mesh2fbx/mesh_reader.py in the sibling
"mesh2fbx" and "ultimate ace convertor" tools).

This module is pure Python (no `bpy`) so it can be unit-tested standalone -
`__init__.py` is the only bpy-dependent file in this addon, and only calls
into `load_mesh_file` / `replace_lod` from here.

MeshFile (top-level message):
    2 : varint   formatVersion (observed: 1)                              - preserved verbatim
    3 : varint   unknown (observed: 1)                                    - preserved verbatim
    5 : Lod      repeated - one entry per LOD (index 0 = LOD_A / highest detail)
    6 : Vec3     overall bounding-box min                                  - recomputed on save
    7 : Vec3     overall bounding-box max                                  - recomputed on save
    20: bytes    unknown / always empty in every sample seen               - preserved verbatim

Lod:
     1: varint   enabled flag (observed: always 1)
     3: float32  LOD-in distance (0.0 is omitted by protobuf, so LOD_A has no field 3)
     4: MaterialRange   repeated - submesh -> index-buffer range table
     5: bytes    positions   packed float32 vec3 per vertex
     6: bytes    normals     packed float32 vec3 per vertex
     7: bytes    uv0         packed float32 vec2 per vertex
     8: bytes    tangent     packed float32 vec4 per vertex (w = handedness)
     9: bytes    boneWeights packed float32 vec4 per vertex (rigid: always (1,0,0,0))
    10: bytes    boneIndices packed float32 vec4 per vertex (only [0] used)
    11: bytes    indices     packed varint triangle list (into the vertex streams above)
    12: Bone     repeated - rigid hierarchy used to animate movable parts (doors, mirrors, wipers...)
    13: bytes    extra       packed float32 vec4 per vertex (purpose unclear, looks like an 8bpc
                              mask; all zero on most vertices) - preserved verbatim if the vertex
                              count didn't change, zero-filled otherwise
    14: bytes    uv1         packed float32 vec2 per vertex (observed always zero-filled)
    15: bytes    uv2         packed float32 vec2 per vertex (observed always zero-filled)
    16: bytes    uv3         packed float32 vec2 per vertex (observed always zero-filled)
    17: Vec3     LOD-local bounding-box min      - recomputed on save
    18: Vec3     LOD-local bounding-box max      - recomputed on save

MaterialRange:
    1: string  material name
    2: varint  start offset into the index buffer (default 0, omitted for the first range)
    3: varint  number of indices used by this submesh
    4: string  material path (content\\...\\materials\\NAME.material)

Bone:
    1: string  name
    2: string  parent name (absent for the root bone)
    3: Matrix  local transform          - editable via the imported Empty's position/
                                           rotation/scale; re-encoded on save only for
                                           bones whose Empty actually moved (see Bone.raw)

Matrix (sparse row-major 4x4, zero components omitted by protobuf):
    1..16: float32  m[0][0]..m[3][3], row-major (translation lives in fields 13,14,15)

Vec3:
    1: float32 x
    2: float32 y
    3: float32 z

Vertex positions/normals are already baked to the car's *resting pose* -
editing them in Blender and saving back does not require touching the bone
hierarchy at all (see mesh2fbx's notes for the full reasoning).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# low-level protobuf primitives
# ---------------------------------------------------------------------------

def read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def write_varint(value: int) -> bytes:
    out = bytearray()
    v = value & 0xFFFFFFFFFFFFFFFF
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_tag(field_no: int, wire_type: int) -> bytes:
    return write_varint((field_no << 3) | wire_type)


def encode_varint_field(field_no: int, value: int, *, always: bool = False) -> bytes:
    if value == 0 and not always:
        return b""
    return encode_tag(field_no, 0) + write_varint(value)


def encode_fixed32_field(field_no: int, value: float, *, always: bool = False) -> bytes:
    if value == 0.0 and not always:
        return b""
    return encode_tag(field_no, 5) + struct.pack("<f", value)


def encode_bytes_field(field_no: int, payload: bytes, *, always: bool = False) -> bytes:
    if not payload and not always:
        return b""
    return encode_tag(field_no, 2) + write_varint(len(payload)) + payload


def encode_string_field(field_no: int, value: str | None) -> bytes:
    if not value:
        return b""
    return encode_bytes_field(field_no, value.encode("utf-8"), always=True)


def encode_packed_varint_field(field_no: int, values) -> bytes:
    if not values:
        return b""
    body = b"".join(write_varint(v) for v in values)
    return encode_bytes_field(field_no, body, always=True)


def encode_packed_float32_field(field_no: int, flat_values) -> bytes:
    if not flat_values:
        return b""
    body = struct.pack(f"<{len(flat_values)}f", *flat_values)
    return encode_bytes_field(field_no, body, always=True)


def iter_fields(buf: bytes):
    """Yield (field_no, wire_type, value, next_pos) for one protobuf message."""
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = read_varint(buf, pos)
        field_no = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            val, pos = read_varint(buf, pos)
        elif wire_type == 1:
            val = buf[pos:pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = read_varint(buf, pos)
            val = buf[pos:pos + length]
            pos += length
        elif wire_type == 5:
            val = struct.unpack_from("<f", buf, pos)[0]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} at offset {pos}")
        yield field_no, wire_type, val, pos


def group_fields(buf: bytes):
    out: dict[int, list] = {}
    for field_no, _wt, val, _pos in iter_fields(buf):
        out.setdefault(field_no, []).append(val)
    return out


def packed_varints(buf: bytes):
    pos = 0
    out = []
    while pos < len(buf):
        v, pos = read_varint(buf, pos)
        out.append(v)
    return out


def packed_floats(buf: bytes):
    n = len(buf) // 4
    return struct.unpack(f"<{n}f", buf)


def chunk(seq, n):
    return [tuple(seq[i:i + n]) for i in range(0, len(seq), n)]


def flatten(seq_of_tuples):
    out = []
    for t in seq_of_tuples:
        out.extend(t)
    return out


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class MaterialRange:
    name: str
    start: int
    count: int
    path: str | None


@dataclass
class Bone:
    name: str
    parent: str | None
    matrix: list  # 4x4, row-major, list of 4 lists of 4 floats
    raw: bytes | None  # exact original field12 payload, used verbatim on encode
    # whenever present. (Individual matrix cells that happen to equal exactly
    # 0.0 aren't always omitted by the original encoder, so re-deriving bytes
    # from the densified `matrix` can silently drop a few explicit-zero
    # fields and produce a non-identical - if numerically equivalent -
    # result; kept untouched for bones that weren't edited.) None means this
    # Bone was rebuilt from an edited Empty's transform - encode fresh bytes
    # from `matrix` instead.


@dataclass
class Lod:
    index: int
    distance: float
    materials: list  # MaterialRange
    positions: list  # (x,y,z), meters
    normals: list  # (x,y,z)
    uv0: list  # (u,v)
    tangents: list  # (x,y,z,w)
    bone_weights: list  # (w0,w1,w2,w3)
    bone_indices: list  # (i0,i1,i2,i3) as ints
    indices: list  # flat triangle index list
    bones: list  # Bone
    extra: list  # (x,y,z,w) - field13, purpose unclear, preserved
    uv1: list  # (u,v) - field14
    uv2: list  # (u,v) - field15
    uv3: list  # (u,v) - field16

    @property
    def vertex_count(self):
        return len(self.positions)

    @property
    def triangle_count(self):
        return len(self.indices) // 3

    def compute_bbox(self):
        if not self.positions:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        zs = [p[2] for p in self.positions]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


@dataclass
class MeshFile:
    version: int
    unknown3: int
    lods: list  # Lod
    provenance: bytes = b""  # field1, optional - seen holding the original author's local file path
    trailing: list = field(default_factory=list)  # every other top-level field (6, 7, 20, and
    # anything not yet catalogued, e.g. a lone field16 seen on one sample), as
    # (field_no, wire_type, raw_value) tuples in original order. Fields 6/7
    # (bbox) get their value substituted with the freshly recomputed bbox on
    # encode; everything else is replayed byte-for-byte untouched - the same
    # "never corrupt what we don't understand" approach material_codec.py
    # uses for unrecognised .material fields.

    def compute_bbox(self):
        # Observed convention: the file-level bbox simply mirrors LOD 0's own
        # bbox (not a union across LODs) - verified byte-exact against every
        # sample file that has more than one LOD.
        if not self.lods:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        return self.lods[0].compute_bbox()


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def _decode_vec3(buf: bytes):
    if not buf:
        return (0.0, 0.0, 0.0)
    f = group_fields(buf)
    return (f.get(1, [0.0])[0], f.get(2, [0.0])[0], f.get(3, [0.0])[0])


def _decode_material_range(buf: bytes) -> MaterialRange:
    f = group_fields(buf)
    name = f[1][0].decode("utf-8")
    start = f.get(2, [0])[0]
    count = f.get(3, [0])[0]
    path = f[4][0].decode("utf-8") if 4 in f else None
    return MaterialRange(name=name, start=start, count=count, path=path)


def _decode_matrix(buf: bytes):
    f = group_fields(buf)
    flat = [f.get(i, [0.0])[0] for i in range(1, 17)]
    return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]


def _decode_bone(buf: bytes) -> Bone:
    f = group_fields(buf)
    name = f[1][0].decode("utf-8")
    parent = f[2][0].decode("utf-8") if 2 in f else None
    matrix_buf = f.get(3, [b""])[0]
    matrix = _decode_matrix(matrix_buf) if matrix_buf else [
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ]
    return Bone(name=name, parent=parent, matrix=matrix, raw=buf)


def _decode_lod(buf: bytes, index: int) -> Lod:
    f = group_fields(buf)

    distance = f.get(3, [0.0])[0]
    materials = [_decode_material_range(b) for b in f.get(4, [])]

    positions = chunk(packed_floats(f[5][0]), 3) if 5 in f else []
    normals = chunk(packed_floats(f[6][0]), 3) if 6 in f else []
    uv0 = chunk(packed_floats(f[7][0]), 2) if 7 in f else []
    tangents = chunk(packed_floats(f[8][0]), 4) if 8 in f else []
    bone_weights = chunk(packed_floats(f[9][0]), 4) if 9 in f else []
    bone_indices_f = chunk(packed_floats(f[10][0]), 4) if 10 in f else []
    bone_indices = [tuple(int(round(x)) for x in v) for v in bone_indices_f]

    indices = packed_varints(f[11][0]) if 11 in f else []
    bones = [_decode_bone(b) for b in f.get(12, [])]

    extra = chunk(packed_floats(f[13][0]), 4) if 13 in f else []
    uv1 = chunk(packed_floats(f[14][0]), 2) if 14 in f else []
    uv2 = chunk(packed_floats(f[15][0]), 2) if 15 in f else []
    uv3 = chunk(packed_floats(f[16][0]), 2) if 16 in f else []

    return Lod(
        index=index, distance=distance, materials=materials,
        positions=positions, normals=normals, uv0=uv0, tangents=tangents,
        bone_weights=bone_weights, bone_indices=bone_indices, indices=indices,
        bones=bones, extra=extra, uv1=uv1, uv2=uv2, uv3=uv3,
    )


def decode_mesh_file(buf: bytes) -> MeshFile:
    version = 0
    unknown3 = 0
    provenance = b""
    lods = []
    trailing = []

    for field_no, wire_type, val, _pos in iter_fields(buf):
        if field_no == 1:
            provenance = val
        elif field_no == 2:
            version = val
        elif field_no == 3:
            unknown3 = val
        elif field_no == 5:
            lods.append(_decode_lod(val, len(lods)))
        else:
            trailing.append((field_no, wire_type, val))

    return MeshFile(version=version, unknown3=unknown3, lods=lods, provenance=provenance, trailing=trailing)


def load_mesh_file(path: str) -> MeshFile:
    with open(path, "rb") as fh:
        return decode_mesh_file(fh.read())


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def _encode_vec3(v) -> bytes:
    x, y, z = v
    return (
        encode_fixed32_field(1, x)
        + encode_fixed32_field(2, y)
        + encode_fixed32_field(3, z)
    )


def _encode_material_range(mr: MaterialRange) -> bytes:
    payload = (
        encode_string_field(1, mr.name)
        + encode_varint_field(2, mr.start)
        + encode_varint_field(3, mr.count, always=True)
        + encode_string_field(4, mr.path)
    )
    return encode_bytes_field(4, payload, always=True)


def _encode_matrix(matrix) -> bytes:
    out = bytearray()
    for field_no, v in enumerate((x for row in matrix for x in row), start=1):
        out += encode_fixed32_field(field_no, v)
    return bytes(out)


def _encode_bone(bone: Bone) -> bytes:
    if bone.raw is not None:
        # Untouched since decode - replay the exact original bytes (see
        # Bone.raw docstring for why this is preferred over re-deriving).
        return encode_bytes_field(12, bone.raw, always=True)
    payload = (
        encode_string_field(1, bone.name)
        + encode_string_field(2, bone.parent)
        + encode_bytes_field(3, _encode_matrix(bone.matrix))
    )
    return encode_bytes_field(12, payload, always=True)


def encode_lod(lod: Lod) -> bytes:
    out = bytearray()
    out += encode_varint_field(1, 1, always=True)
    out += encode_fixed32_field(3, lod.distance)
    for mr in lod.materials:
        out += _encode_material_range(mr)

    out += encode_packed_float32_field(5, flatten(lod.positions))
    out += encode_packed_float32_field(6, flatten(lod.normals))
    out += encode_packed_float32_field(7, flatten(lod.uv0))
    out += encode_packed_float32_field(8, flatten(lod.tangents))
    out += encode_packed_float32_field(9, flatten(lod.bone_weights))
    out += encode_packed_float32_field(10, [float(x) for v in lod.bone_indices for x in v])
    out += encode_packed_varint_field(11, lod.indices)

    for bone in lod.bones:
        out += _encode_bone(bone)

    out += encode_packed_float32_field(13, flatten(lod.extra))
    out += encode_packed_float32_field(14, flatten(lod.uv1))
    out += encode_packed_float32_field(15, flatten(lod.uv2))
    out += encode_packed_float32_field(16, flatten(lod.uv3))

    bbox_min, bbox_max = lod.compute_bbox()
    out += encode_bytes_field(17, _encode_vec3(bbox_min))
    out += encode_bytes_field(18, _encode_vec3(bbox_max))

    return bytes(out)


def _reencode_raw_field(field_no: int, wire_type: int, val) -> bytes:
    """Replays one top-level field exactly as `iter_fields` decoded it -
    used for anything we don't have a dedicated model for (see
    MeshFile.trailing)."""
    if wire_type == 0:
        return encode_tag(field_no, 0) + write_varint(val)
    if wire_type == 2:
        return encode_bytes_field(field_no, val, always=True)
    if wire_type == 5:
        return encode_tag(field_no, 5) + struct.pack("<f", val)
    if wire_type == 1:
        return encode_tag(field_no, 1) + struct.pack("<d", val)
    raise ValueError(f"unsupported wire type {wire_type} for field {field_no}")


def encode_mesh_file(mf: MeshFile) -> bytes:
    out = bytearray()
    if mf.provenance:
        out += encode_bytes_field(1, mf.provenance, always=True)
    out += encode_varint_field(2, mf.version, always=True)
    out += encode_varint_field(3, mf.unknown3, always=True)
    for lod in mf.lods:
        out += encode_bytes_field(5, encode_lod(lod), always=True)

    bbox_min, bbox_max = mf.compute_bbox()
    for field_no, wire_type, val in mf.trailing:
        if field_no == 6:
            out += encode_bytes_field(6, _encode_vec3(bbox_min), always=True)
        elif field_no == 7:
            out += encode_bytes_field(7, _encode_vec3(bbox_max), always=True)
        else:
            out += _reencode_raw_field(field_no, wire_type, val)
    return bytes(out)


def save_mesh_file(mf: MeshFile, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(encode_mesh_file(mf))


# ---------------------------------------------------------------------------
# self-check (run directly: `python mesh_codec.py somefile.mesh`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    original = open(path, "rb").read()
    mf = decode_mesh_file(original)
    rebuilt = encode_mesh_file(mf)

    print(f"version={mf.version} unknown3={mf.unknown3} lods={len(mf.lods)}")
    for lod in mf.lods:
        print(f"  LOD {lod.index}: dist={lod.distance} verts={lod.vertex_count} "
              f"tris={lod.triangle_count} materials={len(lod.materials)} bones={len(lod.bones)}")

    print(f"original size={len(original)} rebuilt size={len(rebuilt)}")
    if original == rebuilt:
        print("ROUND-TRIP BYTE-IDENTICAL")
    else:
        print("round-trip differs at byte level - checking structural equivalence...")
        mf2 = decode_mesh_file(rebuilt)
        print("structurally equal:", mf == mf2)
        for i, (a, b) in enumerate(zip(original, rebuilt)):
            if a != b:
                print(f"first differing byte at offset {i}: original={a:#x} rebuilt={b:#x}")
                break
