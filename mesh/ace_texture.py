#!/usr/bin/env python3
"""
ace_texture.py - Reverse-engineered converter for Assetto Corsa EVO's
proprietary .texture / .texturemips format.

Usage (mimics acedit-texture.exe drag-and-drop behaviour):

    python ace_texture.py somefile.dds        -> writes somefile.texture + somefile.texturemips
    python ace_texture.py somefile.png         -> writes somefile.texture + somefile.texturemips
                                                   (uncompressed RGBA8, single mip - see NOTES)
    python ace_texture.py somefile.texture     -> writes somefile.dds
                                                   (needs somefile.texturemips next to it)

"""
import struct
import sys
import os
import io

# --------------------------------------------------------------------------
# protobuf-ish varint helpers (this is hand-rolled wire-format-compatible
# protobuf, not a real .proto - confirmed by disassembling protoReader /
# protoWriter: readVarint/writeVarint and tag = (field<<3)|wiretype are
# textbook proto3, including the "omit zero value" behaviour).
# --------------------------------------------------------------------------

def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def write_varint(value):
    out = bytearray()
    v = value & 0xFFFFFFFFFFFFFFFF
    while v >= 0x80:
        out.append((v & 0x7f) | 0x80)
        v >>= 7
    out.append(v & 0x7f)
    return bytes(out)


def write_tag(field_no, wire_type):
    return write_varint((field_no << 3) | wire_type)


def write_int(field_no, value):
    if value == 0:
        return b''
    return write_tag(field_no, 0) + write_varint(value)


def write_int_always(field_no, value):
    return write_tag(field_no, 0) + write_varint(value if value > 0 else 1)


def write_bool(field_no, value):
    if not value:
        return b''
    return write_tag(field_no, 0) + write_varint(1)


def write_bool_always(field_no, value):
    return write_tag(field_no, 0) + write_varint(1 if value else 0)


def write_bytes_field(field_no, data):
    if not data:
        return b''
    return write_tag(field_no, 2) + write_varint(len(data)) + data


def write_fixed32(field_no, f):
    return write_tag(field_no, 5) + struct.pack('<f', f)


def write_packed_varint(field_no, values):
    if not values:
        return b''
    body = b''.join(write_varint(v) for v in values)
    return write_tag(field_no, 2) + write_varint(len(body)) + body


def decode_message(data):
    """Generic decode: {field_no: value}. Repeated/bytes fields keep last;
    good enough for our purposes since we know the schema."""
    pos = 0
    fields = {}
    while pos < len(data):
        tag, pos = read_varint(data, pos)
        fn = tag >> 3
        wt = tag & 7
        if wt == 0:
            v, pos = read_varint(data, pos)
            fields[fn] = v
        elif wt == 2:
            ln, pos = read_varint(data, pos)
            fields[fn] = data[pos:pos + ln]
            pos += ln
        elif wt == 5:
            fields[fn] = struct.unpack_from('<I', data, pos)[0]
            pos += 4
        elif wt == 1:
            fields[fn] = struct.unpack_from('<Q', data, pos)[0]
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wt} at {pos}")
    return fields


# --------------------------------------------------------------------------
# Pixel format table.
#
# "kunos" is the enum value stored in Metadata field 4 (kstexture.PixelFormat).
# Empirically measured by round-tripping known-DXGI-format DDS files through
# the real acedit-texture.exe and reading back the resulting .texture. Note
# BC4_UNORM/SNORM (and BC5_UNORM/SNORM) measured to share the same kunos
# value - the game's enum doesn't seem to distinguish signedness there, so
# the reverse (kunos -> dxgi) lookup always picks the UNORM variant.
#
#   DXGI_FORMAT_R8G8B8A8_UNORM       (28) -> kunos  2
#   DXGI_FORMAT_B8G8R8A8_UNORM       (87) -> kunos  2
#   DXGI_FORMAT_R8G8B8A8_UNORM_SRGB  (29) -> kunos  3
#   DXGI_FORMAT_B8G8R8A8_UNORM_SRGB  (91) -> kunos  3
#   DXGI_FORMAT_BC1_UNORM             (71) -> kunos 12
#   DXGI_FORMAT_BC1_UNORM_SRGB        (72) -> kunos 13
#   DXGI_FORMAT_BC2_UNORM             (74) -> kunos 14
#   DXGI_FORMAT_BC2_UNORM_SRGB        (75) -> kunos 15
#   DXGI_FORMAT_BC3_UNORM             (77) -> kunos 16
#   DXGI_FORMAT_BC3_UNORM_SRGB        (78) -> kunos 17
#   DXGI_FORMAT_BC6H_UF16             (95) -> kunos 26
#   DXGI_FORMAT_BC4_UNORM/SNORM   (80/81) -> kunos 27
#   DXGI_FORMAT_BC6H_SF16             (96) -> kunos 32
#   DXGI_FORMAT_BC7_UNORM             (98) -> kunos 33
#   DXGI_FORMAT_BC7_UNORM_SRGB        (99) -> kunos 34
#   DXGI_FORMAT_BC5_UNORM/SNORM   (83/84) -> kunos 35
#
# NOT supported yet (add following the same probing method if you need
# them): R8_UNORM/R8G8_UNORM (measured but results looked unreliable -
# collapsed to the same kunos as RGBA8, needs more investigation before
# trusting it), R8G8B8A8_SNORM (the real exe itself rejected it), any
# array/cubemap texture (array_size > 1 untested).
#
# bytes_per_unit / block_texels describe the DDS side (standard D3D block
# layout): compressed formats consume `bytes_per_unit` bytes per 4x4 texel
# block; uncompressed formats consume `bytes_per_unit` bytes per texel.
# `compressed=True` formats are the only ones that go through the 64KiB
# tiling scheme (see tile_mip_blocks below) - measured directly: an
# RGBA8_UNORM roundtrip produced a plain concatenated .texturemips with *no*
# padding at all, while every BC format produced 64KiB-tile-aligned output.
# --------------------------------------------------------------------------

class Fmt:
    def __init__(self, dxgi, kunos, bytes_per_unit, compressed, name):
        self.dxgi = dxgi
        self.kunos = kunos
        self.bytes_per_unit = bytes_per_unit
        self.compressed = compressed
        self.name = name


FORMATS = [
    # uncompressed - B8G8R8A8 measured to map to the same kunos value as
    # R8G8B8A8 (game seems to not distinguish channel order in this enum).
    # image_to_texture() (below) always writes kunos=2 with plain (R,G,B,A)
    # byte order, and files written that way read back with correct colours
    # in-game - so R8G8B8A8 has to be the one that wins the kunos->dxgi
    # reverse lookup (FORMATS_BY_KUNOS, used by texture_to_dds() whenever we
    # decode our own output for a Blender preview). Getting this backwards
    # tags the DDS as BGRA on decode only, which Blender's loader then
    # renders with red and blue swapped - correct in-game, blue-tinted in
    # Blender - without the file's actual bytes ever being wrong. B8G8R8A8
    # is listed first so R8G8B8A8 stays the one defined last, same
    # "last one wins" convention used below for BC4/BC5.
    Fmt(87, 2, 4, False, 'B8G8R8A8_UNORM'),
    Fmt(28, 2, 4, False, 'R8G8B8A8_UNORM'),
    Fmt(29, 3, 4, False, 'R8G8B8A8_UNORM_SRGB'),
    Fmt(91, 3, 4, False, 'B8G8R8A8_UNORM_SRGB'),
    # BC1 (DXT1)
    Fmt(71, 12, 8, True, 'BC1_UNORM'),
    Fmt(72, 13, 8, True, 'BC1_UNORM_SRGB'),
    # BC2 (DXT3)
    Fmt(74, 14, 16, True, 'BC2_UNORM'),
    Fmt(75, 15, 16, True, 'BC2_UNORM_SRGB'),
    # BC3 (DXT5) - typical for normal maps stored as DXT5nm (X=alpha, Y=green)
    Fmt(77, 16, 16, True, 'BC3_UNORM'),
    Fmt(78, 17, 16, True, 'BC3_UNORM_SRGB'),
    # BC4/BC5 (single/dual channel, roughness/metalness/normal maps) - SNORM
    # and UNORM measured to share the same kunos enum value; listed first so
    # the UNORM entry below wins the kunos->dxgi reverse lookup (more common)
    Fmt(81, 27, 8, True, 'BC4_SNORM'),
    Fmt(80, 27, 8, True, 'BC4_UNORM'),
    Fmt(84, 35, 16, True, 'BC5_SNORM'),
    Fmt(83, 35, 16, True, 'BC5_UNORM'),
    # BC6H (HDR)
    Fmt(95, 26, 16, True, 'BC6H_UF16'),
    Fmt(96, 32, 16, True, 'BC6H_SF16'),
    # BC7
    Fmt(98, 33, 16, True, 'BC7_UNORM'),
    Fmt(99, 34, 16, True, 'BC7_UNORM_SRGB'),
]
FORMATS_BY_DXGI = {f.dxgi: f for f in FORMATS}
FORMATS_BY_KUNOS = {f.kunos: f for f in FORMATS}

# Standard tile shape, measured directly (matches the D3D12 "standard tile
# shape" convention for block-compressed formats): a tile is always exactly
# 65536 bytes; its texel footprint depends only on bytes-per-block.
TILE_BYTES = 65536


def tile_shape_texels(bytes_per_block):
    if bytes_per_block == 8:
        return 512, 256
    elif bytes_per_block == 16:
        return 256, 256
    else:
        raise ValueError(f"no known tile shape for {bytes_per_block}-byte blocks")


# --------------------------------------------------------------------------
# DDS reading / writing (DX10 header only - what acedit-texture.exe itself
# produces; legacy FourCC headers for BC1/2/3 are also accepted on read).
# --------------------------------------------------------------------------

DDS_MAGIC = b'DDS '
FOURCC_LEGACY = {
    b'DXT1': 71,  # BC1_UNORM
    b'DXT3': 74,  # BC2_UNORM
    b'DXT5': 77,  # BC3_UNORM
}

# Legacy (pre-DX10, pre-FourCC) uncompressed DDS: DDPF_RGB (0x40) set instead
# of DDPF_FOURCC (0x4), pixel format described by a bit count + R/G/B/A bit
# masks rather than a DXGI enum. Common from older export tools (Photoshop,
# paint.net, NVIDIA legacy plugin, ...) - e.g. a 32bpp export with masks
# R=0x00ff0000 G=0x0000ff00 B=0x000000ff A=0xff000000 is plain B8G8R8A8_UNORM
# byte order in memory, just spelled out the old way instead of DXGI_FORMAT 87.
DDPF_RGB = 0x40
_LEGACY_RGB_MASKS = {
    # (bit_count, r_mask, g_mask, b_mask, a_mask) -> dxgi (must exist in FORMATS)
    (32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000): 87,  # B8G8R8A8_UNORM
    (32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000): 28,  # R8G8B8A8_UNORM
}


class DDSImage:
    def __init__(self, width, height, dxgi_format, mips):
        self.width = width
        self.height = height
        self.dxgi_format = dxgi_format
        self.mips = mips  # list of raw bytes, one per mip level, mip0 first


class DDSHeaderInfo:
    """Lightweight, format-agnostic parse: always succeeds on any valid DDS,
    even one whose pixel format isn't in FORMATS_BY_DXGI yet - lets the GUI
    show width/height/dxgi-format-number instead of just failing."""
    def __init__(self, width, height, mipmapcount, dxgi_format):
        self.width = width
        self.height = height
        self.mipmapcount = mipmapcount
        self.dxgi_format = dxgi_format


def read_dds_header(data):
    if data[0:4] != DDS_MAGIC:
        raise ValueError("not a DDS file")
    header_size = struct.unpack_from('<I', data, 4)[0]
    assert header_size == 124
    height = struct.unpack_from('<I', data, 12)[0]
    width = struct.unpack_from('<I', data, 16)[0]
    mipmapcount = struct.unpack_from('<I', data, 28)[0] or 1
    pf_flags = struct.unpack_from('<I', data, 80)[0]
    fourcc = data[84:88]
    if pf_flags & 0x4 and fourcc == b'DX10':
        dxgi_format = struct.unpack_from('<I', data, 128)[0]
    elif fourcc in FOURCC_LEGACY:
        dxgi_format = FOURCC_LEGACY[fourcc]
    elif pf_flags & DDPF_RGB:
        rgb_bit_count = struct.unpack_from('<I', data, 88)[0]
        r_mask, g_mask, b_mask, a_mask = struct.unpack_from('<4I', data, 92)
        key = (rgb_bit_count, r_mask, g_mask, b_mask, a_mask)
        dxgi_format = _LEGACY_RGB_MASKS.get(key)
        if dxgi_format is None:
            raise ValueError(
                f"format DDS non compresse non reconnu (bits={rgb_bit_count}, "
                f"masques R={r_mask:#x} G={g_mask:#x} B={b_mask:#x} A={a_mask:#x}) - "
                f"ajoute-le a _LEGACY_RGB_MASKS dans ace_texture.py"
            )
    else:
        raise ValueError(f"unsupported DDS pixel format (fourcc={fourcc!r})")
    return DDSHeaderInfo(width, height, mipmapcount, dxgi_format)


def read_dds(data):
    hdr = read_dds_header(data)
    width, height, mipmapcount, dxgi_format = hdr.width, hdr.height, hdr.mipmapcount, hdr.dxgi_format
    off = 148 if data[84:88] == b'DX10' else 128

    fmt = FORMATS_BY_DXGI.get(dxgi_format)
    if fmt is None:
        raise ValueError(
            f"format DXGI {dxgi_format} non supporte - ajoute-le a FORMATS dans "
            f"ace_texture.py (voir NOTES en bas de fichier pour la methode)")

    mips = []
    w, h = width, height
    for _ in range(mipmapcount):
        if fmt.compressed:
            bw = max(1, (w + 3) // 4)
            bh = max(1, (h + 3) // 4)
            nbytes = bw * bh * fmt.bytes_per_unit
        else:
            nbytes = w * h * fmt.bytes_per_unit
        mips.append(data[off:off + nbytes])
        off += nbytes
        w = max(1, w // 2)
        h = max(1, h // 2)

    img = DDSImage(width, height, dxgi_format, mips)
    return img


def write_dds(img: DDSImage):
    fmt = FORMATS_BY_DXGI[img.dxgi_format]
    mipcount = len(img.mips)
    pitch = len(img.mips[0])
    header = bytearray(128)
    struct.pack_into('<4s', header, 0, b'DDS ')
    struct.pack_into('<I', header, 4, 124)
    flags = 0x1007 | 0x20000 | (0x80000 if fmt.compressed else 0x8)
    struct.pack_into('<I', header, 8, flags)
    struct.pack_into('<I', header, 12, img.height)
    struct.pack_into('<I', header, 16, img.width)
    struct.pack_into('<I', header, 20, pitch)
    struct.pack_into('<I', header, 28, mipcount)
    struct.pack_into('<I', header, 76, 32)
    struct.pack_into('<I', header, 80, 0x4)
    struct.pack_into('<4s', header, 84, b'DX10')
    caps = 0x401008 if mipcount > 1 else 0x1000
    struct.pack_into('<I', header, 108, caps)
    dx10 = struct.pack('<5I', img.dxgi_format, 3, 0, 1, 0)
    return bytes(header) + dx10 + b''.join(img.mips)


# --------------------------------------------------------------------------
# Tiling / detiling - fully reverse engineered and validated byte-for-byte
# against a real 4096x4096 game asset (ext_skin_c.texture/.texturemips).
#
# Algorithm, per mip level, independently:
#   1. tile_w, tile_h = tile shape in blocks (see tile_shape_texels/4)
#   2. lay the mip's tiles out in row-major order (all tile-columns of a
#      tile-row before moving to the next tile-row)
#   3. each tile is exactly TILE_BYTES; blocks beyond the real mip
#      width/height (right/bottom edge tiles, or a mip smaller than one
#      tile) are zero-padded, not wrapped/repeated
#   4. after a mip's tiles, the next mip starts at the next tile boundary
#      (which is automatic since every tile is a fixed TILE_BYTES)
#
# Uncompressed formats are NOT tiled at all: mips are simply concatenated
# with no padding (measured directly).
# --------------------------------------------------------------------------

def tile_mip(block_data, blocks_w, blocks_h, bytes_per_block, tile_blocks_w, tile_blocks_h):
    out = bytearray()
    num_tile_cols = (blocks_w + tile_blocks_w - 1) // tile_blocks_w
    num_tile_rows = (blocks_h + tile_blocks_h - 1) // tile_blocks_h
    row_stride = blocks_w * bytes_per_block
    tile_full_bytes = tile_blocks_w * tile_blocks_h * bytes_per_block
    for tr in range(num_tile_rows):
        for tc in range(num_tile_cols):
            tile_buf = bytearray(tile_full_bytes)
            for lr in range(tile_blocks_h):
                grow = tr * tile_blocks_h + lr
                if grow >= blocks_h:
                    break
                gcol0 = tc * tile_blocks_w
                ncols = min(tile_blocks_w, blocks_w - gcol0)
                src_off = grow * row_stride + gcol0 * bytes_per_block
                dst_off = lr * tile_blocks_w * bytes_per_block
                nbytes = ncols * bytes_per_block
                tile_buf[dst_off:dst_off + nbytes] = block_data[src_off:src_off + nbytes]
            out += tile_buf
            if len(tile_buf) < TILE_BYTES:
                out += b'\x00' * (TILE_BYTES - len(tile_buf))
    return bytes(out)


def detile_mip(tiled_data, offset, blocks_w, blocks_h, bytes_per_block, tile_blocks_w, tile_blocks_h):
    num_tile_cols = (blocks_w + tile_blocks_w - 1) // tile_blocks_w
    num_tile_rows = (blocks_h + tile_blocks_h - 1) // tile_blocks_h
    row_stride = blocks_w * bytes_per_block
    out = bytearray(blocks_h * row_stride)
    pos = offset
    for tr in range(num_tile_rows):
        for tc in range(num_tile_cols):
            for lr in range(tile_blocks_h):
                grow = tr * tile_blocks_h + lr
                if grow >= blocks_h:
                    continue
                gcol0 = tc * tile_blocks_w
                ncols = min(tile_blocks_w, blocks_w - gcol0)
                src_off = pos + lr * tile_blocks_w * bytes_per_block
                dst_off = grow * row_stride + gcol0 * bytes_per_block
                nbytes = ncols * bytes_per_block
                out[dst_off:dst_off + nbytes] = tiled_data[src_off:src_off + nbytes]
            pos += tile_blocks_w * tile_blocks_h * bytes_per_block
    consumed_tiles = num_tile_rows * num_tile_cols
    new_offset = offset + consumed_tiles * TILE_BYTES
    return bytes(out), new_offset


# --------------------------------------------------------------------------
# Metadata (.texture file) encode/decode.
#
# Field numbers, reverse engineered from disassembling
# kstexture.EncodeKSMetadata / ParseKSMetadata (Go struct field offsets ->
# writeInt/writeBool/writeBytes call sites with immediate field numbers):
#
#   1  Width          (varint)
#   2  Height          (varint)
#   3  MipCount        (varint)
#   4  PixelFormat     (varint, kunos enum - see FORMATS table)
#   6  ArraySize       (varint, always written, clamped to >=1)
#   7  bool flag       (always 0 in every sample seen - possibly IsCubemap)
#   8  bool flag       (always 0 in every sample seen - possibly IsNormalMap)
#   10 ConversionSettings (bytes, nested msg) - OPTIONAL provenance info
#      (source file path, source width/height...) only present on assets
#      built through Kunos' internal editor, never emitted by
#      acedit-texture.exe itself. Not required for the game to load the
#      texture (round-trip tested without it).
#   11 bytes (nested msg, small) - encoder settings blob. acedit-texture.exe
#      always writes {field1: 1}, but real Editor-authored files show a
#      value tied to the pixel FORMAT, 100% consistent across dozens of real
#      samples (see FIELD11_SETTING_BY_KUNOS): 8 for BC1_UNORM_SRGB, 2 for
#      BC4_UNORM, 9 for BC5_UNORM, absent entirely for BC7_UNORM_SRGB. Also
#      always carries field9=1 and field11=0.5 (fixed32) when field10
#      (ConversionSettings) is present. Writing the wrong value here (or
#      acedit-texture.exe's generic {1:1}) for BC4/BC5 specifically has been
#      linked to in-game crashes on load - always match the real per-format
#      value when known.
#   12 TilingInfo (bytes, nested msg) - only present for block-compressed
#      formats:
#        1 TileWidthInTexels
#        2 TileHeightInTexels
#        3 constant 1
#        4 packed bytes, one per mip (seen as 0,1,2,...) - purpose unclear,
#          not required for correct decode, reproduced for fidelity
#        5 packed bytes, one per mip, all 1s in every sample seen
#        7 packed varint, one per 64KiB tile used, constant value 1024
#   13 varint, 1 when TilingInfo is present
# --------------------------------------------------------------------------

class Metadata:
    def __init__(self, width, height, mipcount, kunos_format, array_size=1):
        self.width = width
        self.height = height
        self.mipcount = mipcount
        self.kunos_format = kunos_format
        self.array_size = array_size


# field11.field1, keyed by kunos pixel format - measured across 100+ real
# Editor-authored .texture files (ks_citroen_ra), zero exceptions per format:
#   BC1_UNORM_SRGB (13): always 8      (54 files)
#   BC4_UNORM      (27): always 2      (18 files)
#   BC5_UNORM      (35): always 9      (18 files)
#   BC7_UNORM_SRGB (34): field1 absent entirely (25 files) - do not write it
# Formats not listed here: unconfirmed, field1 is omitted (safer than
# guessing wrong - matches the BC7 behaviour of "when unsure, omit").
FIELD11_SETTING_BY_KUNOS = {
    13: 8,   # BC1_UNORM_SRGB
    26: 3,   # BC6H_UF16   - measured on 73/73 normal maps in real car content
    27: 2,   # BC4_UNORM
    35: 9,   # BC5_UNORM
}

# Formats whose field11 is just {1: <value>} with nothing else - no field9/
# field11-float pair, since that pair only ever accompanies the
# ConversionSettings blob (field 10) which this writer never emits.
# Measured on every uncompressed texture found in real car content (4/4,
# e.g. EXT_Lights_F_1.texture) and matching a reference file produced by an
# external converter byte-for-byte.
FIELD11_SIMPLE_BY_KUNOS = {
    2: 1,    # R8G8B8A8_UNORM / B8G8R8A8_UNORM
}


def encode_metadata(meta: Metadata, tile_counts_per_mip=None):
    """tile_counts_per_mip: list with one entry per mip level, giving the
    number of 64KiB tiles that mip occupies (only meaningful/required for
    block-compressed formats). Required for TilingInfo fields 4 and 5:

      field 4 = cumulative tile START index per mip (running sum of
                tile_counts_per_mip, i.e. field4[i] = sum(counts[:i]))
      field 5 = tile_counts_per_mip itself (tiles used by mip i)

    Both reverse engineered from a real BC7 4096x4096 asset - a texture
    where mip0 alone spans 256 tiles. An earlier version of this function
    wrote the plain mip index (0,1,2,...) here instead of real tile
    offsets/counts; that only happened to look right for small/synthetic
    textures where every mip fit in exactly one tile (index == offset by
    coincidence), and produced a black/corrupted texture in-game for any
    texture whose mip0 needs more than one tile - i.e. anything bigger than
    512px for 8-byte-block formats (BC1/BC4) or 256px for 16-byte-block
    formats (BC3/BC5/BC7).
    """
    out = bytearray()
    out += write_int(1, meta.width)
    out += write_int(2, meta.height)
    out += write_int(3, meta.mipcount)
    out += write_int(4, meta.kunos_format)
    out += write_int_always(6, meta.array_size)
    out += write_bool_always(7, False)
    out += write_bool_always(8, False)

    if meta.kunos_format in FIELD11_SIMPLE_BY_KUNOS:
        out += write_bytes_field(11, write_int(1, FIELD11_SIMPLE_BY_KUNOS[meta.kunos_format]))
    else:
        setting11 = bytearray()
        if meta.kunos_format in FIELD11_SETTING_BY_KUNOS:
            setting11 += write_int(1, FIELD11_SETTING_BY_KUNOS[meta.kunos_format])
        setting11 += write_int(9, 1)
        setting11 += write_fixed32(11, 0.5)
        out += write_bytes_field(11, bytes(setting11))

    fmt = FORMATS_BY_KUNOS[meta.kunos_format]
    if fmt.compressed:
        assert tile_counts_per_mip and len(tile_counts_per_mip) == meta.mipcount
        tile_w_tex, tile_h_tex = tile_shape_texels(fmt.bytes_per_unit)
        cumulative = []
        running = 0
        for c in tile_counts_per_mip:
            cumulative.append(running)
            running += c
        total_tiles = running

        tiling = bytearray()
        tiling += write_int(1, tile_w_tex)
        tiling += write_int(2, tile_h_tex)
        tiling += write_int(3, 1)
        tiling += write_packed_varint(4, cumulative)
        tiling += write_packed_varint(5, tile_counts_per_mip)
        tiling += write_packed_varint(7, [1024] * total_tiles)
        out += write_bytes_field(12, bytes(tiling))
        out += write_int(13, 1)
    return bytes(out)


def decode_metadata(data):
    fields = decode_message(data)
    meta = Metadata(
        width=fields.get(1, 0),
        height=fields.get(2, 0),
        mipcount=fields.get(3, 1),
        kunos_format=fields.get(4, 0),
        array_size=fields.get(6, 1),
    )
    return meta


# --------------------------------------------------------------------------
# High level conversion functions
# --------------------------------------------------------------------------

def dds_to_texture(dds_bytes):
    img = read_dds(dds_bytes)
    fmt = FORMATS_BY_DXGI[img.dxgi_format]
    mipcount = len(img.mips)

    texturemips = bytearray()
    tile_counts_per_mip = []
    if fmt.compressed:
        tile_w_tex, tile_h_tex = tile_shape_texels(fmt.bytes_per_unit)
        tile_blocks_w = tile_w_tex // 4
        tile_blocks_h = tile_h_tex // 4
        w, h = img.width, img.height
        for mip in img.mips:
            bw = max(1, (w + 3) // 4)
            bh = max(1, (h + 3) // 4)
            tiled = tile_mip(mip, bw, bh, fmt.bytes_per_unit, tile_blocks_w, tile_blocks_h)
            texturemips += tiled
            tile_counts_per_mip.append(len(tiled) // TILE_BYTES)
            w = max(1, w // 2)
            h = max(1, h // 2)
    else:
        for mip in img.mips:
            texturemips += mip

    meta = Metadata(img.width, img.height, mipcount, fmt.kunos, array_size=1)
    texture = encode_metadata(meta, tile_counts_per_mip=tile_counts_per_mip if fmt.compressed else None)
    return texture, bytes(texturemips)


def texture_to_dds(texture_bytes, texturemips_bytes):
    meta = decode_metadata(texture_bytes)
    fmt = FORMATS_BY_KUNOS.get(meta.kunos_format)
    if fmt is None:
        raise ValueError(f"unknown kunos pixel format {meta.kunos_format} - "
                          f"extend FORMATS in ace_texture.py")

    mips = []
    w, h = meta.width, meta.height
    offset = 0
    for _ in range(meta.mipcount):
        if fmt.compressed:
            tile_w_tex, tile_h_tex = tile_shape_texels(fmt.bytes_per_unit)
            tile_blocks_w = tile_w_tex // 4
            tile_blocks_h = tile_h_tex // 4
            bw = max(1, (w + 3) // 4)
            bh = max(1, (h + 3) // 4)
            mip_bytes, offset = detile_mip(
                texturemips_bytes, offset, bw, bh, fmt.bytes_per_unit,
                tile_blocks_w, tile_blocks_h)
        else:
            nbytes = w * h * fmt.bytes_per_unit
            mip_bytes = texturemips_bytes[offset:offset + nbytes]
            offset += nbytes
        mips.append(mip_bytes)
        w = max(1, w // 2)
        h = max(1, h // 2)

    img = DDSImage(meta.width, meta.height, fmt.dxgi, mips)
    return write_dds(img)


# --------------------------------------------------------------------------
# BC1 encoding from raw pixels + mip chain generation.
#
# Fills the gap listed under NOT IMPLEMENTED below: acedit-texture.exe only
# re-tiles data that already arrives compressed in a DDS, so producing a real
# game-ready compressed texture used to require an external compressor. The
# encoder here is a straightforward bounding-box BC1 fitter - not competitive
# with nvtt on photographic content, but effectively lossless on the flat,
# high-contrast function masks it is meant for (exactly the case the SDK calls
# out as "ideal for splatmaps and masks with clear contrasts and no
# gradients"). Output goes through the already byte-validated
# write_dds/dds_to_texture path rather than re-implementing tiling.
# --------------------------------------------------------------------------

def _rgb565_pack(c):
    import numpy as np
    r = (c[:, 0].astype(np.uint32) >> 3) & 0x1F
    g = (c[:, 1].astype(np.uint32) >> 2) & 0x3F
    b = (c[:, 2].astype(np.uint32) >> 3) & 0x1F
    return (r << 11) | (g << 5) | b


def _rgb565_unpack(v):
    import numpy as np
    r5 = (v >> 11) & 0x1F
    g6 = (v >> 5) & 0x3F
    b5 = v & 0x1F
    # replicate high bits into the low ones, exactly how the hardware expands
    r = (r5 << 3) | (r5 >> 2)
    g = (g6 << 2) | (g6 >> 4)
    b = (b5 << 3) | (b5 >> 2)
    return np.stack([r, g, b], axis=1).astype(np.int32)


def encode_bc1(rgb, width, height, chunk_blocks=16384):
    """rgb: uint8 array (height, width, 3). Returns raw BC1 blocks, row-major.

    Always emits the opaque 4-colour mode (colour0 > colour1 enforced): the
    3-colour mode's 4th entry is transparent black, and on the mostly-black
    masks this is used for that would silently punch holes wherever a block
    is uniform."""
    import numpy as np
    blocks_w = max(1, (width + 3) // 4)
    blocks_h = max(1, (height + 3) // 4)
    padded = np.zeros((blocks_h * 4, blocks_w * 4, 3), dtype=np.uint8)
    padded[:height, :width] = rgb[:height, :width]
    blocks = (padded.reshape(blocks_h, 4, blocks_w, 4, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(-1, 16, 3)
                    .astype(np.int32))

    n = blocks.shape[0]
    out = np.empty((n, 8), dtype=np.uint8)
    shifts = (2 * np.arange(16)).astype(np.uint32)

    for start in range(0, n, chunk_blocks):
        blk = blocks[start:start + chunk_blocks]
        c_hi = _rgb565_pack(blk.max(axis=1))
        c_lo = _rgb565_pack(blk.min(axis=1))
        c0 = np.maximum(c_hi, c_lo)
        c1 = np.minimum(c_hi, c_lo)
        # force c0 > c1 so the decoder never takes the 3-colour/alpha path
        same = c0 == c1
        c1 = np.where(same & (c1 > 0), c1 - 1, c1)
        c0 = np.where(c0 == c1, c0 + 1, c0)

        p0 = _rgb565_unpack(c0)
        p1 = _rgb565_unpack(c1)
        palette = np.stack([p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3], axis=1)

        diff = blk[:, :, None, :] - palette[:, None, :, :]
        idx = (diff * diff).sum(axis=3).argmin(axis=2).astype(np.uint32)
        bits = (idx << shifts[None, :]).sum(axis=1, dtype=np.uint32)

        sl = slice(start, start + blk.shape[0])
        out[sl, 0] = (c0 & 0xFF).astype(np.uint8)
        out[sl, 1] = ((c0 >> 8) & 0xFF).astype(np.uint8)
        out[sl, 2] = (c1 & 0xFF).astype(np.uint8)
        out[sl, 3] = ((c1 >> 8) & 0xFF).astype(np.uint8)
        out[sl, 4] = (bits & 0xFF).astype(np.uint8)
        out[sl, 5] = ((bits >> 8) & 0xFF).astype(np.uint8)
        out[sl, 6] = ((bits >> 16) & 0xFF).astype(np.uint8)
        out[sl, 7] = ((bits >> 24) & 0xFF).astype(np.uint8)
    return out.tobytes()


# --------------------------------------------------------------------------
# BC6H_UF16 encoding.
#
# Normal maps in real car content are BC6H_UF16 without exception (73/73
# sampled), so producing one is required to match the shipped pipeline.
#
# Only mode 10 is emitted - the single-subset, no-transform mode with four
# 10-bit absolute endpoint fields. That is the one BC6H mode with a plain
# sequential bit layout (5-bit mode + 6x10-bit endpoints + 63 index bits =
# exactly 128), so it avoids the scrambled per-mode bit tables the two-subset
# modes need, while still giving 10-bit endpoints and 16 interpolation steps
# per block. Quality is far above BC1 and adequate for normal maps; a full
# encoder choosing between all 14 modes would gain little here.
#
# The value pipeline follows the D3D BC6H spec: a texel's half-float bits are
# reached via finish_unquantize(x) = (x * 31) >> 6 on an interpolated 16-bit
# "unquantized" value, so encoding works backwards from half bits through
# that scale before quantising endpoints to 10 bits.
# --------------------------------------------------------------------------

_BC6H_WEIGHTS4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]


def _bc6h_unquantize10(comp):
    """Spec `unquantize` for an unsigned 10-bit endpoint field."""
    import numpy as np
    comp = comp.astype(np.int64)
    out = (comp * 64) + 32
    out = np.where(comp == 0, 0, out)
    out = np.where(comp >= 1023, 0xFFFF, out)
    return out


def _bc6h_fit_endpoints(blk, power_iters=6, refine_iters=2):
    """Endpoints for a batch of blocks, shaped (blocks, 16 texels, 3).

    A block's 16 colours are approximated by a single line, so the endpoints
    have to be the extremes along the line that best fits them - the principal
    axis. Using the per-channel bounding box instead (the obvious choice, and
    what a BC1 encoder can get away with) is a poor fit whenever a block's
    channels vary independently, which is exactly what a normal map does;
    measured against Blender's own BC6H decoder it left a 0.15 worst-case
    error where this leaves ~0.01.

    The axis comes from a few power iterations on the covariance (no eigen
    solver needed), then the endpoints are refined by least squares against
    the indices they produce."""
    import numpy as np
    values = blk.astype(np.float64)
    mean = values.mean(axis=1)
    centred = values - mean[:, None, :]

    direction = values.max(axis=1) - values.min(axis=1)  # bbox diagonal as the seed
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.where(norm > 0, direction / np.maximum(norm, 1e-12), np.array([1.0, 0.0, 0.0]))
    for _ in range(power_iters):
        proj = np.einsum("btc,bc->bt", centred, direction)
        direction = np.einsum("bt,btc->bc", proj, centred)
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        flat = (norm <= 1e-12).ravel()
        direction = direction / np.maximum(norm, 1e-12)
        if flat.any():
            direction[flat] = np.array([1.0, 0.0, 0.0])

    proj = np.einsum("btc,bc->bt", centred, direction)
    ep0 = mean + proj.min(axis=1)[:, None] * direction
    ep1 = mean + proj.max(axis=1)[:, None] * direction

    # Least-squares refit: with the weights each texel ends up using, solve
    # for the endpoint pair minimising the squared error, and repeat.
    steps = np.array(_BC6H_WEIGHTS4, dtype=np.float64) / 64.0
    for _ in range(refine_iters):
        span = ep1 - ep0
        denom = (span * span).sum(axis=1)
        t = np.where(denom[:, None] > 1e-12,
                     np.einsum("btc,bc->bt", values - ep0[:, None, :], span) / np.maximum(denom, 1e-12)[:, None],
                     0.0)
        w = steps[np.abs(t[:, :, None] - steps[None, None, :]).argmin(axis=2)]
        a = ((1.0 - w) ** 2).sum(axis=1)
        b = (w * (1.0 - w)).sum(axis=1)
        c = (w ** 2).sum(axis=1)
        det = a * c - b * b
        p = np.einsum("bt,btc->bc", 1.0 - w, values)
        q = np.einsum("bt,btc->bc", w, values)
        ok = np.abs(det) > 1e-9
        new0 = np.where(ok[:, None], (c[:, None] * p - b[:, None] * q) / np.where(ok, det, 1.0)[:, None], ep0)
        new1 = np.where(ok[:, None], (a[:, None] * q - b[:, None] * p) / np.where(ok, det, 1.0)[:, None], ep1)
        ep0, ep1 = new0, new1

    ep0 = np.clip(np.rint(ep0), 0, 0xFFFF).astype(np.int64)
    ep1 = np.clip(np.rint(ep1), 0, 0xFFFF).astype(np.int64)
    return ep0, ep1


def encode_bc6h_uf16(rgb, width, height, chunk_blocks=8192):
    """rgb: float array (h, w, 3) with values >= 0. Returns raw BC6H blocks."""
    import numpy as np
    blocks_w = max(1, (width + 3) // 4)
    blocks_h = max(1, (height + 3) // 4)
    padded = np.zeros((blocks_h * 4, blocks_w * 4, 3), dtype=np.float32)
    padded[:height, :width] = np.maximum(rgb[:height, :width], 0.0)

    # half-float bit patterns -> the spec's pre-finish_unquantize domain
    half_bits = padded.astype(np.float16).view(np.uint16).astype(np.int64)
    target = np.clip((half_bits * 64 + 15) // 31, 0, 0xFFFF)
    blocks = (target.reshape(blocks_h, 4, blocks_w, 4, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(-1, 16, 3))

    n = blocks.shape[0]
    out_lo = np.zeros(n, dtype=np.uint64)
    out_hi = np.zeros(n, dtype=np.uint64)
    weights = np.array(_BC6H_WEIGHTS4, dtype=np.int64)

    for start in range(0, n, chunk_blocks):
        blk = blocks[start:start + chunk_blocks]
        lo_end, hi_end = _bc6h_fit_endpoints(blk)
        # quantise to 10 bits (inverse of _bc6h_unquantize10)
        q0 = np.clip((lo_end - 32 + 32) // 64, 0, 1023)
        q1 = np.clip((hi_end - 32 + 32) // 64, 0, 1023)

        u0 = _bc6h_unquantize10(q0)
        u1 = _bc6h_unquantize10(q1)
        # palette in the unquantized domain, shape (blocks, 16 steps, 3)
        pal = ((u0[:, None, :] * (64 - weights[None, :, None])
                + u1[:, None, :] * weights[None, :, None] + 32) >> 6)

        diff = blk[:, :, None, :].astype(np.int64) - pal[:, None, :, :]
        idx = (diff * diff).sum(axis=3).argmin(axis=2).astype(np.int64)

        # The anchor index (texel 0) is stored in 3 bits, so its top bit must
        # be 0; if it isn't, swap the endpoints and mirror every index.
        flip = idx[:, 0] >= 8
        if flip.any():
            idx[flip] = 15 - idx[flip]
            q0[flip], q1[flip] = q1[flip].copy(), q0[flip].copy()

        u = np.uint64
        lo = np.full(blk.shape[0], 3, dtype=np.uint64)          # mode 10 == 0b00011
        lo |= q0[:, 0].astype(np.uint64) << u(5)                # rw
        lo |= q0[:, 1].astype(np.uint64) << u(15)               # gw
        lo |= q0[:, 2].astype(np.uint64) << u(25)               # bw
        lo |= q1[:, 0].astype(np.uint64) << u(35)               # rx
        lo |= q1[:, 1].astype(np.uint64) << u(45)               # gx
        bx = q1[:, 2].astype(np.uint64)
        lo |= (bx & u(0x1FF)) << u(55)                          # bx low 9 bits
        hi = (bx >> u(9)) & u(1)                                # bx bit 9 -> global bit 64
        hi |= idx[:, 0].astype(np.uint64) << u(1)               # anchor, 3 bits
        for k in range(1, 16):
            hi |= idx[:, k].astype(np.uint64) << u(1 + 3 + (k - 1) * 4)

        out_lo[start:start + blk.shape[0]] = lo
        out_hi[start:start + blk.shape[0]] = hi

    packed = np.empty((n, 2), dtype="<u8")
    packed[:, 0] = out_lo
    packed[:, 1] = out_hi
    return packed.tobytes()


def rgb_to_bc6h_texture(rgb, width, height, generate_mips=True):
    """rgb: float (h, w, 3) in 0..1. Returns (texture_bytes, texturemips_bytes)."""
    import numpy as np
    levels = [(rgb.astype(np.float32), width, height)]
    if generate_mips:
        as_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        levels = [(lvl.astype(np.float32) / 255.0, lw, lh)
                  for lvl, lw, lh in build_mip_chain(as_u8, width, height)]
        levels[0] = (rgb.astype(np.float32), width, height)
    mips = [encode_bc6h_uf16(level, lw, lh) for level, lw, lh in levels]
    dds = write_dds(DDSImage(width, height, 95, mips))  # DXGI BC6H_UF16
    return dds_to_texture(dds)


def build_mip_chain(rgb, width, height):
    """Full 2x2 box-filtered mip chain down to 1x1, as [(rgb, w, h), ...].
    Averaged on the stored (sRGB-encoded) values rather than in linear light:
    these are function masks whose channel value is a semantic level, so
    preserving the authored level through the chain matters more than
    photometric correctness."""
    import numpy as np
    chain = [(rgb, width, height)]
    w, h = width, height
    cur = rgb
    while w > 1 or h > 1:
        nw, nh = max(1, w // 2), max(1, h // 2)
        src = cur[:nh * 2, :nw * 2].astype(np.uint16)
        if src.shape[0] >= 2 and src.shape[1] >= 2:
            cur = ((src[0::2, 0::2] + src[1::2, 0::2]
                    + src[0::2, 1::2] + src[1::2, 1::2] + 2) // 4).astype(np.uint8)
        else:
            cur = cur[:nh, :nw]
        w, h = nw, nh
        chain.append((cur, w, h))
    return chain


def rgba_to_bc1_texture(rgb, width, height, srgb=True, generate_mips=True):
    """rgb: uint8 (h, w, 3). Returns (texture_bytes, texturemips_bytes) ready
    to drop next to the car's other .texture files."""
    dxgi = 72 if srgb else 71  # BC1_UNORM_SRGB / BC1_UNORM
    levels = build_mip_chain(rgb, width, height) if generate_mips else [(rgb, width, height)]
    mips = [encode_bc1(level, lw, lh) for level, lw, lh in levels]
    dds = write_dds(DDSImage(width, height, dxgi, mips))
    return dds_to_texture(dds)


def image_to_texture(pixels_rgba, width, height):
    """Uncompressed RGBA8 single-mip, matching acedit-texture.exe's own
    default behaviour for a plain PNG with no recognised authoring context
    (verified: it does NOT BC-compress or generate mips for a bare PNG -
    only pre-compressed/pre-mipped DDS input goes through the tiling path).
    If you need real BC7/mip production output (as used by real car mod
    textures), pre-compress with e.g. texconv/nvcompress and feed the
    resulting DDS to dds_to_texture() instead - that path is fully general.
    """
    meta = Metadata(width, height, 1, FORMATS_BY_DXGI[28].kunos, array_size=1)
    texture = encode_metadata(meta, tile_counts_per_mip=None)
    return texture, pixels_rgba


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _read_png_rgba(path):
    from PIL import Image
    img = Image.open(path).convert('RGBA')
    return img.tobytes(), img.width, img.height


def _write_png_rgba(path, pixels, width, height):
    from PIL import Image
    img = Image.frombytes('RGBA', (width, height), pixels)
    img.save(path)


def inspect_file(path):
    """Read-only: figure out what a file is and report its metadata without
    converting anything. Used by the GUI to show info as soon as a file is
    picked, before the user hits Convert."""
    base, ext = os.path.splitext(path)
    ext = ext.lower()
    info = {'input_path': path, 'input_ext': ext}

    if ext == '.texture':
        texture = open(path, 'rb').read()
        meta = decode_metadata(texture)
        fmt = FORMATS_BY_KUNOS.get(meta.kunos_format)
        info.update({
            'input_type': 'texture',
            'width': meta.width, 'height': meta.height,
            'mipcount': meta.mipcount,
            'format_name': fmt.name if fmt else f'unknown (kunos={meta.kunos_format})',
            'compressed': fmt.compressed if fmt else None,
            'will_output': [base + '.dds'],
        })
        mips_path = base + '.texturemips'
        if not os.path.exists(mips_path):
            info['warning'] = f'{os.path.basename(mips_path)} introuvable a cote du fichier'
    elif ext == '.dds':
        data = open(path, 'rb').read()
        hdr = read_dds_header(data)
        fmt = FORMATS_BY_DXGI.get(hdr.dxgi_format)
        info.update({
            'input_type': 'dds' if fmt else None,
            'width': hdr.width, 'height': hdr.height,
            'mipcount': hdr.mipmapcount,
            'format_name': fmt.name if fmt else f'DXGI {hdr.dxgi_format} (non supporte)',
            'compressed': fmt.compressed if fmt else None,
            'will_output': [base + '.texture', base + '.texturemips'] if fmt else [],
        })
        if not fmt:
            info['error'] = (
                f"Format DXGI {hdr.dxgi_format} pas encore gere par ace_texture.py. "
                f"Dimensions/mips lus depuis l'entete DDS, mais impossible de convertir "
                f"tant que ce format n'est pas ajoute a la table FORMATS."
            )
    elif ext in ('.png', '.bmp', '.tga', '.jpg', '.jpeg'):
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        info.update({
            'input_type': 'image',
            'width': w, 'height': h,
            'mipcount': 1,
            'format_name': 'RGBA8_UNORM (non compresse, genere par ace_texture.py)',
            'compressed': False,
            'will_output': [base + '.texture', base + '.texturemips'],
        })
    else:
        info['input_type'] = None
        info['error'] = f'extension non supportee: {ext}'
    return info


def convert_file(path):
    """Do the actual conversion. Returns an info dict (same shape as
    inspect_file) plus 'ok', 'outputs' (list of (path, size_bytes)) and
    'error' on failure."""
    info = inspect_file(path)
    base, ext = os.path.splitext(path)
    ext = ext.lower()

    try:
        if ext == '.texture':
            mips_path = base + '.texturemips'
            if not os.path.exists(mips_path):
                raise FileNotFoundError(f"{os.path.basename(mips_path)} introuvable a cote du fichier")
            texture = open(path, 'rb').read()
            texturemips = open(mips_path, 'rb').read()
            dds = texture_to_dds(texture, texturemips)
            out_path = base + '.dds'
            open(out_path, 'wb').write(dds)
            info['outputs'] = [(out_path, len(dds))]

        elif ext == '.dds':
            data = open(path, 'rb').read()
            texture, texturemips = dds_to_texture(data)
            out1, out2 = base + '.texture', base + '.texturemips'
            open(out1, 'wb').write(texture)
            open(out2, 'wb').write(texturemips)
            info['outputs'] = [(out1, len(texture)), (out2, len(texturemips))]

        elif ext in ('.png', '.bmp', '.tga', '.jpg', '.jpeg'):
            pixels, w, h = _read_png_rgba(path)
            texture, texturemips = image_to_texture(pixels, w, h)
            out1, out2 = base + '.texture', base + '.texturemips'
            open(out1, 'wb').write(texture)
            open(out2, 'wb').write(texturemips)
            info['outputs'] = [(out1, len(texture)), (out2, len(texturemips))]

        else:
            raise ValueError(f"extension non supportee: {ext}")

        info['ok'] = True
    except Exception as e:
        info['ok'] = False
        info['error'] = str(e)
    return info


def main(argv):
    if len(argv) < 2:
        print("usage: ace_texture.py <file.dds|file.png|file.texture>")
        return 1
    result = convert_file(argv[1])
    if result['ok']:
        print(f"pass {os.path.basename(argv[1])}")
        return 0
    else:
        print(f"fail {os.path.basename(argv[1])} ({result.get('error')})")
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))


# --------------------------------------------------------------------------
# NOTES / confidence levels
# --------------------------------------------------------------------------
# SOLID (byte-for-byte validated against real game asset ext_skin_c
# 4096x4096 BC1_SRGB, 13 mips, and against acedit-texture.exe output on
# synthetic marker files at multiple sizes: 64, 512, 1024):
#   - protobuf wire format / field numbers 1,2,3,4,6,7,8
#   - tiling/detiling algorithm and tile shapes for 8-byte and 16-byte
#     block formats
#   - uncompressed formats are stored unpadded/untiled
#
# BEST EFFORT (plausible, replicated from observation, not required for
# correctness of the pixel data itself):
#   - TilingInfo sub-fields 3,4,5,7 exact semantics
#   - field 11's exact meaning
#   - PixelFormat enum values for formats never tested (BC2, BC6H, BC1
#     non-sRGB, R8G8B8A8 non-sRGB variants beyond what's listed, etc.) -
#     extend FORMATS_BY_DXGI / FORMATS_BY_KUNOS as needed by probing
#     acedit-texture.exe the same way this module's development did:
#     feed it a synthetic DX10 DDS of the format you need and read back
#     the resulting field-4 value from the .texture output.
#
# NOT IMPLEMENTED:
#   - Real BC1/BC7 compression from raw pixels (acedit-texture.exe itself
#     does not do this for plain PNG input either - it only tiles/retiles
#     data that arrives already compressed in a DDS). For production mod
#     textures needing BC7 + full mip chains, compress with an external
#     tool (texconv, nvcompress, compressonator) to DDS first, then run
#     that DDS through dds_to_texture().
#   - ConversionSettings (field 10) - provenance metadata only, not needed
#     for the game to load the texture.
