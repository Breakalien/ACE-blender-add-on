#!/usr/bin/env python3
"""
acevo_lite_codec - generic byte-exact protobuf codec for .actor and
.carlightingsystem files.

Stripped-down subset of tools/acevo_pb.py from ac-evo-data-tools: only the
schema-independent wire-format decode/encode is kept (no descriptor pool, no
protobuf package, no field naming/enum lookup, no CLI). Stdlib only, so it
can be dropped into a Blender add-on's Python environment unmodified.

Reads the protobuf wire format, infers each field's type, and produces a
dict that re-encodes to the exact same bytes. Dict keys are "<field_number>:
<type>" (types: msg, str, bytes, f32, f64, i32, i64, varint, packed_f32).
A repeated field becomes a list.
"""

import struct

# ---------------------------------------------------------------- wire format


def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def write_varint(value):
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def iter_fields(buf):
    """Walk a protobuf message, yielding (field_number, wire_type, raw_value)."""
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if field == 0:
            raise ValueError("field number 0")
        if wire == 0:
            val, pos = read_varint(buf, pos)
        elif wire == 1:
            if pos + 8 > len(buf):
                raise ValueError("truncated fixed64")
            val, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            length, pos = read_varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("truncated length-delimited block")
            val, pos = buf[pos:pos + length], pos + length
        elif wire == 5:
            if pos + 4 > len(buf):
                raise ValueError("truncated fixed32")
            val, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError("unsupported wire type %d" % wire)
        yield field, wire, val


# ---------------------------------------------------------------- heuristics


def is_message(data):
    """Does the block parse as a complete, plausible protobuf message?"""
    if not data:
        return False
    try:
        n = 0
        for field, wire, _ in iter_fields(data):
            if field > 4096:
                return False
            n += 1
        return n > 0
    except ValueError:
        return False


def is_text(data):
    """Strict printable ASCII. A binary block can accidentally be valid UTF-8."""
    if not data:
        return False
    return all(0x20 <= b <= 0x7E for b in data)


def as_float(raw4):
    """fixed32 -> float when the magnitude is plausible, otherwise None."""
    f = struct.unpack("<f", raw4)[0]
    if f != f or f in (float("inf"), float("-inf")):
        return None
    a = abs(f)
    if f == 0.0 or 1e-6 <= a <= 1e12:
        return f
    return None


def is_packed_f32(data):
    """A run of packed floats (used by some curve files)."""
    if not data or len(data) % 4 or len(data) < 8:
        return False
    return all(as_float(data[i:i + 4]) is not None for i in range(0, len(data), 4))


def clean_float(f):
    """Round away float32 artefacts for readable JSON, without losing precision."""
    for digits in range(1, 10):
        r = round(f, digits)
        if struct.pack("<f", r) == struct.pack("<f", f):
            return r
    return f


# ---------------------------------------------------------------- decoding


def decode_value(wire, raw):
    """-> (type_str, value)"""
    if wire == 0:
        return "varint", raw
    if wire == 1:
        d = struct.unpack("<d", raw)[0]
        if d == d and abs(d) < 1e300:
            return "f64", d
        return "i64", struct.unpack("<Q", raw)[0]
    if wire == 5:
        f = as_float(raw)
        if f is not None:
            return "f32", clean_float(f)
        return "i32", struct.unpack("<I", raw)[0]
    # Try the sub-message first: real strings almost always fail to parse as
    # protobuf, whereas the converse is not true.
    if is_message(raw):
        return "msg", decode_message(raw)
    if is_text(raw):
        return "str", raw.decode("ascii")
    if is_packed_f32(raw):
        return "packed_f32", [clean_float(as_float(raw[i:i + 4]))
                              for i in range(0, len(raw), 4)]
    if not raw:
        return "msg", {}
    return "bytes", raw.hex()


def keys_contiguous(keys):
    """Does grouping repeated fields preserve the original order?

    Encoding rewrites every occurrence of a key at the position of its first
    appearance, so order survives if and only if each key's occurrences are
    contiguous."""
    closed = set()
    prev = None
    for k in keys:
        if k != prev:
            if k in closed:
                return False
            if prev is not None:
                closed.add(prev)
            prev = k
    return True


def decode_message(buf):
    items = []
    for field, wire, raw in iter_fields(buf):
        typ, val = decode_value(wire, raw)
        items.append((field, typ, val))

    out = {}
    keys = []
    for field, typ, val in items:
        key = "%d:%s" % (field, typ)
        keys.append(key)
        if key in out:
            cur = out[key]
            if not (isinstance(cur, list) and
                    (typ != "packed_f32" or (cur and isinstance(cur[0], list)))):
                cur = out[key] = [cur]
            cur.append(val)
        else:
            out[key] = val

    if not keys_contiguous(keys):
        # interleaved fields: fall back to an exact sequential form
        return {"_seq": [{"f": f, "t": t, "v": v} for f, t, v in items]}
    return out


# ---------------------------------------------------------------- encoding


def encode_value(field, typ, val):
    if typ == "varint":
        return write_varint(field << 3) + write_varint(val)
    if typ == "f64":
        return write_varint((field << 3) | 1) + struct.pack("<d", val)
    if typ == "i64":
        return write_varint((field << 3) | 1) + struct.pack("<Q", val)
    if typ == "f32":
        return write_varint((field << 3) | 5) + struct.pack("<f", val)
    if typ == "i32":
        return write_varint((field << 3) | 5) + struct.pack("<I", val)
    if typ == "str":
        payload = val.encode("utf-8")
    elif typ == "bytes":
        payload = bytes.fromhex(val)
    elif typ == "packed_f32":
        payload = b"".join(struct.pack("<f", f) for f in val)
    elif typ == "msg":
        payload = encode_message(val)
    else:
        raise ValueError("unknown type: %s" % typ)
    return write_varint((field << 3) | 2) + write_varint(len(payload)) + payload


def encode_message(obj):
    out = bytearray()
    if "_seq" in obj:
        for it in obj["_seq"]:
            out += encode_value(it["f"], it["t"], it["v"])
        return bytes(out)
    for key, val in obj.items():
        # only the number and the type matter here; a ":name" suffix (from a
        # file previously decoded by the full acevo_pb tool) is decorative
        parts = key.split(":")
        field, typ = int(parts[0]), parts[1]
        repeated = isinstance(val, list) and (
            typ != "packed_f32" or (val and isinstance(val[0], list)))
        for item in (val if repeated else [val]):
            out += encode_value(field, typ, item)
    return bytes(out)
