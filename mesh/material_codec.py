"""
Codec bidirectionnel pour les fichiers .material d'Assetto Corsa EVO.

Format sous-jacent : protobuf wire-format, auto-descriptif (les noms de
proprietes/textures sont stockes comme des chaines a l'interieur du binaire).

Schema deduit par inspection de ext_skin_design_2.material (voir notes ci-dessous),
valide par un test d'aller-retour bit-a-bit (decode -> encode -> bytes identiques).

    Racine
      field1 (string)          -> nom du shader (ex: "UberVehicleMaterial")
      field4 (repeated msg)    -> Property
          field1 (string)      = nom
          field2 (msg, Value)  = TOUJOURS present (eventuellement vide)
              field1 (fixed32)      -> kind=1 SCALAR, valeur directe
              field2 (msg 1-2 flt)  -> kind=2 VEC2  {field1=x, field2=y}
              field3 (msg 1-3 flt)  -> kind=3 VEC3  {field1..3=x,y,z}
              field4 (msg 1-4 flt)  -> kind=4 VEC4  {field1..4=x,y,z,w} (couleurs RGBA)
              (rien)                -> kind=0 UNSET (mot-cle shader non renseigne)
              A l'interieur d'un VEC2/3/4, les composantes valant exactement 0.0
              peuvent etre omises individuellement (presence explicite sinon).
      field5 (repeated msg)    -> TextureSlot
          field1 (string)      = nom du slot
          field2 (msg)         = TOUJOURS present (eventuellement vide)
              field2 (string)  -> chemin relatif de la texture assignee
              (rien)           -> aucune texture assignee a ce slot

Tout champ top-level inattendu (ne collant pas a field1/4/5) est conserve tel
quel (RawField) pour ne jamais corrompre un fichier qu'on ne comprend pas
entierement.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Wire-format primitives
# ---------------------------------------------------------------------------

def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def write_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def encode_tag(field_no: int, wiretype: int) -> bytes:
    return write_varint((field_no << 3) | wiretype)


def encode_bytes_field(field_no: int, payload: bytes) -> bytes:
    return encode_tag(field_no, 2) + write_varint(len(payload)) + payload


def encode_fixed32_field(field_no: int, value: float) -> bytes:
    return encode_tag(field_no, 5) + struct.pack("<f", value)


def parse_message(buf: bytes, start: int, end: int) -> list[tuple[int, str, object]]:
    fields = []
    pos = start
    while pos < end:
        tag, pos = read_varint(buf, pos)
        field_no = tag >> 3
        wiretype = tag & 0x7
        if wiretype == 0:
            val, pos = read_varint(buf, pos)
            fields.append((field_no, "varint", val))
        elif wiretype == 1:
            val = struct.unpack_from("<d", buf, pos)[0]
            pos += 8
            fields.append((field_no, "fixed64", val))
        elif wiretype == 2:
            length, pos = read_varint(buf, pos)
            chunk = buf[pos:pos + length]
            pos += length
            fields.append((field_no, "bytes", chunk))
        elif wiretype == 5:
            val = struct.unpack_from("<f", buf, pos)[0]
            pos += 4
            fields.append((field_no, "fixed32", val))
        else:
            raise ValueError(f"Wiretype non supporte {wiretype} a l'offset {pos}")
    return fields


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

KIND_UNSET = 0
KIND_SCALAR = 1
KIND_VEC2 = 2
KIND_VEC3 = 3
KIND_VEC4 = 4


@dataclass
class Property:
    name: str
    kind: int = KIND_UNSET
    components: dict[int, float] = field(default_factory=dict)

    def get(self, idx: int, default: float = 0.0) -> float:
        return self.components.get(idx, default)


@dataclass
class TextureSlot:
    name: str
    path: Optional[str] = None


@dataclass
class RawField:
    field_no: int
    wiretype: str
    value: object


Item = Union[Property, TextureSlot, RawField]


@dataclass
class MaterialFile:
    shader_name: str
    items: list[Item] = field(default_factory=list)

    @property
    def properties(self) -> list[Property]:
        return [it for it in self.items if isinstance(it, Property)]

    @property
    def textures(self) -> list[TextureSlot]:
        return [it for it in self.items if isinstance(it, TextureSlot)]


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _decode_value(buf: bytes) -> tuple[int, dict[int, float]]:
    fields = parse_message(buf, 0, len(buf))
    if not fields:
        return KIND_UNSET, {}
    field_no, wiretype, value = fields[0]
    if field_no == KIND_SCALAR and wiretype == "fixed32":
        return KIND_SCALAR, {1: value}
    if field_no in (KIND_VEC2, KIND_VEC3, KIND_VEC4) and wiretype == "bytes":
        inner = parse_message(value, 0, len(value))
        components = {f: v for f, w, v in inner if w == "fixed32"}
        return field_no, components
    raise ValueError(f"Value inattendue: field{field_no} ({wiretype})")


def decode_material(path: Path) -> MaterialFile:
    data = Path(path).read_bytes()
    top = parse_message(data, 0, len(data))

    shader_name = ""
    items: list[Item] = []

    for field_no, wiretype, value in top:
        if field_no == 1 and wiretype == "bytes":
            shader_name = value.decode("utf-8")
        elif field_no == 4 and wiretype == "bytes":
            sub = parse_message(value, 0, len(value))
            name = next((v.decode("utf-8") for f, w, v in sub if f == 1 and w == "bytes"), "")
            value_bytes = next((v for f, w, v in sub if f == 2 and w == "bytes"), b"")
            kind, components = _decode_value(value_bytes)
            items.append(Property(name, kind, components))
        elif field_no == 5 and wiretype == "bytes":
            sub = parse_message(value, 0, len(value))
            name = next((v.decode("utf-8") for f, w, v in sub if f == 1 and w == "bytes"), "")
            value_bytes = next((v for f, w, v in sub if f == 2 and w == "bytes"), b"")
            inner = parse_message(value_bytes, 0, len(value_bytes))
            path_bytes = next((v for f, w, v in inner if f == 2 and w == "bytes"), None)
            tex_path = path_bytes.decode("utf-8") if path_bytes is not None else None
            items.append(TextureSlot(name, tex_path))
        else:
            items.append(RawField(field_no, wiretype, value))

    return MaterialFile(shader_name, items)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _encode_value(prop: Property) -> bytes:
    if prop.kind == KIND_UNSET:
        return b""
    if prop.kind == KIND_SCALAR:
        return encode_fixed32_field(1, prop.components.get(1, 0.0))
    inner = b"".join(
        encode_fixed32_field(t, v) for t, v in sorted(prop.components.items())
    )
    return encode_bytes_field(prop.kind, inner)


def _encode_property(prop: Property) -> bytes:
    payload = encode_bytes_field(1, prop.name.encode("utf-8"))
    payload += encode_bytes_field(2, _encode_value(prop))
    return encode_bytes_field(4, payload)


def _encode_texture(tex: TextureSlot) -> bytes:
    payload = encode_bytes_field(1, tex.name.encode("utf-8"))
    inner = encode_bytes_field(2, tex.path.encode("utf-8")) if tex.path else b""
    payload += encode_bytes_field(2, inner)
    return encode_bytes_field(5, payload)


def _encode_raw(raw: RawField) -> bytes:
    if raw.wiretype == "varint":
        return encode_tag(raw.field_no, 0) + write_varint(raw.value)
    if raw.wiretype == "fixed64":
        return encode_tag(raw.field_no, 1) + struct.pack("<d", raw.value)
    if raw.wiretype == "bytes":
        return encode_bytes_field(raw.field_no, raw.value)
    if raw.wiretype == "fixed32":
        return encode_tag(raw.field_no, 5) + struct.pack("<f", raw.value)
    raise ValueError(f"Wiretype brut non supporte: {raw.wiretype}")


def encode_material(mf: MaterialFile) -> bytes:
    out = bytearray()
    out += encode_bytes_field(1, mf.shader_name.encode("utf-8"))
    for item in mf.items:
        if isinstance(item, Property):
            out += _encode_property(item)
        elif isinstance(item, TextureSlot):
            out += _encode_texture(item)
        elif isinstance(item, RawField):
            out += _encode_raw(item)
        else:
            raise TypeError(f"Type d'item inconnu: {type(item)}")
    return bytes(out)


def save_material(mf: MaterialFile, path: Path) -> None:
    Path(path).write_bytes(encode_material(mf))


# ---------------------------------------------------------------------------
# Presets (etat complet des proprietes d'un shader donne : quelles proprietes
# sont activees, avec quelles valeurs). Independant des textures - un preset
# ne touche que field4 (Property), jamais field5 (TextureSlot).
# ---------------------------------------------------------------------------

def properties_to_dict(properties: list[Property]) -> dict:
    return {
        p.name: {"kind": p.kind, "components": {str(k): v for k, v in p.components.items()}}
        for p in properties
    }


def apply_properties_from_dict(properties: list[Property], data: dict) -> tuple[int, int]:
    """Ecrase en place les proprietes de `properties` (matchees par nom) avec
    l'etat sauvegarde dans `data` (issu de properties_to_dict) - type ET valeurs.
    Renvoie (nb appliquees, nb ignorees car absentes de `properties`).
    """
    by_name = {p.name: p for p in properties}
    applied = 0
    skipped = 0
    for name, entry in data.items():
        prop = by_name.get(name)
        if prop is None:
            skipped += 1
            continue
        prop.kind = entry["kind"]
        prop.components = {int(k): v for k, v in entry.get("components", {}).items()}
        applied += 1
    return applied, skipped


def apply_activation_from_dict(properties: list[Property], data: dict) -> tuple[int, int]:
    """Comme apply_properties_from_dict, mais ne copie QUE l'activation
    (active/desactive + type). Les valeurs ne sont jamais copiees depuis le
    preset : une propriete deja active garde ses valeurs actuelles, une
    propriete nouvellement activee repart sur des valeurs par defaut (0.0).
    Renvoie (nb appliquees, nb ignorees car absentes de `properties`).
    """
    by_name = {p.name: p for p in properties}
    applied = 0
    skipped = 0
    for name, entry in data.items():
        prop = by_name.get(name)
        if prop is None:
            skipped += 1
            continue
        target_kind = entry["kind"]
        if target_kind == KIND_UNSET:
            prop.kind = KIND_UNSET
            prop.components = {}
        elif prop.kind != target_kind:
            prop.kind = target_kind
            prop.components = {i: 0.0 for i in range(1, target_kind + 1)}
        # sinon : deja actif avec le meme type -> on ne touche pas aux valeurs
        applied += 1
    return applied, skipped


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1])
    mf = decode_material(src)
    original = src.read_bytes()
    rebuilt = encode_material(mf)
    print(f"Shader: {mf.shader_name}")
    print(f"Proprietes: {len(mf.properties)}  Textures: {len(mf.textures)}  Autres: {len(mf.items) - len(mf.properties) - len(mf.textures)}")
    print(f"Taille originale: {len(original)}  Taille reconstruite: {len(rebuilt)}")
    print("ROUND-TRIP IDENTIQUE" if original == rebuilt else "!!! DIFFERENCE DETECTEE !!!")
    if original != rebuilt:
        for i, (a, b) in enumerate(zip(original, rebuilt)):
            if a != b:
                print(f"Premier octet different a l'offset {i}: original={a:#x} reconstruit={b:#x}")
                break
