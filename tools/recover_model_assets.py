"""Rebuild build/cyclegan_model from the RTL ROM sources alone.

The original build/cyclegan_model assets (params_manifest.json,
generator_params_q88.mem, reference vectors) were generated on another
machine and are not present in this workspace. Everything needed to
reconstruct them still lives in the repository:

* rtl/mn_generator_rom.sv   - exact epoch11 Q8.8 words 0..21602, packed as
                              22 low 10-bit planes + 3 sign planes
* rtl/mn_residual_rom.sv    - words 0..14026 (14 low + 2 sign), independent copy
* rtl/mn_decoder_rom.sv     - words 0..20426 (20 low + 3 sign), independent copy
* rtl/mn_encoder_rom.sv     - words 0..1066 as direct 16-bit words
* rtl/mn_model_params_init.vh - the 10 sparse exception (address, value) pairs
* rtl/cyclegan_encoder.sv   - ground-truth parameter base addresses per layer
* docs/cyclegan_stage3_deployment.md - decoder/output parameter bases

Cross-checking the four independent ROM copies validates the unpacking.
The manifest layout mirrors tools/prepare_instance_norm_model.py and was
confirmed against the hardware schedulers in rtl/cyclegan_encoder.sv.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
OUT = ROOT / "build/cyclegan_model"


def parse_ram_instances(text):
    """Return [(read_width, [INIT values in INIT_0..INIT_27 order])] in file order."""
    instances = []
    for m in re.finditer(r"EFX_RAM10\s*#\((.*?)\)\s*\w+\s*\(", text, re.S):
        body = m.group(1)
        width = int(re.search(r"READ_WIDTH\((\d+)\)", body).group(1))
        inits = {int(im.group(1), 16): int(im.group(2), 16)
                 for im in re.finditer(r"\.INIT_([0-9A-Fa-f]+)\(256'h([0-9a-f]{64})\)", body)}
        order = sorted(inits)
        instances.append((width, [inits[i] for i in order]))
    return instances


def unpack_plane(inits, width, count, msb_first):
    """Unpack one BRAM plane of `count` words x `width` bits.

    Vendor model (conv_mem/conv_mem_unit/efx_ram10.v): mem[256*k+i]=INIT_k[i]
    and rdata[i]=mem[addr*width+i], i.e. INIT bit i means bit i (LSB first).
    msb_first instead treats each INIT word MSB-first; the caller picks the
    convention that satisfies the cross-checks.
    """
    bits = []
    for v in inits:
        if msb_first:
            bits.extend((v >> (255 - i)) & 1 for i in range(256))
        else:
            bits.extend((v >> i) & 1 for i in range(256))
    return [sum(bits[width * w + b] << b for b in range(width)) for w in range(count)]


def parse_packed_rom(path, reference=None):
    """Unpack a 10-bit-low + 1-bit-sign ROM file, resolving the bit order.

    Each word is {6{sign}, low10}: the sign plane selected by addr[14:13]
    drives the bit while the other planes are synchronously reset to zero,
    so sign(w) = plane[w>>13][w&8191]. `reference` (known-correct words)
    decides the convention when both are plausible.
    """
    instances = parse_ram_instances(path.read_text(encoding="utf8"))
    lo_planes = [v for w, v in instances if w == 10]
    hi_planes = [v for w, v in instances if w == 1]
    n_words = len(lo_planes) * 1024
    best = None
    for msb_lo in (False, True):
        lo = [unpack_plane(p, 10, 1024, msb_lo) for p in lo_planes]
        low_words = [lo[i >> 10][i & 1023] for i in range(n_words)]
        for msb_hi in (False, True):
            planes = [unpack_plane(p, 1, 8192, msb_hi) for p in hi_planes]
            words = [low | (0xFC00 if planes[i >> 13][i & 8191] else 0)
                     for i, low in enumerate(low_words)]
            if reference:
                score = sum(1 for a, b in zip(words, reference) if a == b)
            else:
                score = n_words
            if best is None or score > best[0]:
                best = (score, words, msb_lo, msb_hi)
    return best[1], best[2], best[3]


def parse_direct_rom(path, exceptions):
    """Parse a direct 16-bit ROM (mn_encoder_rom.sv style).

    Each EFX_RAM10 instance is one 512-word bank; only INIT_0..INIT_1F
    carry data (8192 bits = 512x16), the rest is zero padding. Vendor
    mapping: word w bit b = mem[16*w+b], mem[256*k+i]=INIT_k[i].
    """
    text = path.read_text(encoding="utf8")
    words = []
    for m in re.finditer(r"EFX_RAM10\s*#\((.*?)\)\s*\w+\s*\(", text, re.S):
        body = m.group(1)
        inits = {int(im.group(1), 16): int(im.group(2), 16)
                 for im in re.finditer(r"\.INIT_([0-9A-Fa-f]+)\(256'h([0-9a-f]{64})\)", body)}
        for s in range(32):
            v = inits.get(s, 0)
            for lane in range(16):
                words.append((v >> (16 * lane)) & 0xFFFF)
    bad = [(a, to_signed(words[a]), v) for a, v in exceptions
           if a < len(words) and to_signed(words[a]) != v]
    assert not bad, f"{path.name}: exception check failed: {bad}"
    return words


def parse_exceptions(path):
    text = path.read_text(encoding="utf8")
    addrs = [int(m.group(1)) for m in
             re.finditer(r"param_exception_addr=15'd(\d+)", text)]
    values = [int(m.group(1)) for m in
              re.finditer(r"param_exception_value=16'd(\d+)", text)]
    return list(zip(addrs, values))


def to_signed(v):
    return v - 65536 if v >= 32768 else v


def main():
    exceptions = parse_exceptions(RTL / "mn_model_params_init.vh")
    print(f"exceptions: {exceptions}")

    encoder = parse_direct_rom(RTL / "mn_encoder_rom.sv", exceptions)
    print(f"encoder ROM: {len(encoder)} words, exception check OK")

    gen, msb_lo, msb_hi = parse_packed_rom(RTL / "mn_generator_rom.sv",
                                           reference=encoder[:1067])
    print(f"generator ROM: {len(gen)} words (lo msb={msb_lo}, hi msb={msb_hi})")
    # The packed planes cannot represent the 10 exception words; patch them
    # from the exception table, then every copy must agree exactly.
    for addr, value in exceptions:
        gen[addr] = value
    # mn_encoder_rom.sv pads 1067 real words to 3x512; compare the real words.
    assert gen[:1067] == encoder[:1067], "generator/encoder ROM mismatch"

    for name, count in (("mn_residual_rom.sv", 14027), ("mn_decoder_rom.sv", 20426)):
        words, _, _ = parse_packed_rom(RTL / name, reference=gen[:count])
        for addr, value in exceptions:
            if addr < count:
                words[addr] = value
        match = sum(1 for a, b in zip(words[:count], gen[:count]) if a == b)
        print(f"{name}: {match}/{count} words match generator copy")
        assert match == count, f"{name} disagrees with generator ROM"
    exc_addrs = {a for a, _ in exceptions}
    for i, v in enumerate(gen):
        if i not in exc_addrs:
            assert ((v >> 10) & 63) == (63 if v & 1024 else 0), \
                f"word {i} malformed"

    total = 21603
    assert all(v == 0 for v in gen[total:]), "unexpected data beyond word 21602"
    padded = gen[:total] + [0] * (22016 - total)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generator_params_q88.mem").write_text(
        "\n".join(f"{v & 0xFFFF:04x}" for v in padded) + "\n", encoding="ascii")

    manifest = {
        "checkpoint_sha256": "7d6aaac3a6192dddde81602e6120d1aa4dc732690efb8c908cb4757e12dc9a8b",
        "recovered_from": "rtl ROMs; see tools/recover_model_assets.py",
        "encoder": [
            {"layer": 1, "cin": 3, "cout": 8, "dw": 0, "pw": 27, "pw_bn": 51},
            {"layer": 5, "cin": 8, "cout": 16, "dw": 67, "dw_bn": 139,
             "pw": 155, "pw_bn": 283},
            {"layer": 7, "cin": 16, "cout": 32, "dw": 315, "dw_bn": 459,
             "pw": 491, "pw_bn": 1003},
        ],
        "blocks": [
            {"layer": 9 + i, "cin": 32, "cout": 32,
             "dw": 1067 + 1440 * i, "dw_bn": 1355 + 1440 * i,
             "pw": 1419 + 1440 * i, "pw_bn": 2443 + 1440 * i}
            for i in range(9)
        ],
        "decoder": [
            {"model_index": 18, "cin": 32, "cout": 16,
             "weight_base": 14027, "bn_base": 18635},
            {"model_index": 21, "cin": 16, "cout": 8,
             "weight_base": 18667, "bn_base": 19819},
            {"model_index": 24, "cin": 8, "cout": 8,
             "weight_base": 19835, "bn_base": 20411},
        ],
        "output": {"kernel": 7, "weight_base": 20427},
    }
    (OUT / "params_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf8")

    vectors = OUT / "vectors"
    vectors.mkdir(exist_ok=True)
    rgb = test_image()
    frame = np.concatenate(
        [rgb, np.zeros((240, 320, 1), dtype=np.uint8)], axis=2)
    (vectors / "frame_rgbx.bin").write_bytes(frame.tobytes())
    from PIL import Image
    Image.fromarray(rgb).save(OUT / "input.png")

    print("RECOVER_MODEL_ASSETS_PASS "
          f"{total} words, manifest and test input written to {OUT}")


def test_image():
    """Deterministic 320x240 synthetic photo-like image."""
    y, x = np.mgrid[0:240, 0:320]
    r = 96 + 60 * np.sin(x / 34.0) + 40 * np.cos(y / 27.0)
    g = 120 + 70 * np.sin((x + y) / 41.0)
    b = 140 + 60 * np.cos(x / 23.0 - y / 31.0)
    image = np.stack([r, g, b], axis=2)
    yy, xx = np.mgrid[0:240, 0:320]
    mask = (yy - 120) ** 2 + (xx - 160) ** 2 < 70 ** 2
    image[mask] = [230, 180, 90]
    mask2 = (np.abs(yy - 60) < 25) & (np.abs(xx - 80) < 45)
    image[mask2] = [60, 140, 200]
    return np.clip(image, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
