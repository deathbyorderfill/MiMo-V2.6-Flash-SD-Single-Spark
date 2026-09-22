"""Typeset the MiMo-V2.6-Flash-SD benchmark numbers over generated art.

Usage:  python3 docs/make_hero.py docs/hero_art.png docs/hero.png

Values come from the repo README table / docs/benchmarks.svg (the published set),
so this graphic cannot drift from the chart it sits next to.

Layout follows the art: the plasma occupies the horizontal middle band, so text
lives in the near-black top and bottom strips and the scrims only touch the edges.
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080

# The shipped hero.png was rendered with Segoe UI. On a box without it the
# script still runs, but the type will not match the committed PNG exactly.
FONT_DIRS = ["C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/",
             "/usr/share/fonts/truetype/msttcorefonts/"]
FALLBACK = {"segoeuib.ttf": "DejaVuSans-Bold.ttf",
            "segoeui.ttf": "DejaVuSans.ttf",
            "segoeuil.ttf": "DejaVuSans.ttf"}

STATS = [
    ("PROSE", "32.2", "18.4", (0x5A, 0xAC, 0xFF)),
    ("CODE", "36.7", "18.1", (0xFF, 0x91, 0x47)),
    ("MATH", "38.8", "18.4", (0x35, 0xDD, 0x92)),
]
AGG_NEW, AGG_OLD = "57.7", "30.5"
VIOLET = (0xBB, 0x8C, 0xFF)

WHITE = (255, 255, 255)
DIM = (163, 171, 190)
FAINT = (118, 126, 147)


def font(name, size):
    for d in FONT_DIRS:
        for cand in (name, FALLBACK.get(name, name)):
            p = os.path.join(d, cand)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    raise SystemExit(f"no font found for {name}; install Segoe UI or DejaVu")


def tracked(d, xy, text, fnt, fill, track=0.0):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch) + track
    return x - xy[0]


def band(size, stops):
    """Vertical alpha ramp from (y_frac, alpha) stops."""
    g = Image.new("L", (1, size[1]))
    px = g.load()
    for y in range(size[1]):
        t = y / (size[1] - 1)
        for (t0, a0), (t1, a1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / max(1e-6, t1 - t0)
                px[0, y] = int(a0 + (a1 - a0) * k)
                break
    return g.resize(size, Image.BILINEAR)


def cover(im, w, h):
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, t, l + w, t + h))


def main(bg_path, out_path):
    img = cover(Image.open(bg_path).convert("RGB"), W, H)

    # scrims: strong at top and bottom, untouched through the plasma core
    black = Image.new("RGB", (W, H), (1, 2, 6))
    img = Image.composite(black, img, band((W, H), [
        (0.00, 225), (0.20, 60), (0.30, 0),      # top strip
        (0.62, 0), (0.74, 165), (1.00, 238),     # bottom strip
    ]))
    d = ImageDraw.Draw(img)

    x0 = 112

    # ---- top: identity ----------------------------------------------------
    tracked(d, (x0, 74), "309B MoE  ·  15B ACTIVE  ·  256K CONTEXT  ·  EVERY EXPERT RESIDENT",
            font("segoeuib.ttf", 20), (128, 192, 255), track=3.2)
    d.text((x0, 116), "MiMo-V2.6-Flash", font=font("segoeuib.ttf", 92), fill=WHITE)
    d.text((x0 + 4, 228), "on one DGX Spark", font=font("segoeuil.ttf", 46), fill=DIM)

    # ---- bottom: the numbers ----------------------------------------------
    f_lab = font("segoeuib.ttf", 23)
    f_num = font("segoeuib.ttf", 112)
    f_unit = font("segoeui.ttf", 27)
    f_base = font("segoeui.ttf", 24)

    y_lab, y_num, y_base = 772, 806, 944
    for i, (lab, new, old, col) in enumerate(STATS):
        cx = x0 + i * 300
        tracked(d, (cx, y_lab), lab, f_lab, col, track=2.6)
        d.text((cx, y_num), new, font=f_num, fill=WHITE)
        d.text((cx + d.textlength(new, font=f_num) + 8, y_num + 70), "tok/s", font=f_unit, fill=FAINT)
        d.text((cx, y_base), f"from {old}", font=f_base, fill=FAINT)

    # divider, then the headline aggregate
    dx = x0 + 872
    d.line([(dx, 764), (dx, 976)], fill=(72, 80, 104), width=2)

    ax = dx + 84
    tracked(d, (ax, y_lab), "3 STREAMS, AGGREGATE", f_lab, VIOLET, track=2.6)

    f_agg = font("segoeuib.ttf", 158)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((ax, y_num - 28), AGG_NEW, font=f_agg, fill=VIOLET + (165,))
    img = Image.alpha_composite(img.convert("RGBA"),
                                glow.filter(ImageFilter.GaussianBlur(30))).convert("RGB")
    d = ImageDraw.Draw(img)
    d.text((ax, y_num - 28), AGG_NEW, font=f_agg, fill=WHITE)
    d.text((ax + d.textlength(AGG_NEW, font=f_agg) + 12, y_num + 74), "tok/s",
           font=font("segoeui.ttf", 34), fill=(196, 172, 244))
    # the 158px numeral drops well past the 112px columns, so this sits lower
    d.text((ax, y_base + 18), f"from {AGG_OLD}", font=f_base, fill=FAINT)

    # ---- footer -----------------------------------------------------------
    d.line([(x0, 1004), (W - x0, 1004)], fill=(52, 60, 82), width=1)
    d.text((x0, 1026), "GSM8K 188 / 200", font=font("segoeuib.ttf", 23), fill=WHITE)
    d.text((x0 + 196, 1026),
           "·  needle passes at 254k tokens  ·  \u201cfrom\u201d = same build, speculation off  "
           "·  400-token greedy, decode after first token",
           font=font("segoeui.ttf", 23), fill=FAINT)

    img.save(out_path, optimize=True)
    print(f"wrote {out_path} {img.size}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
