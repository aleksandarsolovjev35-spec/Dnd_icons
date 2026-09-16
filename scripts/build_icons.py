#!/usr/bin/env python3
"""
D&D Trinkets — icons.

Two deliverables:

1. Коллекция «Ручная живопись» — 6 безделушек из таблицы d100 в одном стиле:
       icons/painted/<id>-<slug>.png
2. Один предмет (d100 №01 «Мумифицированная рука гоблина») в 10 стилях:
       icons/styles/<style>.png

Each icon is 256x256 PNG. Where the artwork has a flat background, the object is
cut out and re-centred; icons/styles/transparent/ holds the versions with an
alpha channel. Styles whose background is part of the artwork (engraving on
parchment, stained glass panel) are framed instead of cut out.

Usage:  python3 scripts/build_icons.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MASTERS = ROOT / "masters"
ICONS = ROOT / "icons"
PREVIEW = ROOT / "preview"

CANVAS = 256        # final icon size
OBJECT_BOX = 202    # object bounding box (longest side) inside the canvas (~79%)

# ---------------------------------------------------------------- catalogue --
PAINTED_ITEMS = [
    dict(id="01", d100=1, slug="mummified-goblin-hand",
         ru="Мумифицированная рука гоблина", en="A mummified goblin hand"),
    dict(id="02", d100=2, slug="moonlight-crystal",
         ru="Кристалл, светящийся в лунном свете", en="A piece of crystal that faintly glows in the moonlight"),
    dict(id="07", d100=7, slug="skull-knucklebones",
         ru="Кости-«шестёрки» с черепом", en="A pair of knucklebone dice with a skull symbol on the six"),
    dict(id="06", d100=6, slug="glass-chess-piece",
         ru="Старая стеклянная шахматная фигура", en="An old chess piece made from glass"),
    dict(id="13", d100=13, slug="beast-tooth",
         ru="Зуб неизвестного зверя", en="A tooth from an unknown beast"),
    dict(id="14", d100=14, slug="dragon-scale",
         ru="Огромная чешуйка, возможно драконья", en="An enormous scale, perhaps from a dragon"),
    dict(id="15", d100=15, slug="green-feather",
         ru="Ярко-зелёное перо", en="A bright green feather"),
    dict(id="17", d100=17, slug="smoke-orb",
         ru="Стеклянный шар с движущимся дымом", en="A glass orb filled with moving smoke"),
    dict(id="23", d100=23, slug="rune-etched-brass-orb",
         ru="Латунная сфера с рунами", en="A brass orb etched with strange runes"),
    dict(id="43", d100=43, slug="silver-bell-without-clapper",
         ru="Серебряный колокольчик без язычка", en="A tiny silver bell without a clapper"),
    dict(id="44", d100=44, slug="gnomish-canary-lamp",
         ru="Механический канарейка в гномьем фонаре", en="A mechanical canary inside a gnomish lamp"),
]

# style -> how to build the 256x256 icon from masters/ten/<key>.png
#   cut    : cut out the flat background and re-centre the object
#   frame  : the artwork background is part of the style → square crop around the content
#   mosaic : square crop to the panel, plus a tile mosaic at 128px (stained glass needs resolution)
STYLE_VARIANTS = [
    dict(key="painted", ru="Ручная живопись", en="Hand-painted",
         mode="cut", thr=0.035, feather=0.6, bg=(40, 50, 58)),
    dict(key="pixel16", ru="16-битный пиксель", en="16-bit pixel art",
         mode="cut", thr=0.045, feather=0.0, pixel=True),
    dict(key="vector", ru="Плоский вектор", en="Flat vector",
         mode="cut", thr=0.035, feather=0.3),
    dict(key="realism", ru="Фотореализм 3D", en="Photoreal 3D",
         mode="cut", thr=0.030, feather=0.7),
    dict(key="comic", ru="Комикс", en="Comic ink",
         mode="cut", thr=0.035, feather=0.4),
    dict(key="watercolor", ru="Акварель", en="Watercolour",
         mode="frame", margin=0.05),
    dict(key="engraving", ru="Гравюра", en="Engraving",
         mode="frame", margin=0.05),
    dict(key="isometric", ru="Low-poly 3D", en="Low-poly render",
         mode="cut", thr=0.035, feather=0.5),
    dict(key="stained-glass", ru="Витраж", en="Stained glass",
         mode="mosaic", bbox=(252, 200, 960, 950)),
    dict(key="neon", ru="Неон/киберпанк", en="Neon cyberpunk",
         mode="cut", thr=0.050, feather=0.7),
]

ITEM_01 = PAINTED_ITEMS[0]


# ------------------------------------------------------------- background ---
def background_colour(arr: np.ndarray) -> np.ndarray:
    ring = np.concatenate([
        arr[0:4, :, :3].reshape(-1, 3), arr[-4:, :, :3].reshape(-1, 3),
        arr[:, 0:4, :3].reshape(-1, 3), arr[:, -4:, :3].reshape(-1, 3),
    ])
    return np.median(ring, axis=0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    areas = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == int(np.argmax(areas)) + 1


def cut_out(img: Image.Image, thr: float, feather: float) -> tuple[Image.Image, tuple[int, int, int]]:
    """Cut the flat background away: largest blob only, soft halo is dropped."""
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.int16)
    bg = background_colour(arr)
    dist = np.abs(arr[:, :, :3] - bg).sum(axis=2) / 3.0

    mask = largest_component(dist > thr * 255)
    mask = ndimage.binary_fill_holes(mask)

    alpha = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    out.putalpha(alpha)
    return out, tuple(int(c) for c in bg)


# ---------------------------------------------------------------- geometry --
def trim(img: Image.Image) -> Image.Image:
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return img.crop(bbox) if bbox else img


def detect_pixel_block(img: Image.Image, max_block: int = 24, tol: float = 6.0) -> int:
    """Largest N the picture survives downscale-by-N + nearest upscale with."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    ref = np.asarray(rgb, np.int16)
    best = 1
    for b in range(2, max_block + 1):
        if w % b or h % b:
            continue
        small = rgb.resize((w // b, h // b), Image.NEAREST)
        if np.abs(ref - np.asarray(small.resize((w, h), Image.NEAREST), np.int16)).mean() < tol:
            best = b
        else:
            break
    return best


def scale_object(img: Image.Image, pixel: bool = False, box: int = OBJECT_BOX) -> Image.Image:
    if pixel:
        block = detect_pixel_block(img)
        native = img.resize((max(1, img.width // block), max(1, img.height // block)), Image.NEAREST)
        scale = max(1, box // max(native.size))
        while scale > 1 and (native.width * scale > box or native.height * scale > box):
            scale -= 1
        return native.resize((native.width * scale, native.height * scale), Image.NEAREST)

    k = box / max(img.size)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def compose(img: Image.Image, bg: tuple[int, int, int] | None) -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS, CANVAS), bg + (255,) if bg else (0, 0, 0, 0))
    canvas.alpha_composite(img, ((CANVAS - img.width) // 2, (CANVAS - img.height) // 2))
    return canvas


def content_box(img: Image.Image, thr: float = 0.06) -> tuple[int, int, int, int]:
    """Bounding box of everything that differs from the artwork's own background."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    bg = background_colour(arr)
    dist = np.abs(arr - bg).sum(axis=2) / 3.0
    mask = ndimage.binary_opening(largest_component(dist > thr * 255), iterations=3)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return (0, 0, img.width, img.height)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_crop(img: Image.Image, box: tuple[int, int, int, int], margin: float = 0.05) -> Image.Image:
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = img.convert("RGB").crop((
        int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
    return out.resize((CANVAS, CANVAS), Image.LANCZOS).convert("RGBA")


def tile_mosaic(img: Image.Image, size: int = 128) -> Image.Image:
    """Break the icon up into 128 square tiles so a mosaic generator can rebuild it."""
    out = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    grid = CANVAS // size
    for row in range(grid):
        for col in range(grid):
            tile = img.crop((col * size, row * size, (col + 1) * size, (row + 1) * size))
            out.alpha_composite(tile)
    return out


# ------------------------------------------------------------------- build --
def build_painted() -> list[dict]:
    """Collection: the same hand-painted style across several trinkets."""
    style = STYLE_VARIANTS[0]
    out_dir = ICONS / "painted"
    (out_dir / "transparent").mkdir(parents=True, exist_ok=True)
    entries = []

    for item in PAINTED_ITEMS:
        src = MASTERS / "painted" / f"{item['id']}-{item['slug']}.png"
        if not src.exists():
            print(f"  ! skip painted/{item['slug']} (no master yet)")
            continue

        master = Image.open(src).convert("RGBA")
        obj, bg = cut_out(master, style["thr"], style["feather"])
        bg = style.get("bg", bg)          # единый фон для всей коллекции
        obj = scale_object(trim(obj))

        name = f"{item['id']}-{item['slug']}.png"
        compose(obj, bg).save(out_dir / name, optimize=True)
        compose(obj, None).save(out_dir / "transparent" / name, optimize=True)
        entries.append({"style": "painted", **item, "bg": list(bg), "cutout": True})
        print(f"  painted  {name}")

    return entries


def build_styles() -> list[dict]:
    """One trinket, ten art styles."""
    out_dir = ICONS / "styles"
    (out_dir / "transparent").mkdir(parents=True, exist_ok=True)
    entries = []

    for style in STYLE_VARIANTS:
        src = MASTERS / "ten" / f"{style['key']}.png"
        if not src.exists():
            print(f"  ! skip style {style['key']} (no master yet)")
            continue

        master = Image.open(src).convert("RGBA")
        entry = {**ITEM_01, **{k: style[k] for k in ("key", "ru", "en", "mode")}}

        if style["mode"] == "cut":
            obj, bg = cut_out(master, style["thr"], style["feather"])
            bg = style.get("bg", bg)
            obj = scale_object(trim(obj), pixel=style.get("pixel", False))
            icon = compose(obj, bg)
            compose(obj, None).save(out_dir / "transparent" / f"{style['key']}.png", optimize=True)
            entry["bg"] = list(bg)
            entry["cutout"] = True
        elif style["mode"] == "frame":
            icon = square_crop(master, content_box(master), style["margin"])
            entry["bg"] = None
            entry["cutout"] = False
        else:  # mosaic
            panel = master.crop(style["bbox"])
            side = max(panel.size)
            square = Image.new("RGB", (side, side), (10, 10, 12))
            square.paste(panel.convert("RGB"), ((side - panel.width) // 2, (side - panel.height) // 2))
            icon = square.resize((CANVAS, CANVAS), Image.LANCZOS).convert("RGBA")
            entry["bg"] = None
            entry["cutout"] = False
            entry["mosaic"] = True

        icon.save(out_dir / f"{style['key']}.png", optimize=True)
        entry["icon"] = icon
        entries.append(entry)
        print(f"  style    {style['key']}.png")

    return entries


# ---------------------------------------------------------------- previews --
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("/usr/share/fonts/truetype/dejavu") / (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int, max_lines: int = 2) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        probe = f"{line} {word}".strip()
        if d.textlength(probe, font=f) <= width:
            line = probe
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines[:max_lines]


def grid(entries: list[dict], path: Path, dir_for, title: str, subtitle: str,
         cols: int = 5, scale: int = 1) -> None:
    tsize = CANVAS * scale
    gap, pad, head = 20, 26, 82
    f_name, f_sub = font(19, True), font(14)

    rows = (len(entries) + cols - 1) // cols
    w = pad * 2 + cols * tsize + (cols - 1) * gap
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cap = max(
        12 + 24 * len(wrap(measure, e.get("ru", ""), f_name, tsize))
        + 20 * len(wrap(measure, e.get("en", ""), f_sub, tsize))
        for e in entries
    )
    h = pad * 2 + head + rows * (tsize + cap) + (rows - 1) * gap

    img = Image.new("RGB", (w, h), (18, 19, 22))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), title, font=font(28, True), fill=(244, 244, 246))
    d.text((pad, pad + 40), subtitle, font=font(16), fill=(140, 146, 158))

    for i, entry in enumerate(entries):
        col, row = i % cols, i // cols
        x = pad + col * (tsize + gap)
        y = pad + head + row * (tsize + cap + gap)

        icon = Image.open(dir_for(entry)).convert("RGBA")
        if scale != 1:
            icon = icon.resize((tsize, tsize), Image.NEAREST)
        img.paste(icon, (x, y), icon)
        d.rectangle((x, y, x + tsize - 1, y + tsize - 1), outline=(58, 62, 70))

        ly = y + tsize + 12
        for line in wrap(d, entry.get("ru", ""), f_name, tsize):
            d.text((x, ly), line, font=f_name, fill=(238, 238, 240))
            ly += 24
        for line in wrap(d, entry.get("en", ""), f_sub, tsize):
            d.text((x, ly + 2), line, font=f_sub, fill=(138, 144, 156))
            ly += 20

    img.save(path)


def size_strip(entries: list[dict], path: Path, dir_for) -> None:
    sizes = [CANVAS, 128, 64, 32]
    pad, gap, head = 26, 16, 74
    w = pad * 2 + sum(sizes) + gap * (len(sizes) - 1)
    h = pad * 2 + head + len(entries) * (CANVAS + gap) - gap

    img = Image.new("RGB", (w, h), (18, 19, 22))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), "Проверка читаемости: 256 / 128 / 64 / 32 px", font=font(24, True), fill=(244, 244, 246))
    x = pad
    for size in sizes:
        d.text((x, pad + 44), f"{size}px", font=font(14), fill=(140, 146, 158))
        x += size + gap

    for row, entry in enumerate(entries):
        y = pad + head + row * (CANVAS + gap)
        x = pad
        src = Image.open(dir_for(entry)).convert("RGBA")
        for size in sizes:
            img.paste(src.resize((size, size), Image.LANCZOS), (x, y + (CANVAS - size)), None)
            x += size + gap

    img.save(path)


def gallery(painted: list[dict], styles: list[dict], path: Path) -> None:
    def cards(entries, folder, extra):
        out = []
        for e in entries:
            name = f"{e['id']}-{e['slug']}.png" if folder == "painted" else f"{e['key']}.png"
            note = "" if e.get("cutout") else " · фон — часть стиля"
            out.append(f"""
        <figure class="card">
          <div class="preview">
            <img src="icons/{folder}/{name}" alt="{e['ru']}">
            <img class="small" src="icons/{folder}/{name}" alt="" width="64" height="64">
          </div>
          <figcaption>
            <span class="name">{e['ru']}</span>
            <span class="meta">icons/{folder}/{name}{note}{extra(e) if extra else ''}</span>
          </figcaption>
        </figure>""")
        return "".join(out)

    def d100(e):
        return f" · d100: {e['d100']:02d}" if "d100" in e else ""

    path.write_text(f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D&D безделушки — иконки 256×256</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 44px 34px 80px; background: #14161a; color: #e9eaec;
         font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; }}
  h2 {{ margin: 54px 0 6px; font-size: 21px; }}
  p.sub {{ margin: 0 0 30px; color: #949aa6; max-width: 90ch; }}
  p.sub code, .meta {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
  p.sub code {{ color: #c7ccd6; }}
  .grid {{ display: grid; gap: 26px; grid-template-columns: repeat(auto-fill, minmax(268px, 1fr)); }}
  .card {{ margin: 0; }}
  .preview {{ position: relative; border: 1px solid #2c3037; border-radius: 14px;
              overflow: hidden; background: #1b1e23; aspect-ratio: 1; }}
  .preview img {{ display: block; width: 100%; height: auto; }}
  .preview img.small {{ position: absolute; right: 12px; bottom: 12px; width: 64px;
                        height: 64px; border-radius: 6px; border: 1px solid #2c3037; }}
  figcaption {{ display: flex; flex-direction: column; gap: 2px; padding: 12px 2px 0; }}
  .name {{ font-weight: 600; }}
  .meta {{ color: #8d94a1; font-size: 12.5px; }}
</style>
</head>
<body>
  <h1>D&amp;D безделушки — иконки 256×256</h1>
  <p class="sub">Всё сгенерировано в едином стиле «ручная живопись»; снизу — тот же файл в 64&nbsp;px
  для проверки читаемости. Файлы с прозрачным фоном лежат рядом в подпапке <code>transparent/</code>.
  Витраж и мозаика: <code>icons/styles/stained-glass.png</code> + <code>icons/mosaic/</code>.</p>

  <h2>Коллекция «Ручная живопись» — {len(painted)} безделушек</h2>
  <p class="sub">Папка <code>icons/painted/</code></p>
  <div class="grid">{cards(painted, "painted", d100)}</div>

  <h2>Один предмет — {len(styles)} стилей</h2>
  <p class="sub">«{ITEM_01['ru']}» (d100 №01), папка <code>icons/styles/</code></p>
  <div class="grid">{cards(styles, "styles", None)}</div>
</body>
</html>
""", encoding="utf-8")


def main() -> None:
    print("building icons...")
    ICONS.mkdir(exist_ok=True)
    painted = build_painted()
    styles = build_styles()

    # stained glass keeps its detail only at native resolution -> mosaic tiles
    mosaic_dir = ICONS / "mosaic"
    mosaic_dir.mkdir(exist_ok=True)
    for entry in styles:
        if entry.get("mosaic"):
            src = Image.open(ICONS / "styles" / f"{entry['key']}.png").convert("RGBA")
            for row in range(2):
                for col in range(2):
                    tile = src.crop((col * 128, row * 128, (col + 1) * 128, (row + 1) * 128))
                    tile.save(mosaic_dir / f"{entry['key']}-{row}{col}.png", optimize=True)
            print(f"  mosaic   {entry['key']} -> 4 x 128px tiles")

    (ROOT / "manifest.json").write_text(json.dumps(
        {"collection": painted, "styles": [{k: v for k, v in s.items() if k != "icon"} for s in styles]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    PREVIEW.mkdir(exist_ok=True)
    grid(painted, PREVIEW / "painted-collection.png", lambda e: ICONS / "painted" / f"{e['id']}-{e['slug']}.png",
         "Коллекция «Ручная живопись»", f"{len(painted)} безделушек из таблицы d100 · 256×256 PNG · icons/painted/",
         cols=3)
    grid(painted, PREVIEW / "painted-collection-2x.png", lambda e: ICONS / "painted" / f"{e['id']}-{e['slug']}.png",
         "Коллекция «Ручная живопись» (детали ×2)", "то же самое крупным планом · 512×512", cols=3, scale=2)
    grid(styles, PREVIEW / "01-ten-styles.png", lambda e: ICONS / "styles" / f"{e['key']}.png",
         f"«{ITEM_01['ru']}» — {len(styles)} стилей", "один и тот же предмет · 256×256 PNG · icons/styles/",
         cols=5)
    grid(styles, PREVIEW / "01-ten-styles-2x.png", lambda e: ICONS / "styles" / f"{e['key']}.png",
         f"«{ITEM_01['ru']}» — детали ×2", "то же самое крупным планом · 512×512", cols=5, scale=2)
    size_strip(painted, PREVIEW / "painted-sizes.png",
               lambda e: ICONS / "painted" / f"{e['id']}-{e['slug']}.png")
    gallery(painted, styles, ROOT / "index.html")
    print(f"done: {len(painted)} painted icons, {len(styles)} style variants")


if __name__ == "__main__":
    main()
