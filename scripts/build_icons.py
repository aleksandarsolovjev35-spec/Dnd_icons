#!/usr/bin/env python3
"""
D&D Trinkets — коллекция иконок 256x256 в стиле «ручная живопись».

Пайплайн:
    masters/painted/<id>-<slug>.png   исходники 1024x1024
        → вырезаем предмет из плоского фона
        → масштабируем и центрируем на холсте 256x256
    icons/painted/<id>-<slug>.png            готовая иконка
    icons/painted/transparent/<id>-<slug>.png иконка с прозрачным фоном

Плюс превью в preview/ и статичная галерея index.html.

Запуск:  python3 scripts/build_icons.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MASTERS = ROOT / "masters" / "painted"
ICONS = ROOT / "icons" / "painted"
PREVIEW = ROOT / "preview"

CANVAS = 256       # размер иконки
OBJECT_BOX = 202   # длинная сторона предмета внутри иконки (~79%)
BG = (40, 50, 58)  # единый фон коллекции, #28323A
THR = 0.035        # допуск «это фон» при вырезании
FEATHER = 0.6      # сглаживание края маски

# ---------------------------------------------------------------- каталог ---
# thr/feather можно переопределить для отдельных предметов (свечение, тени)
ITEMS = [
    dict(id="01", d100=1, slug="mummified-goblin-hand",
         ru="Мумифицированная рука гоблина", en="A mummified goblin hand"),
    dict(id="02", d100=2, slug="moonlight-crystal",
         ru="Кристалл, светящийся в лунном свете", en="A piece of crystal that faintly glows in the moonlight"),
    dict(id="06", d100=6, slug="glass-chess-piece",
         ru="Старая стеклянная шахматная фигура", en="An old chess piece made from glass"),
    dict(id="07", d100=7, slug="skull-knucklebones",
         ru="Кости-«шестёрки» с черепом", en="A pair of knucklebone dice with a skull symbol on the six"),
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


# ------------------------------------------------------------- вырезание ---
def background_colour(arr: np.ndarray) -> np.ndarray:
    """Медианный цвет рамки вокруг изображения = плоский фона."""
    ring = np.concatenate([
        arr[0:4, :, :3].reshape(-1, 3), arr[-4:, :, :3].reshape(-1, 3),
        arr[:, 0:4, :3].reshape(-1, 3), arr[:, -4:, :3].reshape(-1, 3),
    ])
    return np.median(ring, axis=0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Оставляем только самый крупный объект: мягкая тень под предметом не попадает."""
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    areas = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == int(np.argmax(areas)) + 1


def cut_out(img: Image.Image, thr: float, feather: float) -> tuple[Image.Image, tuple[int, int, int]]:
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.int16)
    bg = background_colour(arr)
    dist = np.abs(arr[:, :, :3] - bg).sum(axis=2) / 3.0

    mask = ndimage.binary_fill_holes(largest_component(dist > thr * 255))
    alpha = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    out.putalpha(alpha)
    return out, tuple(int(c) for c in bg)


def trim(img: Image.Image) -> Image.Image:
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return img.crop(bbox) if bbox else img


def scale_object(img: Image.Image, box: int = OBJECT_BOX) -> Image.Image:
    k = box / max(img.size)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


def compose(img: Image.Image, bg: tuple[int, int, int] | None) -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS, CANVAS), bg + (255,) if bg else (0, 0, 0, 0))
    canvas.alpha_composite(img, ((CANVAS - img.width) // 2, (CANVAS - img.height) // 2))
    return canvas


def make_icon(master: Image.Image, item: dict) -> tuple[Image.Image, Image.Image]:
    obj, _ = cut_out(master, item.get("thr", THR), item.get("feather", FEATHER))
    obj = scale_object(trim(obj), item.get("box", OBJECT_BOX))
    return compose(obj, BG), compose(obj, None)


# --------------------------------------------------------------- превью ----
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


def icon_path(item: dict) -> Path:
    return ICONS / f"{item['id']}-{item['slug']}.png"


def grid(items: list[dict], path: Path, cols: int = 3, scale: int = 1) -> None:
    tsize = CANVAS * scale
    gap, pad, head = 20, 26, 82
    f_name, f_sub = font(19, True), font(14)
    rows = (len(items) + cols - 1) // cols
    w = pad * 2 + cols * tsize + (cols - 1) * gap
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    cap = max(12 + 24 * len(wrap(measure, i["ru"], f_name, tsize))
              + 20 * len(wrap(measure, i["en"], f_sub, tsize)) for i in items)
    h = pad * 2 + head + rows * (tsize + cap) + (rows - 1) * gap

    img = Image.new("RGB", (w, h), (18, 19, 22))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), f"Коллекция «Ручная живопись»{' (детали ×2)' if scale > 1 else ''}",
           font=font(28, True), fill=(244, 244, 246))
    d.text((pad, pad + 40), f"{len(items)} безделушек из таблицы d100 · {CANVAS}×{CANVAS} PNG · icons/painted/",
           font=font(16), fill=(140, 146, 158))

    for i, item in enumerate(items):
        x = pad + (i % cols) * (tsize + gap)
        y = pad + head + (i // cols) * (tsize + cap + gap)
        icon = Image.open(icon_path(item)).convert("RGBA")
        if scale != 1:
            icon = icon.resize((tsize, tsize), Image.NEAREST)
        img.paste(icon, (x, y), icon)
        d.rectangle((x, y, x + tsize - 1, y + tsize - 1), outline=(58, 62, 70))

        ly = y + tsize + 12
        for line in wrap(d, item["ru"], f_name, tsize):
            d.text((x, ly), line, font=f_name, fill=(238, 238, 240))
            ly += 24
        for line in wrap(d, item["en"], f_sub, tsize):
            d.text((x, ly + 2), line, font=f_sub, fill=(138, 144, 156))
            ly += 20

    img.save(path)


def size_strip(items: list[dict], path: Path) -> None:
    sizes = [CANVAS, 128, 64, 32]
    pad, gap, head = 26, 16, 74
    w = pad * 2 + sum(sizes) + gap * (len(sizes) - 1)
    h = pad * 2 + head + len(items) * (CANVAS + gap) - gap

    img = Image.new("RGB", (w, h), (18, 19, 22))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), "Проверка читаемости: 256 / 128 / 64 / 32 px", font=font(24, True), fill=(244, 244, 246))
    x = pad
    for size in sizes:
        d.text((x, pad + 44), f"{size}px", font=font(14), fill=(140, 146, 158))
        x += size + gap

    for row, item in enumerate(items):
        y = pad + head + row * (CANVAS + gap)
        x = pad
        src = Image.open(icon_path(item)).convert("RGBA")
        for size in sizes:
            img.paste(src.resize((size, size), Image.LANCZOS), (x, y + (CANVAS - size)), None)
            x += size + gap

    img.save(path)


def gallery(items: list[dict], path: Path) -> None:
    cards = "".join(f"""
      <figure class="card">
        <div class="preview">
          <img src="icons/painted/{i['id']}-{i['slug']}.png" alt="{i['ru']}">
          <img class="small" src="icons/painted/{i['id']}-{i['slug']}.png" alt="" width="64" height="64">
        </div>
        <figcaption>
          <span class="name">{i['ru']}</span>
          <span class="meta">icons/painted/{i['id']}-{i['slug']}.png · d100: {i['d100']:02d}</span>
        </figcaption>
      </figure>""" for i in items)

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
  <p class="sub">Коллекция в едином стиле «ручная живопись» ({len(items)} предметов из таблицы d100).
  Снизу справа — тот же файл в 64&nbsp;px для проверки читаемости, версии с прозрачным фоном лежат
  в <code>icons/painted/transparent/</code>.</p>
  <div class="grid">{cards}
  </div>
</body>
</html>
""", encoding="utf-8")


# ----------------------------------------------------------------- сборка --
def main() -> None:
    print("building icons...")
    ICONS.mkdir(parents=True, exist_ok=True)
    (ICONS / "transparent").mkdir(parents=True, exist_ok=True)
    manifest = []

    for item in ITEMS:
        src = MASTERS / f"{item['id']}-{item['slug']}.png"
        if not src.exists():
            print(f"  ! skip {item['slug']} (нет мастер-файла)")
            continue

        icon, transparent = make_icon(Image.open(src), item)
        name = f"{item['id']}-{item['slug']}.png"
        icon.save(ICONS / name, optimize=True)
        transparent.save(ICONS / "transparent" / name, optimize=True)
        manifest.append({**item, "bg": list(BG)})
        print(f"  {name}")

    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    PREVIEW.mkdir(exist_ok=True)
    grid(manifest, PREVIEW / "painted-collection.png")
    grid(manifest, PREVIEW / "painted-collection-2x.png", scale=2)
    size_strip(manifest, PREVIEW / "painted-sizes.png")
    gallery(manifest, ROOT / "index.html")
    print(f"done: {len(manifest)} icons")


if __name__ == "__main__":
    main()
