#!/usr/bin/env python3
"""
D&D Trinkets — коллекция иконок 256x256 в стиле «ручная живопись».

Данные о предметах — data/trinkets.json (официальные описания PHB:
английский — Wizards of the Coast, русский — перевод Hobby Games, стр. 159).

Пайплайн:
    masters/painted/<id>-<slug>.png   исходники 1024x1024
        → вырезаем предмет из плоского фона
        → масштабируем и центрируем на холсте 256x256
        → собираем «плитку»: мягкая подсветка за предметом + тень под ним
    icons/painted/<id>-<slug>.png             иконка для интерфейса
    icons/painted/transparent/<id>-<slug>.png предмет без фона (для движка)
    icons/painted/metadata.csv                таблица описаний для импорта

Превью:
    preview/pack-sheet.png   страница пака (заголовок + демо-ряд + плотная сетка)
    preview/spec-sheet.png   контактный лист с подписями и статусами сверки
    preview/icon-sizes.png   читаемость в 256/128/64/32 px

Запуск:  python3 scripts/build_icons.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data" / "trinkets.json").read_text(encoding="utf-8"))
ITEMS: list[dict] = DATA["items"]
STYLE = DATA["style"]

MASTERS = ROOT / "masters" / "painted"
ICONS = ROOT / "icons" / "painted"
PREVIEW = ROOT / "preview"

CANVAS: int = DATA["icons"]["size"]                 # 256
OBJECT_BOX: int = DATA["icons"]["object_box"]       # 202 (~79% кадра)
BG: tuple[int, int, int] = tuple(
    int(DATA["icons"]["bg"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
THR = 0.035         # допуск «это фон» при вырезании
FEATHER = 0.6       # сглаживание края маски

BCK = DATA["icons"]["background"]          # градиент фона плитки
SHADOW_ALPHA = 0.55 # плотность контактной тени под предметом
SHADOW_BLUR = 10
SHADOW_DROP = 8
TILE_RADIUS = 12    # скругление плиток в превью

STATUS_COLOUR = {"ok": (110, 190, 130), "fixed": (230, 180, 90), "todo": (220, 110, 110)}
STATUS_RU = {"ok": "сверено с описанием", "fixed": "картинка поправлена по описанию",
             "todo": "требует сверки"}


def icon_name(item: dict) -> str:
    return f"{item['id']}-{item['slug']}.png"


def icon_path(item: dict) -> Path:
    return ICONS / icon_name(item)


# ------------------------------------------------------------- вырезание ---
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


def cut_out(img: Image.Image, thr: float, feather: float) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.int16)
    dist = np.abs(arr[:, :, :3] - background_colour(arr)).sum(axis=2) / 3.0

    mask = ndimage.binary_fill_holes(largest_component(dist > thr * 255))
    # убираем 2 px по краю: там пиксели смешаны с прежним фоном и на чёрном
    # фоне дают светлую кайму
    mask = ndimage.binary_erosion(mask, iterations=2)
    alpha = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))

    out = img.copy()
    out.putalpha(alpha)
    return out


def trim(img: Image.Image) -> Image.Image:
    bbox = img.getchannel("A").point(lambda v: 255 if v > 8 else 0).getbbox()
    return img.crop(bbox) if bbox else img


def scale_object(img: Image.Image, box: int = OBJECT_BOX) -> Image.Image:
    k = box / max(img.size)
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)


# --------------------------------------------------------- фон и плитка ----
def backdrop(size: int) -> Image.Image:
    """Фон плитки: красивый градиент на чёрном — светлее в центре, чернота к краям."""
    y, x = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    r = np.sqrt(((x - c) / c) ** 2 + ((y - c) / c) ** 2)
    # лёгкий вертикальный сдвиг: центр подсветки чуть выше середины
    r = np.sqrt(r ** 2 + BCK["bias"] * ((y - c) / c))
    k = np.clip(1.0 - (r / 1.35) ** BCK["falloff"], 0.0, 1.0)
    centre = np.array(BCK["center"], np.float32)
    edge = np.array(BCK["edge"], np.float32)
    base = edge[None, None, :] + (centre - edge)[None, None, :] * k[:, :, None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def compose(img: Image.Image, size: int = CANVAS, bg: tuple[int, int, int] | None = BG,
            shadow: bool = True) -> Image.Image:
    """Собрать иконку: подсвеченный фон, тень под предметом, сам предмет."""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pos = ((size - img.width) // 2, (size - img.height) // 2)
    if bg:
        base = backdrop(size)
        if shadow:
            # силуэт предмета целиком на холсте → размытие и сдвиг вниз
            silhouette = Image.new("L", (size, size), 0)
            silhouette.paste(img.getchannel("A"), pos)
            soft = silhouette.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
            soft = ImageChops.offset(soft, 0, SHADOW_DROP)
            dark = np.asarray(base, np.float32) * (1 - SHADOW_ALPHA * np.asarray(soft, np.float32)[:, :, None] / 255)
            base = Image.fromarray(np.clip(dark, 0, 255).astype(np.uint8), "RGB")
        canvas.paste(base, (0, 0))
    canvas.alpha_composite(img, pos)
    return canvas


def make_icon(master: Image.Image, item: dict) -> tuple[Image.Image, Image.Image]:
    obj = trim(cut_out(master, item.get("thr", THR), item.get("feather", FEATHER)))
    obj = scale_object(obj, item.get("box", OBJECT_BOX))
    return compose(obj, CANVAS, BG, shadow=True), compose(obj, CANVAS, None, shadow=False)


# --------------------------------------------------------------- превью ----
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("/usr/share/fonts/truetype/dejavu") / (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int, max_lines: int = 3) -> list[str]:
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


def rounded(icon: Image.Image, radius: int = TILE_RADIUS) -> Image.Image:
    """Скруглить плитку и добавить тонкую внутреннюю рамку."""
    s = icon.size[0]
    r = max(2, round(radius * s / CANVAS))
    mask = Image.new("L", (s * 4, s * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s * 4 - 1, s * 4 - 1), radius=r * 4, fill=255)
    out = icon.copy()
    out.putalpha(mask.resize((s, s), Image.LANCZOS))
    ImageDraw.Draw(out).rounded_rectangle((0, 0, s - 1, s - 1), radius=r, outline=(255, 255, 255, 30), width=1)
    return out


def pack_sheet(path: Path) -> None:
    """Страница пака: заголовок, демо-ряд, плотная сетка плиток без подписей."""
    W = 1200
    pad = 28
    cols, tile, gap = 9, 120, 8
    demo_sizes = [224, 448, 224]
    demo_items = ["44", "01", "02"]

    grid_w = cols * tile + (cols - 1) * gap
    demo_w = sum(demo_sizes) + 2 * 24
    head_h, demo_h, foot_h = 148, max(demo_sizes) + 16, 74
    rows = (len(ITEMS) + cols - 1) // cols
    H = pad + head_h + demo_h + 22 + rows * tile + (rows - 1) * gap + foot_h + pad

    img = Image.new("RGB", (W, H), (13, 14, 17))
    d = ImageDraw.Draw(img)

    d.text((pad, pad), "D&D TRINKETS — RPG ICONS", font=font(40, True), fill=(248, 248, 250))
    d.text((pad, pad + 56), f"{len(ITEMS)} icons      d100 (PHB)      {CANVAS}x{CANVAS} pixels size",
           font=font(19, True), fill=(206, 210, 218))
    d.text((pad, pad + 86), f"transparent + backup  ·  {DATA['source']['ru']}", font=font(14), fill=(140, 146, 158))
    d.text((pad, pad + 106), "стиль «ручная живопись»: эталон — "
                             + ", ".join(f"{i}" for i in ("01 rука гоблина", "44 канарейка")),
           font=font(14), fill=(140, 146, 158))

    by_id = {i["id"]: i for i in ITEMS}
    x = (W - demo_w) // 2
    y = pad + head_h
    for size, item_id in zip(demo_sizes, demo_items):
        item = by_id[item_id]
        icon = rounded(Image.open(icon_path(item)).convert("RGBA").resize((size, size), Image.LANCZOS))
        img.paste(icon, (x, y + (max(demo_sizes) - size) // 2), icon)
        x += size + 24

    x0 = (W - grid_w) // 2
    y = pad + head_h + demo_h + 22
    for i, item in enumerate(ITEMS):
        cx = x0 + (i % cols) * (tile + gap)
        cy = y + (i // cols) * (tile + gap)
        icon = rounded(Image.open(icon_path(item)).convert("RGBA").resize((tile, tile), Image.LANCZOS))
        img.paste(icon, (cx, cy), icon)

    fy = y + rows * tile + (rows - 1) * gap + 26
    f_idx = font(13)
    index = "   ".join(f"{i['d100']:02d} {i['ru']}" for i in ITEMS)
    for n, line in enumerate(wrap(d, index, f_idx, W - pad * 2)):
        d.text((x0, fy + n * 18), line, font=f_idx, fill=(150, 156, 168))

    img.save(path)


def spec_sheet(path: Path, cols: int = 5, scale: int = 1) -> None:
    """Контактный лист с подписями и статусами сверки."""
    frame = int(CANVAS * scale)
    gap, pad, head = 18, 26, 128
    f_name, f_meta = font(16, True), font(13)
    rows = (len(ITEMS) + cols - 1) // cols
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    titles = [wrap(measure, f"{i['d100']:02d} · {i['ru']}", f_name, frame) for i in ITEMS]
    label = max(26 + 22 * len(t) for t in titles)

    w = pad * 2 + cols * frame + (cols - 1) * gap
    h = pad * 2 + head + rows * (frame + label) + (rows - 1) * gap
    img = Image.new("RGB", (w, h), (16, 17, 20))
    d = ImageDraw.Draw(img)

    d.text((pad, pad), "D&D TRINKETS — ICONS", font=font(30, True), fill=(245, 245, 247))
    d.text((pad, pad + 42), f"{len(ITEMS)} trinkets · d100 · {CANVAS}x{CANVAS} PNG · transparent + backup · "
                           f"{DATA['source']['ru']}", font=font(15), fill=(150, 156, 168))
    d.text((pad, pad + 64), "под каждой иконкой — номер по d100 и официальное название из Книги игрока",
           font=font(15), fill=(150, 156, 168))
    lx = pad
    for key, colour in STATUS_COLOUR.items():
        d.ellipse((lx, pad + 96, lx + 9, pad + 105), fill=colour)
        d.text((lx + 15, pad + 92), STATUS_RU[key], font=font(13), fill=(150, 156, 168))
        lx += 30 + int(d.textlength(STATUS_RU[key], font=font(13)))

    for i, item in enumerate(ITEMS):
        x = pad + (i % cols) * (frame + gap)
        y = pad + head + (i // cols) * (frame + label + gap)
        icon = Image.open(icon_path(item)).convert("RGBA")
        if scale != 1:
            icon = icon.resize((frame, frame), Image.NEAREST)
        icon = rounded(icon)
        img.paste(icon, (x, y), icon)

        ly = y + frame + 8
        for line in titles[i]:
            d.text((x, ly), line, font=f_name, fill=(238, 238, 240))
            ly += 22
        status = item["icon_check"]["status"]
        d.ellipse((x, ly + 5, x + 9, ly + 14), fill=STATUS_COLOUR[status])
        d.text((x + 15, ly), item["en"][:34], font=f_meta, fill=(138, 144, 156))

    img.save(path)


def size_sheet(path: Path) -> None:
    sizes = [CANVAS, 128, 64, 32]
    pad, gap, head = 26, 22, 128
    w = pad * 2 + sum(sizes) + gap * (len(sizes) - 1)
    h = pad * 2 + head + CANVAS
    img = Image.new("RGB", (w, h), (16, 17, 20))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), "D&D TRINKETS — ICONS", font=font(30, True), fill=(245, 245, 247))
    d.text((pad, pad + 42), "читаемость в интерфейсе: одна и та же иконка в 256 / 128 / 64 / 32 px",
           font=font(15), fill=(150, 156, 168))
    x = pad
    for size in sizes:
        icon = rounded(Image.open(icon_path(ITEMS[0])).convert("RGBA").resize((size, size), Image.LANCZOS),
                       radius=TILE_RADIUS * size / CANVAS)
        img.paste(icon, (x, pad + head + CANVAS - size), icon)
        d.text((x, pad + head + CANVAS + 4), f"{size}px", font=font(13), fill=(138, 144, 156))
        x += size + gap
    img.save(path)


def gallery(path: Path) -> None:
    cards = []
    for item in ITEMS:
        status = item["icon_check"]["status"]
        mark = {"ok": "", "fixed": " · сверено, картинка поправлена", "todo": " · требует сверки"}[status]
        cards.append(f"""
      <figure class="card">
        <div class="preview">
          <img src="icons/painted/{icon_name(item)}" alt="{item['ru']}">
          <img class="small" src="icons/painted/{icon_name(item)}" alt="" width="64" height="64">
        </div>
        <figcaption>
          <span class="name">{item['ru']}</span>
          <span class="src">{item['en']}</span>
          <span class="meta">d100: {item['d100']:02d} · icons/painted/{icon_name(item)}{mark}</span>
        </figcaption>
      </figure>""")

    path.write_text(f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D&D Trinkets — иконки {CANVAS}×{CANVAS}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 44px 34px 80px; background: #14161a; color: #e9eaec;
         font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: .3px; }}
  p.sub {{ margin: 0 0 8px; color: #949aa6; max-width: 100ch; }}
  p.sub.small {{ font-size: 13px; margin-bottom: 30px; }}
  p.sub code, .meta {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
  p.sub code {{ color: #c7ccd6; }}
  .grid {{ display: grid; gap: 26px; grid-template-columns: repeat(auto-fill, minmax(268px, 1fr)); }}
  .card {{ margin: 0; }}
  .preview {{ position: relative; border: 1px solid #2c3037; border-radius: 14px;
              overflow: hidden; background: #1b1e23; aspect-ratio: 1; }}
  .preview img {{ display: block; width: 100%; height: auto; }}
  .preview img.small {{ position: absolute; right: 12px; bottom: 12px; width: 64px;
                        height: 64px; border-radius: 6px; border: 1px solid #2c3037; }}
  figcaption {{ display: flex; flex-direction: column; gap: 3px; padding: 12px 2px 0; }}
  .name {{ font-weight: 600; }}
  .src {{ color: #a9b0bd; font-size: 13px; }}
  .meta {{ color: #8d94a1; font-size: 12.5px; }}
</style>
</head>
<body>
  <h1>D&amp;D Trinkets — иконки {CANVAS}×{CANVAS}</h1>
  <p class="sub">Коллекция в закреплённом стиле «ручная живопись»: {len(ITEMS)} безделушек из таблицы d100.
  Эталон стиля — 01 «Мумифицированная рука гоблина» и 44 «Механическая канарейка в гномьей лампе».
  Названия и описания официальные: английский текст из Player's Handbook (Wizards of the Coast),
  русский — из перевода Hobby Games. Картинка каждого предмета сверена с описанием
  (пометка «сверено, картинка поправлена» = на иконке убраны детали, которых в описании нет).</p>
  <p class="sub small">Файлы: <code>icons/painted/&lt;id&gt;-&lt;slug&gt;.png</code> — иконка для интерфейса;
  <code>icons/painted/transparent/</code> — предмет без фона; <code>icons/painted/metadata.csv</code> — таблица описаний;
  <code>data/trinkets.json</code> — источник данных; <code>preview/pack-sheet.png</code> — страница пака.</p>
  <div class="grid">{''.join(cards)}
  </div>
</body>
</html>
""", encoding="utf-8")


def metadata_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "d100", "slug", "name_ru", "name_en", "description_source",
                     "icon", "icon_transparent", "icon_check", "icon_check_note"])
        for item in ITEMS:
            wr.writerow([
                item["id"], item["d100"], item["slug"], item["ru"], item["en"],
                DATA["source"]["ru"],
                f"icons/painted/{icon_name(item)}",
                f"icons/painted/transparent/{icon_name(item)}",
                item["icon_check"]["status"], item["icon_check"]["note"],
            ])


def main() -> None:
    print("building icons...")
    ICONS.mkdir(parents=True, exist_ok=True)
    (ICONS / "transparent").mkdir(parents=True, exist_ok=True)

    for item in ITEMS:
        src = MASTERS / icon_name(item)
        if not src.exists():
            print(f"  ! skip {item['slug']} (нет мастер-файла)")
            continue
        icon, transparent = make_icon(Image.open(src), item)
        icon.save(icon_path(item), optimize=True)
        transparent.save(ICONS / "transparent" / icon_name(item), optimize=True)
        print(f"  {icon_name(item)}")

    metadata_csv(ICONS / "metadata.csv")
    (ROOT / "manifest.json").write_text(json.dumps({
        "source": DATA["source"],
        "style": STYLE,
        "icons": DATA["icons"],
        "items": [{**i, "files": {
            "icon": f"icons/painted/{icon_name(i)}",
            "transparent": f"icons/painted/transparent/{icon_name(i)}"}} for i in ITEMS],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    PREVIEW.mkdir(exist_ok=True)
    pack_sheet(PREVIEW / "pack-sheet.png")
    spec_sheet(PREVIEW / "spec-sheet.png")
    size_sheet(PREVIEW / "icon-sizes.png")
    gallery(ROOT / "index.html")
    print(f"done: {len(ITEMS)} icons")


if __name__ == "__main__":
    main()
