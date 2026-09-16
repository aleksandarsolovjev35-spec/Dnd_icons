#!/usr/bin/env python3
"""
D&D Trinkets — коллекция иконок 256x256 в стиле «ручная живопись».

Данные о предметах берутся из data/trinkets.json (официальные описания PHB:
английский — Wizards of the Coast, русский — перевод Hobby Games, стр. 159).

Пайплайн:
    masters/painted/<id>-<slug>.png   исходники 1024x1024
        → вырезаем предмет из плоского фона
        → масштабируем и центрируем на холсте 256x256
    icons/painted/<id>-<slug>.png             готовая иконка (фон #28323A)
    icons/painted/transparent/<id>-<slug>.png иконка с прозрачным фоном
    icons/painted/metadata.csv                таблица для импорта в движок/таблицу

Плюс превью в preview/ (контактный лист в духе паков a-ravlik) и галерея index.html.

Запуск:  python3 scripts/build_icons.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data" / "trinkets.json").read_text(encoding="utf-8"))
ITEMS: list[dict] = DATA["items"]

MASTERS = ROOT / "masters" / "painted"
ICONS = ROOT / "icons" / "painted"
PREVIEW = ROOT / "preview"

CANVAS: int = DATA["icons"]["size"]                 # 256
OBJECT_BOX: int = DATA["icons"]["object_box"]       # 202 (~79% кадра)
BG: tuple[int, int, int] = tuple(
    int(DATA["icons"]["bg"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
THR = 0.035        # допуск «это фон» при вырезании
FEATHER = 0.6      # сглаживание края маски

MARK_STYLE = {      # цвет пометки в CSV и галерее
    "ok": "verified",
    "fixed": "fixed",
    "todo": "needs-review",
}


def icon_name(item: dict) -> str:
    return f"{item['id']}-{item['slug']}.png"


def icon_path(item: dict) -> Path:
    return ICONS / icon_name(item)


# ------------------------------------------------------------- вырезание ---
def background_colour(arr: np.ndarray) -> np.ndarray:
    """Медианный цвет рамки изображения = плоский фон генерации."""
    ring = np.concatenate([
        arr[0:4, :, :3].reshape(-1, 3), arr[-4:, :, :3].reshape(-1, 3),
        arr[:, 0:4, :3].reshape(-1, 3), arr[:, -4:, :3].reshape(-1, 3),
    ])
    return np.median(ring, axis=0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Только самый крупный объект: мягкая тень под предметом не попадает в маску."""
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    areas = ndimage.sum(mask, labels, range(1, count + 1))
    return labels == int(np.argmax(areas)) + 1


def cut_out(img: Image.Image, thr: float, feather: float) -> Image.Image:
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
    return out


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
    obj = cut_out(master, item.get("thr", THR), item.get("feather", FEATHER))
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


STATUS_COLOUR = {"ok": (110, 190, 130), "fixed": (230, 180, 90), "todo": (220, 110, 110)}
STATUS_RU = {"ok": "сверено с описанием", "fixed": "картинка поправлена по описанию",
             "todo": "требует сверки"}


def header(d: ImageDraw.ImageDraw, pad: int, extra: str = "", legend_y: int | None = None) -> None:
    d.text((pad, pad), "D&D TRINKETS — ICONS", font=font(30, True), fill=(245, 245, 247))
    d.text((pad, pad + 42), f"{len(ITEMS)} trinkets  ·  d100  ·  {CANVAS}x{CANVAS} PNG  ·  "
                           f"transparent + backup  ·  {DATA['source']['ru']}",
           font=font(15), fill=(150, 156, 168))
    if extra:
        d.text((pad, pad + 64), extra, font=font(15), fill=(150, 156, 168))
    if legend_y is not None:
        x = pad
        for key, colour in STATUS_COLOUR.items():
            d.ellipse((x, legend_y + 4, x + 9, legend_y + 13), fill=colour)
            d.text((x + 15, legend_y), STATUS_RU[key], font=font(13), fill=(150, 156, 168))
            x += 30 + int(d.textlength(STATUS_RU[key], font=font(13)))


def grid(path: Path, cols: int = 5, scale: int = 1, head: int = 112, extra: str = "") -> None:
    """Контактный лист: плитка с иконкой + подпись, как в превью паков иконок."""
    tsize = frame = int(CANVAS * scale)
    gap, pad = 18, 26
    f_name, f_meta = font(16, True), font(13)
    rows = (len(ITEMS) + cols - 1) // cols
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    titles = [wrap(measure, f"{i['d100']:02d} · {i['ru']}", f_name, frame) for i in ITEMS]
    label = max(26 + 22 * len(t) for t in titles)

    w = pad * 2 + cols * frame + (cols - 1) * gap
    h = pad * 2 + head + 30 + rows * (frame + label) + (rows - 1) * gap

    img = Image.new("RGB", (w, h), (16, 17, 20))
    d = ImageDraw.Draw(img)
    header(d, pad, extra, legend_y=pad + 92)

    for i, item in enumerate(ITEMS):
        x = pad + (i % cols) * (frame + gap)
        y = pad + head + 30 + (i // cols) * (frame + label + gap)

        icon = Image.open(icon_path(item)).convert("RGBA")
        if scale != 1:
            icon = icon.resize((tsize, tsize), Image.NEAREST)
        img.paste(icon, (x, y), icon)
        d.rectangle((x, y, x + frame - 1, y + frame - 1), outline=(52, 56, 64))

        ly = y + frame + 8
        for line in titles[i]:
            d.text((x, ly), line, font=f_name, fill=(238, 238, 240))
            ly += 22

        status = item["icon_check"]["status"]
        d.ellipse((x, ly + 5, x + 9, ly + 14), fill=STATUS_COLOUR[status])
        d.text((x + 15, ly), f"{item['en'][:34]}", font=f_meta, fill=(138, 144, 156))

    img.save(path)


def hero(path: Path) -> None:
    """Строка «бегунок»: одна иконка крупно + размерный ряд (256/128/64/32)."""
    sizes = [CANVAS, 128, 64, 32]
    pad, gap, head = 26, 22, 96
    w = pad * 2 + CANVAS + gap + sum(sizes) + gap * (len(sizes) - 1)
    h = pad * 2 + head + CANVAS

    img = Image.new("RGB", (w, h), (16, 17, 20))
    d = ImageDraw.Draw(img)
    header(d, pad, "читаемость в интерфейсе: одна и та же иконка в 256 / 128 / 64 / 32 px")

    x = pad
    for size in sizes:
        icon = Image.open(icon_path(ITEMS[0])).convert("RGBA").resize((size, size), Image.LANCZOS)
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
  <p class="sub">Коллекция в едином стиле «ручная живопись»: {len(ITEMS)} безделушек из таблицы d100.
  Названия и описания — официальные: английский текст из Player's Handbook (Wizards of the Coast),
  русский — из перевода Hobby Games. Картинка каждого предмета сверена с описанием
  (пометка «сверено, картинка поправлена» = на иконке убраны детали, которых в описании нет).</p>
  <p class="sub small">Файлы: <code>icons/painted/&lt;id&gt;-&lt;slug&gt;.png</code> — фон коллекции;
  <code>icons/painted/transparent/</code> — прозрачный фон; <code>icons/painted/metadata.csv</code> — таблица описаний;
  <code>data/trinkets.json</code> — источник данных.</p>
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
                MARK_STYLE[item["icon_check"]["status"]],
                item["icon_check"]["note"],
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
        "icons": DATA["icons"],
        "items": [{**i, "files": {
            "icon": f"icons/painted/{icon_name(i)}",
            "transparent": f"icons/painted/transparent/{icon_name(i)}"}} for i in ITEMS],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    PREVIEW.mkdir(exist_ok=True)
    grid(PREVIEW / "collection-sheet.png", cols=5,
         extra="под каждой иконкой — номер по d100 и официальное название из Книги игрока")
    grid(PREVIEW / "collection-sheet-2x.png", cols=4, scale=2,
         extra="то же крупным планом, детали видно в натуральную величину")
    hero(PREVIEW / "icon-sizes.png")
    gallery(ROOT / "index.html")
    print(f"done: {len(ITEMS)} icons")


if __name__ == "__main__":
    main()
