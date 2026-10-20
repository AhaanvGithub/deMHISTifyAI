"""Draw expert (green) and finder (red) outlines on a field, 3x enlarged, for eyeballing a score."""
from PIL import Image, ImageDraw


def draw(field, gt, pred, path, z=3):
    im = field.convert("RGB").resize((field.width * z, field.height * z), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    for cs, col in ((gt, (0, 200, 0)), (pred, (230, 0, 0))):
        for c in cs:
            pts = [(x * im.width, y * im.height) for x, y in c["outline"]]
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill=col, width=2)
    im.save(path)
