"""
Configure Mask and Reference Guide Videos / Build Mask and Reference Guide Video
===============================================================================

Builds guide videos in the layout used by Alissonerdx's ltx23_inpaint_masked_r2v (the layout that
BFSNodes' Reserved Region Frame Composer produces):

  - a "reference zone" strip along one side of the frame, filled with a solid
    color (chroma green by default), with the reference image shown on a white
    card centered in it;
  - the rest of the frame holding the video, scaled uniformly to fit, with
    black bars where it doesn't fill the space, and the masked area painted
    with a solid color (magenta by default).

Configure Mask and Reference Guide Videos makes a configuration object (the
reference image and all layout settings) that one or more Build nodes can
share, so both stages of a two-stage workflow lay their guides out the same
way.

With the default configuration the layout matches Reserved Region Frame
Composer's (zone on the left, 256 px, card at 90% of the zone less 12 px
padding, 16 px white border around the reference). Differences from it:

  - The reference is composited onto white with its alpha and cropped to the
    bounding box of its non-transparent pixels before the 16 px border is
    added, so the object fills more of the card. A reference with no
    non-transparent pixels gives a blank white card ("no reference" mode).
  - The card is fully opaque (no faint zone color at soft edges).
  - The mask is never Lanczos-resampled: it's area-averaged to the video's
    new size and thresholded at 0.5, so the painted area is solid with clean
    edges, and it can be pooled to LTX's 32x32 / 8-frame latent blocks
    (exactly as LTXV Pool Mask Blocks with "max" does) on the final canvas
    grid. Pooled blocks are clipped to the video's rectangle.
  - Video frames are resampled in floating point (no 8-bit rounding), and not
    resampled at all when they keep their size.

The LoRA turns the video area of the guide into the whole output frame. In
training the two always had the same aspect ratio (the video was shrunk
uniformly, with black bars above and below), so auto_resize keeps the frame's
aspect ratio as close to the video's as 32 px steps allow, and only uses zone
thicknesses that are multiples of 32, so the zone's edge is on a latent-cell
boundary as it was in training. The output then needs resizing back to the
input's size; the report gives the factors.

Layout choice (see plan_layout): among the zone sides the configuration
allows, the node scores each layout by how much video detail it keeps (most
important), how much reference detail it keeps, whether the side is the
preferred one (a tiebreaker) and whether the reference is rotated (a mild
penalty). Zone thickness and (with auto_resize) the frame size are chosen
from video concerns only.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

from .mask_block_pooling import pool_mask_blocks

CONFIG_TYPE_NAME = "LTXV_MASK_REF_GUIDE_CONFIG"
ConfigType = io.Custom(CONFIG_TYPE_NAME)

DEFAULT_MASK_COLOR = 0xFF00FF   # magenta, what the LoRA was trained with
DEFAULT_ZONE_COLOR = 0x00FF00   # chroma green, what the LoRA was trained with

# Card geometry, as in Reserved Region Frame Composer.
CARD_BORDER_SOURCE_PX = 16   # white border, in reference-image pixels
ZONE_PADDING_PX = 12         # space between the zone's edges and the card box
CARD_SCALE = 0.90            # the card fits inside 90% of the padded zone

# Mask threshold after area-resampling to the video's new size.
MASK_THRESHOLD = 0.5
# A reference pixel counts as part of the object if its alpha exceeds this.
ALPHA_CROP_THRESHOLD = 1.0 / 255.0

SIDES = ("left", "right", "top", "bottom")
VERTICAL_SIDES = ("left", "right")   # zone is a vertical strip

# zone_location choice -> [(side, preferred)], in preference order.
ZONE_LOCATIONS = {
    "left": [("left", True)],
    "right": [("right", True)],
    "top": [("top", True)],
    "bottom": [("bottom", True)],
}
for _h, _v in (("left", "top"), ("right", "top"), ("left", "bottom"), ("right", "bottom")):
    ZONE_LOCATIONS[f"{_h} or {_v} ({_h} preferred)"] = [(_h, True), (_v, False)]
    ZONE_LOCATIONS[f"{_h} or {_v} ({_v} preferred)"] = [(_v, True), (_h, False)]
ZONE_LOCATION_NAMES = list(ZONE_LOCATIONS)

# ---------------------------------------------------------------------------
# Layout scoring weights (all terms are in log units).
#
# Video detail: 1.0 x log(fraction of the source video's pixels kept).
# Reference detail: 0.5 x log(fraction of the card's pixels kept).
# The others are small adjustments; see video_score and plan_layout.
# ---------------------------------------------------------------------------
W_VIDEO = 1.0
W_REFERENCE = 0.5
W_RESAMPLE = 0.25        # x |log(video scale)|: prefer copying pixels 1:1
W_CANVAS_AREA = 0.25     # x log(frame area / source area): bigger costs VRAM/time
W_THICKNESS = 0.02       # per 32 px of thickness away from the target
# auto_resize only: extra cost per unit of log(video scale) below 1, so that
# keeping the video's full resolution wins over saving a row or column of
# blocks (a slightly smaller video is still possible when it saves a lot).
W_AUTO_SHORTFALL = 2.0
# auto_resize only: x |log(frame aspect / video aspect)|. The LoRA turns the
# video area into the whole output frame, and in training the two always had
# the same aspect ratio (a uniform scale). A mismatch means the output comes
# out stretched by that much, to be corrected by resizing it afterwards.
W_WARP = 2.0
PREFERRED_BONUS = 0.01   # preferred side of a two-side choice
ROTATION_PENALTY = 0.10  # rotating needs ~22% more reference pixels to pay off

# Auto-resize search window for the video's scale.
AUTO_MIN_SCALE = 0.85
AUTO_MAX_SCALE = 1.10
GRID = 32


def _floor_to(x, m):
    return (int(x) // m) * m


def _ceil_to(x, m):
    return -(-int(math.ceil(x)) // m) * m


def unpack_rgb_int(value, default, name):
    """0xRRGGBB int (ComfyUI's Color to RGB Int format) -> (r, g, b) floats 0..1."""
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int (0xRRGGBB), got {type(value).__name__}.")
    value = int(value)
    if not 0 <= value <= 0xFFFFFF:
        raise ValueError(f"{name} must be between 0 and 0xFFFFFF (16777215), got {value}.")
    return ((value >> 16) & 255) / 255.0, ((value >> 8) & 255) / 255.0, (value & 255) / 255.0


def describe_color(rgb):
    r, g, b = (int(round(c * 255)) for c in rgb)
    return f"({r}, {g}, {b})"


# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------

@dataclass
class GuideConfig:
    card: np.ndarray            # (h, w, 3) float32, opaque RGB, border included
    card_note: str
    mask_color: tuple
    zone_color: tuple
    auto_resize: bool
    zone_location: str
    target_thickness: int
    min_thickness: int
    max_thickness: int
    allow_rotation: bool
    rotate_ccw: bool
    max_pixels: int             # negative = no limit


def check_settings(zone_location, target_thickness, min_thickness, max_thickness, max_pixels):
    """Return a list of problems with the widget settings (empty if fine)."""
    problems = []
    if zone_location is not None and zone_location not in ZONE_LOCATIONS:
        problems.append(f"zone_location {zone_location!r} is not one of {ZONE_LOCATION_NAMES}")
    for name, v in (("target_thickness", target_thickness), ("min_thickness", min_thickness),
                    ("max_thickness", max_thickness)):
        if v is not None and v < 1:
            problems.append(f"{name} must be at least 1, got {v}")
    if min_thickness is not None and max_thickness is not None and min_thickness > max_thickness:
        problems.append(f"min_thickness ({min_thickness}) is greater than max_thickness ({max_thickness})")
    if max_pixels is not None and not isinstance(max_pixels, (int, np.integer)):
        problems.append("max_pixels must be an int")
    return problems


def prepare_card(reference_image, reference_alpha):
    """Make the opaque reference card: object on white, cropped, 16 px border.

    reference_image: IMAGE tensor (1, H, W, C). reference_alpha: MASK tensor
    (1, H, W) or (H, W) where 1 = opaque, or None to use the image's own alpha
    channel. Returns (card float32 (h, w, 3), note).
    """
    if reference_image is None:
        raise ValueError("reference_image is required.")
    img = reference_image
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4:
        raise ValueError(f"reference_image must be an IMAGE, got shape {tuple(reference_image.shape)}.")
    if img.shape[0] != 1:
        raise ValueError(f"reference_image must be a single image, got a batch of {img.shape[0]}.")
    _, h, w, c = img.shape
    if h < 1 or w < 1:
        raise ValueError(f"reference_image is empty (shape {tuple(img.shape)}).")
    arr = img[0].detach().cpu().float().numpy()

    if reference_alpha is not None:
        a = reference_alpha
        if a.ndim == 3:
            if a.shape[0] != 1:
                raise ValueError(f"reference_alpha must be a single mask, got a batch of {a.shape[0]}.")
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"reference_alpha must be a MASK, got shape {tuple(reference_alpha.shape)}.")
        if tuple(a.shape) != (h, w):
            raise ValueError(
                f"reference_alpha is {a.shape[1]}x{a.shape[0]} but reference_image is {w}x{h}; "
                "they must be the same size.")
        alpha = a.detach().cpu().float().numpy()
        source = "reference_alpha"
        if c < 3:
            raise ValueError(f"reference_image must have at least 3 channels, got {c}.")
    else:
        if c != 4:
            raise ValueError(
                f"reference_alpha is not connected, so reference_image must have an alpha channel "
                f"(4 channels), but it has {c}. Connect the mask from your background-removal step "
                "to reference_alpha (1 = object, 0 = transparent).")
        alpha = arr[..., 3]
        source = "the image's alpha channel"

    rgb = np.clip(arr[..., :3], 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    on_white = rgb * alpha[..., None] + (1.0 - alpha[..., None])

    rows = np.flatnonzero((alpha > ALPHA_CROP_THRESHOLD).any(axis=1))
    cols = np.flatnonzero((alpha > ALPHA_CROP_THRESHOLD).any(axis=0))
    if rows.size == 0:
        cropped = np.ones_like(on_white)
        note = f"reference {w}x{h}, alpha from {source}: fully transparent, so the card is blank white"
    else:
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        cropped = on_white[y0:y1, x0:x1]
        note = (f"reference {w}x{h}, alpha from {source}: object bounding box "
                f"{x1 - x0}x{y1 - y0} at ({x0}, {y0})")

    b = CARD_BORDER_SOURCE_PX
    ch, cw = cropped.shape[:2]
    card = np.ones((ch + 2 * b, cw + 2 * b, 3), dtype=np.float32)
    card[b:b + ch, b:b + cw] = cropped
    return card, note + f"; card with {b} px white border is {cw + 2 * b}x{ch + 2 * b}"


# ---------------------------------------------------------------------------
# Layout planning (pure arithmetic)
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """Frame size and zone thickness for one zone orientation."""
    vertical: bool
    width: int
    height: int
    thickness: int
    video_scale: float
    warp: float            # frame aspect / video aspect - 1
    score: float
    resized: bool


@dataclass
class Layout:
    side: str
    width: int
    height: int
    thickness: int
    zone: tuple            # (x, y, w, h)
    video: tuple           # (x, y, w, h): where the scaled video goes
    card: tuple            # (x, y, w, h): where the card goes
    rotation: str          # "none", "ccw" or "cw"
    video_scale: float
    card_scale: float
    warp: float
    resized: bool
    score: float


def video_box(vertical, width, height, thickness):
    return (width - thickness, height) if vertical else (width, height - thickness)


def video_score(vertical, width, height, thickness, src_w, src_h, target, auto=False):
    """(scale, warp, score) for video concerns only, or None if impossible."""
    bw, bh = video_box(vertical, width, height, thickness)
    if bw < 1 or bh < 1 or thickness < 1:
        return None
    scale = min(bw / src_w, bh / src_h)
    warp = (width / height) / (src_w / src_h)
    s = W_VIDEO * 2.0 * math.log(min(scale, 1.0))
    s -= W_RESAMPLE * abs(math.log(scale))
    if auto:
        s -= W_AUTO_SHORTFALL * max(0.0, -math.log(scale))
        s -= W_WARP * abs(math.log(warp))
    s -= W_CANVAS_AREA * math.log((width * height) / (src_w * src_h))
    s -= W_THICKNESS * abs(thickness - target) / GRID
    return scale, warp - 1.0, s


def fixed_frame(vertical, src_w, src_h, thickness, target, auto=False):
    r = video_score(vertical, src_w, src_h, thickness, src_w, src_h, target, auto)
    if r is None:
        return None
    return Frame(vertical, src_w, src_h, thickness, r[0], r[1], r[2], False)


def allowed_thicknesses(cfg):
    """Multiples of 32 in [min, max]; if there are none, the clamped target."""
    lo, hi = cfg.min_thickness, cfg.max_thickness
    aligned = list(range(_ceil_to(lo, GRID), hi + 1, GRID))
    return aligned or [min(max(cfg.target_thickness, lo), hi)]


def auto_frame(vertical, src_w, src_h, cfg, fixed):
    """Best frame (multiples of 32) and thickness for one orientation, or fixed.

    For each allowed thickness and each frame size across the zone (width for
    a left/right zone) that gives the video a scale near 1, the frame size
    along the zone is the multiple of 32 just below or above the one that
    makes the frame's aspect ratio equal the video's.
    """
    target = cfg.target_thickness
    cap = cfg.max_pixels if cfg.max_pixels >= 0 else None
    min_scale = fixed.video_scale if fixed is not None else 0.0
    best = fixed

    src_across, src_along = (src_w, src_h) if vertical else (src_h, src_w)
    lo_scale = max(min_scale, AUTO_MIN_SCALE)
    for t in allowed_thicknesses(cfg):
        start = _ceil_to(t + src_across * lo_scale, GRID)
        stop = int(t + src_across * AUTO_MAX_SCALE)
        for across in range(max(start, GRID), stop + 1, GRID):
            ideal = across * src_along / src_across
            base = max(GRID, _floor_to(ideal, GRID))
            for along in (base, base + GRID):
                w, h = (across, along) if vertical else (along, across)
                if cap is not None and w * h > cap:
                    continue
                r = video_score(vertical, w, h, t, src_w, src_h, target, auto=True)
                if r is None or r[0] < min_scale - 1e-9:
                    continue
                if best is None or r[2] > best.score + 1e-12:
                    best = Frame(vertical, w, h, t, r[0], r[1], r[2], (w, h) != (src_w, src_h))
    return best


def fit_inside(src_w, src_h, max_w, max_h):
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def place(frame, side, src_w, src_h, card_w, card_h, rotation):
    """Rectangles for a frame on a given side, with the card rotated or not."""
    W, H, t = frame.width, frame.height, frame.thickness
    if side == "left":
        zone, box = (0, 0, t, H), (t, 0, W - t, H)
    elif side == "right":
        zone, box = (W - t, 0, t, H), (0, 0, W - t, H)
    elif side == "top":
        zone, box = (0, 0, W, t), (0, t, W, H - t)
    else:
        zone, box = (0, H - t, W, t), (0, 0, W, H - t)

    vw, vh = fit_inside(src_w, src_h, box[2], box[3])
    vw, vh = min(vw, box[2]), min(vh, box[3])
    video = (box[0] + (box[2] - vw) // 2, box[1] + (box[3] - vh) // 2, vw, vh)

    if rotation != "none":
        card_w, card_h = card_h, card_w
    usable_w = max(1, zone[2] - 2 * ZONE_PADDING_PX)
    usable_h = max(1, zone[3] - 2 * ZONE_PADDING_PX)
    box_w = max(1, int(round(usable_w * CARD_SCALE)))
    box_h = max(1, int(round(usable_h * CARD_SCALE)))
    card_scale = min(box_w / card_w, box_h / card_h)
    cw, ch = fit_inside(card_w, card_h, box_w, box_h)
    card = (zone[0] + ZONE_PADDING_PX + max(0, (usable_w - cw) // 2),
            zone[1] + ZONE_PADDING_PX + max(0, (usable_h - ch) // 2), cw, ch)
    return zone, video, card, card_scale


def plan_layout(src_w, src_h, cfg):
    """Choose the layout. Returns (Layout, report lines)."""
    card_h, card_w = cfg.card.shape[:2]
    report = []
    lo, hi = cfg.min_thickness, cfg.max_thickness
    if lo > hi:
        raise ValueError(f"min_thickness ({lo}) is greater than max_thickness ({hi}).")
    fixed_t = min(max(cfg.target_thickness, lo), hi)
    if fixed_t != cfg.target_thickness:
        report.append(f"target thickness {cfg.target_thickness} is outside "
                      f"[{lo}, {hi}]; using {fixed_t} without auto-resize")

    frames = {}
    for vertical in (True, False):
        if not any((s in VERTICAL_SIDES) == vertical for s, _ in ZONE_LOCATIONS[cfg.zone_location]):
            continue
        fixed = fixed_frame(vertical, src_w, src_h, fixed_t, cfg.target_thickness,
                            cfg.auto_resize)
        frame = auto_frame(vertical, src_w, src_h, cfg, fixed) if cfg.auto_resize else fixed
        name = "vertical strip (left/right)" if vertical else "horizontal strip (top/bottom)"
        if frame is None:
            report.append(f"{name}: impossible (zone thickness leaves no room for the video)")
            continue
        frames[vertical] = frame
        how = "auto-resized" if frame.resized else "original canvas"
        report.append(
            f"{name}: {how} {frame.width}x{frame.height}, thickness {frame.thickness}, "
            f"video scale {frame.video_scale:.4f}, aspect error {100 * frame.warp:+.2f}%, "
            f"video score {frame.score:+.4f}")

    rotations = ["none"]
    if cfg.allow_rotation:
        rotations.append("ccw" if cfg.rotate_ccw else "cw")

    candidates = []
    for side, preferred in ZONE_LOCATIONS[cfg.zone_location]:
        frame = frames.get(side in VERTICAL_SIDES)
        if frame is None:
            continue
        for rot in rotations:
            zone, video, card, cscale = place(frame, side, src_w, src_h, card_w, card_h, rot)
            ref = W_REFERENCE * 2.0 * math.log(min(cscale, 1.0))
            score = frame.score + ref
            if preferred and len(ZONE_LOCATIONS[cfg.zone_location]) > 1:
                score += PREFERRED_BONUS
            if rot != "none":
                score -= ROTATION_PENALTY
            candidates.append(Layout(side, frame.width, frame.height, frame.thickness, zone,
                                     video, card, rot, frame.video_scale, cscale,
                                     frame.warp, frame.resized, score))
            report.append(
                f"  {side}, rotation {rot}: card {card[2]}x{card[3]} (scale {cscale:.4f}), "
                f"reference score {ref:+.4f}, total {score:+.4f}")
    if not candidates:
        raise ValueError(
            f"no layout fits: a {src_w}x{src_h} video leaves no room for a reference zone "
            f"{fixed_t} px thick on the allowed sides ({cfg.zone_location}).")
    best = max(candidates, key=lambda c: c.score)  # first wins ties (preferred side, unrotated)
    return best, report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _resize_float(plane, size, resample):
    """Resize a 2-D float32 array to size=(w, h) with PIL in float mode."""
    img = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    return np.asarray(img.resize(size, resample), dtype=np.float32)


def resize_rgb(rgb, size):
    if (rgb.shape[1], rgb.shape[0]) == tuple(size):
        return rgb
    out = np.stack([_resize_float(rgb[..., k], size, Image.LANCZOS) for k in range(3)], axis=-1)
    return np.clip(out, 0.0, 1.0)


def resize_mask(mask, size):
    if (mask.shape[1], mask.shape[0]) != tuple(size):
        mask = _resize_float(mask, size, Image.BOX)
    return (mask >= MASK_THRESHOLD).astype(np.float32)


def render(images, masks, layout, cfg, pool):
    """images: (N, H, W, 3) float32 numpy; masks: (N or 1, H, W). Returns (N, H', W', 3)."""
    n = images.shape[0]
    W, H = layout.width, layout.height
    zx, zy, zw, zh = layout.zone
    vx, vy, vw, vh = layout.video
    cx, cy, cw, chh = layout.card

    card = cfg.card
    if layout.rotation == "ccw":
        card = np.rot90(card, k=1)
    elif layout.rotation == "cw":
        card = np.rot90(card, k=-1)
    card = resize_rgb(np.ascontiguousarray(card), (cw, chh))

    base = np.zeros((H, W, 3), dtype=np.float32)
    base[zy:zy + zh, zx:zx + zw] = np.asarray(cfg.zone_color, dtype=np.float32)
    base[cy:cy + chh, cx:cx + cw] = card

    # Masks at the video's new size, placed on a canvas-sized grid.
    canvas_masks = np.zeros((masks.shape[0], H, W), dtype=np.float32)
    for i in range(masks.shape[0]):
        canvas_masks[i, vy:vy + vh, vx:vx + vw] = resize_mask(masks[i], (vw, vh))
    if masks.shape[0] == 1 and n > 1:
        canvas_masks = np.repeat(canvas_masks, n, axis=0)
    if pool:
        pooled = pool_mask_blocks(torch.from_numpy(canvas_masks), "max")
        canvas_masks = np.asarray(pooled.cpu().numpy(), dtype=np.float32)
        clip = np.zeros((H, W), dtype=np.float32)
        clip[vy:vy + vh, vx:vx + vw] = 1.0
        canvas_masks *= clip

    out = np.empty((n, H, W, 3), dtype=np.float32)
    mask_rgb = np.asarray(cfg.mask_color, dtype=np.float32)
    for i in range(n):
        out[i] = base
        out[i, vy:vy + vh, vx:vx + vw] = resize_rgb(images[i], (vw, vh))
        out[i][canvas_masks[i] > 0.5] = mask_rgb
    masked_fraction = float((canvas_masks > 0.5).mean())
    return out, masked_fraction


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

THICKNESS_KW = dict(min=GRID, max=8192, step=GRID)


class ConfigureMaskAndReferenceGuideVideos(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConfigureMaskAndReferenceGuideVideos",
            display_name="Configure Mask and Reference Guide Videos",
            category="mask/ltxv",
            description=(
                "Holds the reference image and layout settings for Build Mask and Reference "
                "Guide Video nodes, so several of them (e.g. both stages of a two-stage "
                "workflow) lay out their guides the same way."
            ),
            inputs=[
                io.Image.Input(
                    "reference_image",
                    tooltip="The object to insert, a single image. Shown on a white card in the "
                            "reference zone. Connect an empty/transparent image for text-only "
                            "inpainting (blank card).",
                ),
                io.Mask.Input(
                    "reference_alpha",
                    optional=True,
                    tooltip="Alpha for the reference (1 = object, 0 = background). If not "
                            "connected, reference_image must have an alpha channel. Note: Load "
                            "Image's MASK output is inverted alpha; pass it through Invert Mask.",
                ),
                io.Int.Input(
                    "mask_color",
                    optional=True,
                    force_input=True,
                    tooltip="0xRRGGBB (e.g. Color to RGB Int's rgb_int). Default magenta "
                            "(255, 0, 255).",
                ),
                io.Int.Input(
                    "zone_color",
                    optional=True,
                    force_input=True,
                    tooltip="0xRRGGBB. Default chroma green (0, 255, 0).",
                ),
                io.Boolean.Input(
                    "auto_resize",
                    default=False,
                    tooltip="Grow the frame (in 32 px steps, keeping the video's aspect ratio) so the "
                            "video keeps its full resolution beside the reference zone. Never "
                            "shrinks the video. The sampled output then needs resizing back; "
                            "the report gives the factors.",
                ),
                io.Combo.Input(
                    "zone_location",
                    options=ZONE_LOCATION_NAMES,
                    default="left",
                    tooltip="Side(s) the reference zone may go on. With two sides, the better "
                            "layout is used; the preferred side wins near-ties.",
                ),
                io.Int.Input("target_thickness", default=256, **THICKNESS_KW,
                             tooltip="Reference zone thickness, clamped to min/max. Without auto_resize "
                                     "it's used as is; auto_resize prefers it."),
                io.Int.Input("min_thickness", default=256, **THICKNESS_KW,
                             tooltip="Smallest thickness auto_resize may use (it only uses multiples of 32)."),
                io.Int.Input("max_thickness", default=256, **THICKNESS_KW,
                             tooltip="Largest thickness auto_resize may use (it only uses multiples of 32)."),
                io.Boolean.Input(
                    "allow_rotation",
                    default=False,
                    tooltip="Allow turning the reference a quarter turn when that shows it "
                            "substantially larger in the zone.",
                ),
                io.Boolean.Input("rotate_ccw", default=True,
                                 tooltip="Rotation direction: on = counterclockwise, off = clockwise."),
                io.Int.Input(
                    "max_pixels",
                    default=-1,
                    min=-1,
                    max=1 << 30,
                    step=1,
                    tooltip="Largest frame (width x height) auto_resize may grow to; -1 = no "
                            "limit. Never shrinks a frame; 0 effectively turns auto_resize off.",
                ),
            ],
            outputs=[
                ConfigType.Output(display_name="configuration"),
            ],
        )

    @classmethod
    def validate_inputs(cls, zone_location=None, target_thickness=None, min_thickness=None,
                        max_thickness=None, max_pixels=None, **kwargs):
        problems = check_settings(zone_location, target_thickness, min_thickness,
                                  max_thickness, max_pixels)
        return "; ".join(problems) if problems else True

    @classmethod
    def execute(cls, reference_image, auto_resize, zone_location, target_thickness,
                min_thickness, max_thickness, allow_rotation, rotate_ccw, max_pixels,
                reference_alpha=None, mask_color=None, zone_color=None) -> io.NodeOutput:
        prefix = "Configure Mask and Reference Guide Videos: "
        problems = check_settings(zone_location, target_thickness, min_thickness,
                                  max_thickness, max_pixels)
        try:
            if problems:
                raise ValueError("; ".join(problems))
            mc = unpack_rgb_int(mask_color, DEFAULT_MASK_COLOR, "mask_color")
            zc = unpack_rgb_int(zone_color, DEFAULT_ZONE_COLOR, "zone_color")
            if mc == zc:
                raise ValueError(f"mask_color and zone_color are the same color {describe_color(mc)}.")
            card, note = prepare_card(reference_image, reference_alpha)
        except ValueError as e:
            raise ValueError(prefix + str(e)) from None
        cfg = GuideConfig(
            card=card, card_note=note, mask_color=mc, zone_color=zc,
            auto_resize=bool(auto_resize), zone_location=zone_location,
            target_thickness=int(target_thickness), min_thickness=int(min_thickness),
            max_thickness=int(max_thickness), allow_rotation=bool(allow_rotation),
            rotate_ccw=bool(rotate_ccw), max_pixels=int(max_pixels),
        )
        return io.NodeOutput(cfg)


class BuildMaskAndReferenceGuideVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BuildMaskAndReferenceGuideVideo",
            display_name="Build Mask and Reference Guide Video",
            category="mask/ltxv",
            description=(
                "Lays out a guide video for use with a reference-based inpainting LoRA: a colored "
                "reference zone with the reference on a white card, beside the video scaled to "
                "fit, with the masked area painted in the mask color."
            ),
            inputs=[
                io.Image.Input("video_images", tooltip="The video frames."),
                io.Mask.Input("video_masks",
                              tooltip="One mask per frame (or a single mask for all frames), "
                                      "same size as the frames; 1 = area to regenerate."),
                ConfigType.Input("configuration",
                                 tooltip="From Configure Mask and Reference Guide Videos."),
                io.Boolean.Input(
                    "pool_mask_blocks",
                    default=True,
                    tooltip="After layout, pool the mask to LTX latent blocks (32x32 px on the "
                            "final frame's grid, frame 0 then groups of 8 frames) with max, "
                            "exactly as LTXV Pool Mask Blocks does, clipped to the video area.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="guide_images"),
                io.Boolean.Output(display_name="resized",
                                  tooltip="True if auto_resize changed the frame size."),
                io.String.Output(display_name="report",
                                 tooltip="What the node decided and why."),
            ],
        )

    @classmethod
    def execute(cls, video_images, video_masks, configuration, pool_mask_blocks) -> io.NodeOutput:
        try:
            return cls._run(video_images, video_masks, configuration, pool_mask_blocks)
        except ValueError as e:
            raise ValueError(f"Build Mask and Reference Guide Video: {e}") from None

    @classmethod
    def _run(cls, video_images, video_masks, cfg, pool):
        if not isinstance(cfg, GuideConfig):
            raise ValueError("configuration must come from Configure Mask and Reference Guide Videos.")
        if video_images is None or video_images.ndim != 4:
            raise ValueError(f"video_images must be an IMAGE batch, got shape "
                             f"{None if video_images is None else tuple(video_images.shape)}.")
        n, h, w, c = video_images.shape
        if n < 1 or h < 1 or w < 1 or c < 3:
            raise ValueError(f"video_images has an unusable shape {tuple(video_images.shape)}.")
        masks = video_masks
        if masks is None:
            raise ValueError("video_masks is required.")
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError(f"video_masks must be a MASK batch, got shape {tuple(video_masks.shape)}.")
        if tuple(masks.shape[1:]) != (h, w):
            raise ValueError(f"video_masks are {masks.shape[2]}x{masks.shape[1]} but video_images "
                             f"are {w}x{h}; they must be the same size.")
        if masks.shape[0] not in (1, n):
            raise ValueError(f"{masks.shape[0]} masks for {n} frames; give one per frame or a "
                             "single mask for all.")

        layout, plan_lines = plan_layout(w, h, cfg)

        report = [f"Input: {n} frames, {w}x{h}."]
        if masks.shape[0] == 1 and n > 1:
            report.append("One mask given; used for every frame.")
        report.append(f"Reference: {cfg.card_note}.")
        report.append(
            f"Configuration: zone {cfg.zone_location}, thickness target {cfg.target_thickness} "
            f"[{cfg.min_thickness}, {cfg.max_thickness}], auto_resize "
            f"{'on' if cfg.auto_resize else 'off'}"
            + (f" (max_pixels {cfg.max_pixels if cfg.max_pixels >= 0 else 'unlimited'})"
               if cfg.auto_resize else "")
            + f", rotation {'allowed (' + ('ccw' if cfg.rotate_ccw else 'cw') + ')' if cfg.allow_rotation else 'off'}"
            + f", mask color {describe_color(cfg.mask_color)}, zone color {describe_color(cfg.zone_color)}.")
        report.append("Candidates considered (higher score is better):")
        report += ["  " + line for line in plan_lines]

        L = layout
        report.append(
            f"Chosen: zone on the {L.side}, {L.thickness} px thick; frame {L.width}x{L.height}"
            + (f" (auto-resized from {w}x{h})" if L.resized else " (original size)") + ".")
        report.append(
            f"Video: {L.video[2]}x{L.video[3]} at ({L.video[0]}, {L.video[1]}), scale "
            f"{L.video_scale:.4f} ({100 * min(L.video_scale, 1) ** 2:.1f}% of source pixels"
            + (", copied 1:1" if (L.video[2], L.video[3]) == (w, h) else "") + ").")
        report.append(
            f"Card: {L.card[2]}x{L.card[3]} at ({L.card[0]}, {L.card[1]}), scale "
            f"{L.card_scale:.4f}, rotation {L.rotation}.")
        if L.resized:
            report.append(
                f"Output: the sampled video will be {L.width}x{L.height} with the video filling "
                f"the whole frame; resize it to {w}x{h} (x {w / L.width:.4f}, "
                f"y {h / L.height:.4f}) to match the input"
                + (f". Its aspect ratio is off by {100 * L.warp:+.2f}%, which that resize undoes."
                   if abs(L.warp) > 1e-9 else "; aspect ratio exact."))

        warnings = []
        if L.thickness % GRID:
            warnings.append(f"zone thickness {L.thickness} is not a multiple of 32, so the zone's "
                            "edge isn't on a latent-cell boundary as it was in training")
        if L.width % GRID or L.height % GRID:
            warnings.append(f"frame size {L.width}x{L.height} is not a multiple of 32; LTX will "
                            "resize the guide, which moves the layout off the latent grid "
                            "(consider auto_resize)")
        if (n - 1) % 8:
            warnings.append(f"{n} frames is not of the form 8n+1")

        images_np = video_images[..., :3].detach().cpu().float().numpy().clip(0.0, 1.0)
        masks_np = masks.detach().cpu().float().numpy().clip(0.0, 1.0)
        out, masked = render(images_np, masks_np, layout, cfg, bool(pool))
        report.append(
            f"Mask: area-resampled and thresholded at {MASK_THRESHOLD}"
            + (", pooled to 32x32 px / 8-frame latent blocks (max) and clipped to the video area"
               if pool else ", not pooled")
            + f"; {100 * masked:.1f}% of all frame pixels painted.")
        for wmsg in warnings:
            report.append("Warning: " + wmsg + ".")

        return io.NodeOutput(torch.from_numpy(out), bool(L.resized), "\n".join(report))
