"""
LTXV Add Frame Guides
=====================

Adds LTXV keyframe guides (the same kind LTXVAddGuide adds: appended guide
tokens, recorded in the conditioning, removed later by LTXVCropGuides) for a
chosen set of frames from a full-length image batch.

Install: put this folder in ComfyUI/custom_nodes/ and restart ComfyUI.
Requires a 2026 ComfyUI build whose comfy_extras/nodes_lt.py provides
LTXVAddGuide (encode, get_latent_index, append_keyframe), get_keyframe_idxs
and get_noise_mask. If core's private _append_guide_attention_entry is ever
renamed, the node falls back to its own equivalent. Missing required helpers
produce a clear error when ComfyUI loads the node.

The node never edits the main video part of the latent (its frames or its
noise mask); it only appends guides.

Pipeline:
  1. Decide which frames become guides (a per-frame boolean array):
       - frames whose mask is completely black (if guides_from_empty_masks)
       - frames listed in frame_list (if guides_from_frame_list)
     A frame selected by both sources is still selected only once.
  2. Plan guides from that array:
       - frame 0 is always a single-frame guide (VAE layout: frame 0 is its
         own latent frame)
       - with coalesce_latent_frames (default): every complete, aligned
         8-frame latent group (frames 8k-7 .. 8k) whose frames are all
         selected becomes part of a multi-frame guide, and consecutive
         complete groups are merged into one guide; every other selected
         frame becomes a single-frame guide
       - without it: every selected frame becomes a single-frame guide
     No frames are added or removed from the selection by planning.
  3. Encode and attach each guide, with optional per-frame attention masks.

Two separate guide controls (core LTXVAddGuide uses one "strength" for both):
  integrity           0..1   The guide tokens' noise mask is 1 - integrity.
                             1 = the reference stays exactly as encoded for the
                             whole run; lower = it starts noisier and is partly
                             re-interpreted by the model.
  attention_strength  0..10  Multiplies how much the generated video attends to
                             the guide (log-bias in self-attention). 1 = normal,
                             0 = ignored, >1 = amplified. Attention masks are
                             multiplied by this.

Options (DICT, or the same thing as JSON text in the widget):
  guides_from_empty_masks  bool  required
  guides_from_frame_list   bool  required
  use_attention_masks      bool  optional, default false
  invert_attention_masks   bool  optional, default false
  coalesce_latent_frames   bool  optional, default true
"""

import json
import logging
import re

import torch
from comfy_api.latest import ComfyExtension, io

LOG = logging.getLogger("LTXVAddFrameGuides")

# --------------------------------------------------------------------------
# Core LTX helpers this node relies on. These are ComfyUI internals, so check
# for them up front and fail with a clear message if a core update renames them.
# --------------------------------------------------------------------------

_INCOMPATIBLE = (
    "LTXV Add Frame Guides: this ComfyUI version's LTX code (comfy_extras/nodes_lt.py) "
    "is missing {what}. The node was written against 2026 ComfyUI builds; core's LTX "
    "guide internals have probably changed. Update or adapt this node."
)

try:
    from comfy_extras import nodes_lt as _nodes_lt
except ImportError as e:
    raise ImportError(_INCOMPATIBLE.format(what="the module itself") + f" ({e})") from e

_missing = [
    name for name in ("LTXVAddGuide", "get_keyframe_idxs", "get_noise_mask")
    if not hasattr(_nodes_lt, name)
]
if not _missing:
    _missing = [
        f"LTXVAddGuide.{name}" for name in ("encode", "get_latent_index", "append_keyframe")
        if not hasattr(_nodes_lt.LTXVAddGuide, name)
    ]
if _missing:
    _msg = _INCOMPATIBLE.format(what=", ".join(_missing))
    LOG.error(_msg)
    raise ImportError(_msg)

LTXVAddGuide = _nodes_lt.LTXVAddGuide
get_keyframe_idxs = _nodes_lt.get_keyframe_idxs
get_noise_mask = _nodes_lt.get_noise_mask

_GUIDE_ATTENTION_KEY = "guide_attention_entries"


def _fallback_append_guide_attention_entry(positive, negative, pre_filter_count, latent_shape,
                                           strength=1.0, attention_mask=None):
    """Stand-in for core's private _append_guide_attention_entry, writing the same
    entry format to the same conditioning key (the format LTX's own nodes also use)."""
    import node_helpers

    new_entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask is not None else None,
        "latent_shape": latent_shape,
    }
    results = []
    for cond in (positive, negative):
        existing = []
        for t in cond:
            found = t[1].get(_GUIDE_ATTENTION_KEY, None)
            if found is not None:
                existing = found
                break
        results.append(node_helpers.conditioning_set_values(
            cond, {_GUIDE_ATTENTION_KEY: [*existing, new_entry]}
        ))
    return results[0], results[1]


if hasattr(_nodes_lt, "_append_guide_attention_entry"):
    _append_guide_attention_entry = _nodes_lt._append_guide_attention_entry
else:
    LOG.warning(
        "LTXV Add Frame Guides: core's _append_guide_attention_entry not found; "
        "using the node's built-in equivalent."
    )
    _append_guide_attention_entry = _fallback_append_guide_attention_entry

# A mask frame counts as "completely black" if its maximum value is at or below this.
EMPTY_MASK_EPS = 1e-6

OPTION_SPEC = {
    # key: (required, default)
    "guides_from_empty_masks": (True, None),
    "guides_from_frame_list": (True, None),
    "use_attention_masks": (False, False),
    "invert_attention_masks": (False, False),
    "coalesce_latent_frames": (False, True),
}

DEFAULT_OPTIONS_TEXT = (
    '{\n'
    '  "guides_from_empty_masks": true,\n'
    '  "guides_from_frame_list": true,\n'
    '  "use_attention_masks": false,\n'
    '  "invert_attention_masks": false,\n'
    '  "coalesce_latent_frames": true\n'
    '}'
)


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------

_SINGLE_RE = re.compile(r"[0-9]+")
_RANGE_RE = re.compile(r"([0-9]+)\s*-\s*([0-9]+)")
_NEGATIVE_RE = re.compile(r"-\s*[0-9]+(\s*-\s*-?\s*[0-9]+)?")


def parse_frame_list(text):
    """Parse e.g. "40, 57-64, 120" into a sorted list of unique frame indices.

    Commas are the only separators. Whitespace (any kind) is allowed around
    numbers and dashes. Empty / all-whitespace input means no frames.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        raise ValueError(f"frame_list must be text, got {type(text).__name__}.")
    if text.strip() == "":
        return []

    frames = set()
    for pos, raw in enumerate(text.split(","), start=1):
        item = raw.strip()
        if item == "":
            raise ValueError(
                f"frame_list: item {pos} is empty (check for doubled or trailing commas). "
                f"Full text: {text!r}"
            )
        if _SINGLE_RE.fullmatch(item):
            frames.add(int(item))
            continue
        m = _RANGE_RE.fullmatch(item)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                raise ValueError(f"frame_list: range {item!r} runs backwards ({a} > {b}).")
            frames.update(range(a, b + 1))
            continue
        if _NEGATIVE_RE.fullmatch(item):
            raise ValueError(f"frame_list: negative frame index in {item!r} is not allowed.")
        raise ValueError(
            f"frame_list: cannot understand item {pos}: {item!r}. "
            f"Expected a whole number like 40 or a range like 57-64, separated by commas."
        )
    return sorted(frames)


def parse_options(options):
    """Accept a dict or JSON text describing a dict; validate strictly."""
    if isinstance(options, str):
        if options.strip() == "":
            raise ValueError("options: text is empty; it must be a JSON object.")
        try:
            data = json.loads(options)
        except json.JSONDecodeError as e:
            raise ValueError(f"options: invalid JSON ({e}).") from e
    elif isinstance(options, dict):
        data = options
    else:
        raise ValueError(
            f"options must be a DICT or JSON text, got {type(options).__name__}."
        )
    if not isinstance(data, dict):
        raise ValueError(f"options: must be a JSON object / DICT, got {type(data).__name__}.")

    unknown = sorted(k for k in data if k not in OPTION_SPEC)
    if unknown:
        raise ValueError(
            f"options: unknown key(s) {unknown}. Allowed keys: {sorted(OPTION_SPEC)}."
        )

    result = {}
    for key, (required, default) in OPTION_SPEC.items():
        if key not in data:
            if required:
                raise ValueError(f"options: required key {key!r} is missing.")
            result[key] = default
            continue
        value = data[key]
        if not isinstance(value, bool):
            raise ValueError(
                f"options: {key!r} must be a boolean (true/false), got {value!r} "
                f"({type(value).__name__})."
            )
        result[key] = value
    return result


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def plan_guides(selected, num_frames, time_scale, coalesce):
    """Turn a per-frame boolean list into a list of guides.

    Each guide is (start_frame, num_frames). Multi-frame guides always start
    at a frame of the form time_scale*k + 1 and span whole latent groups.
    """
    if len(selected) != num_frames:
        raise ValueError("internal: selection length mismatch")

    guides = []
    used = [False] * num_frames

    if coalesce:
        num_groups = (num_frames - 1) // time_scale  # complete groups after frame 0
        full = []
        for k in range(1, num_groups + 1):
            lo = time_scale * (k - 1) + 1
            hi = time_scale * k  # inclusive
            full.append(all(selected[lo:hi + 1]))
        k = 0
        while k < num_groups:
            if full[k]:
                start_k = k
                while k < num_groups and full[k]:
                    k += 1
                lo = time_scale * start_k + 1
                hi = time_scale * k  # inclusive end of last group in run
                guides.append((lo, hi - lo + 1))
                for f in range(lo, hi + 1):
                    used[f] = True
            else:
                k += 1

    for f in range(num_frames):
        if selected[f] and not used[f]:
            guides.append((f, 1))

    guides.sort(key=lambda g: g[0])
    return guides


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class LTXVAddFrameGuides(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVAddFrameGuides",
            display_name="LTXV Add Frame Guides",
            category="conditioning/ltxv",
            description=(
                "Adds LTXV keyframe guides for selected frames of a full-length image batch: "
                "frames with completely black masks and/or frames listed in frame_list. "
                "Complete, aligned 8-frame groups are combined into multi-frame guides (optional). "
                "Never edits the main video latent or its noise mask. Remove the guides after "
                "sampling with LTXV Crop Guides."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input(
                    "latent",
                    tooltip="Video-only LTX latent (not a combined audio+video latent). "
                            "Add guides before combining with audio.",
                ),
                io.Vae.Input("vae", tooltip="The LTX video VAE."),
                io.Image.Input(
                    "images",
                    tooltip="Full-length frame batch: one image per video frame, exactly the "
                            "latent's pixel size. Guides are taken from these frames.",
                ),
                io.Mask.Input(
                    "masks",
                    tooltip="One mask per frame, same size as the images. Completely black "
                            "frames can be selected as guides; masks can also serve as "
                            "attention masks.",
                ),
                io.Float.Input(
                    "integrity", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How intact each guide's reference stays (its noise mask = 1 - integrity). "
                            "1 = stays exactly as encoded; lower = starts noisier and can be "
                            "partly re-interpreted. Applies to every guide this node adds.",
                ),
                io.Float.Input(
                    "attention_strength", default=1.0, min=0.0, max=10.0, step=0.01,
                    tooltip="How much the generated video attends to each guide. 1 = normal, "
                            "0 = ignored, above 1 = amplified (outside training range; may cause "
                            "artifacts). Attention masks are multiplied by this.",
                ),
                io.String.Input(
                    "frame_list", default="", multiline=False,
                    tooltip="Frames to make guides, e.g. \"40, 57-64, 120\". Commas only as "
                            "separators. Empty = none. Used only if guides_from_frame_list "
                            "is true.",
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "options", default=DEFAULT_OPTIONS_TEXT, multiline=True,
                        tooltip="JSON object, or connect a DICT. Keys: guides_from_empty_masks "
                                "(required), guides_from_frame_list (required), "
                                "use_attention_masks, invert_attention_masks, "
                                "coalesce_latent_frames.",
                    ),
                    [io.Dict],
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="plan", tooltip="Human-readable list of the guides added."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, images, masks, integrity,
                attention_strength, frame_list, options) -> io.NodeOutput:
        opts = parse_options(options)
        # When guides_from_frame_list is false the frame list is ignored entirely (not even parsed).
        listed = parse_frame_list(frame_list) if opts["guides_from_frame_list"] else []

        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            raise ValueError(
                "latent is a combined audio+video latent. Add guides to the video latent "
                "before LTXV Concat AV Latent."
            )
        if samples.ndim != 5:
            raise ValueError(f"latent must be a 5D video latent, got shape {tuple(samples.shape)}.")

        scale_factors = vae.downscale_index_formula
        time_scale, h_scale, w_scale = scale_factors
        _, _, latent_length, latent_h, latent_w = samples.shape

        _, existing_guides = get_keyframe_idxs(positive, samples.shape)
        main_latent_frames = latent_length - existing_guides
        expected_frames = (main_latent_frames - 1) * time_scale + 1
        expected_h, expected_w = latent_h * h_scale, latent_w * w_scale

        # ---- validate images / masks ----
        if images.ndim != 4:
            raise ValueError(f"images must be an image batch, got shape {tuple(images.shape)}.")
        n, ih, iw, _ = images.shape
        if n != expected_frames:
            raise ValueError(
                f"images has {n} frames, but the latent's video part has {main_latent_frames} "
                f"latent frames, which means {expected_frames} video frames."
            )
        if (ih, iw) != (expected_h, expected_w):
            raise ValueError(
                f"images are {iw}x{ih}, but the latent is {expected_w}x{expected_h} pixels."
            )
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError(f"masks must be a mask batch, got shape {tuple(masks.shape)}.")
        if masks.shape[0] != n:
            raise ValueError(f"masks has {masks.shape[0]} frames but images has {n}.")
        if tuple(masks.shape[1:]) != (ih, iw):
            raise ValueError(
                f"masks are {masks.shape[2]}x{masks.shape[1]}, but images are {iw}x{ih}."
            )

        bad = [f for f in listed if f >= n]
        if bad:
            raise ValueError(
                f"frame_list: frame(s) {bad[:10]}{'...' if len(bad) > 10 else ''} are beyond "
                f"the last frame ({n - 1})."
            )

        # ---- step 1: per-frame selection ----
        masks_cpu = masks.detach().float().cpu()
        empty = (masks_cpu.reshape(n, -1).amax(dim=1) <= EMPTY_MASK_EPS).tolist()
        selected = [False] * n
        if opts["guides_from_empty_masks"]:
            for f in range(n):
                if empty[f]:
                    selected[f] = True
        if opts["guides_from_frame_list"]:
            for f in listed:
                selected[f] = True

        # ---- step 2: plan ----
        guides = plan_guides(selected, n, time_scale, opts["coalesce_latent_frames"])

        if not guides:
            LOG.info("LTXV Add Frame Guides: no guides")
            return io.NodeOutput(positive, negative, latent, "no guides")

        latent_image = samples
        noise_mask = get_noise_mask(latent)
        plan_lines = []

        # ---- step 3: encode and attach ----
        for start, count in guides:
            frames = images[start:start + count]
            multi = count > 1
            causal_fix = not multi  # single frames (incl. frame 0) encode as a lone first frame

            attention_mask = None
            if opts["use_attention_masks"]:
                per_frame = []
                for f in range(start, start + count):
                    if empty[f]:
                        per_frame.append(torch.ones_like(masks_cpu[f]))
                    else:
                        m = masks_cpu[f].clamp(0.0, 1.0)
                        per_frame.append(1.0 - m if opts["invert_attention_masks"] else m)
                per_frame = torch.stack(per_frame)  # (count, H, W)
                if multi:
                    # one mask per latent frame (mean over each 8-frame group), so core's
                    # downsampler maps them 1:1 onto the guide's latent frames
                    per_frame = per_frame.reshape(count // time_scale, time_scale, ih, iw).mean(dim=1)
                if not torch.all(per_frame == 1.0):
                    attention_mask = per_frame

            if multi:
                # Prepend a duplicate so the VAE's lone-first-frame slot falls on the
                # throwaway copy; count is a multiple of time_scale, so count + 1 is
                # already 8n+1 and nothing gets trimmed.
                to_encode = torch.cat([frames[:1], frames], dim=0)
            else:
                to_encode = frames

            _, t = LTXVAddGuide.encode(vae, latent_w, latent_h, to_encode, scale_factors)
            if multi:
                t = t[:, :, 1:, :, :]
                expected_latents = count // time_scale
                if t.shape[2] != expected_latents:
                    raise RuntimeError(
                        f"internal: guide at frame {start} encoded to {t.shape[2]} latent frames, "
                        f"expected {expected_latents}."
                    )

            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_image.shape[2], count, start, scale_factors,
                latent_shape=latent_image.shape,
            )
            if frame_idx != start:
                raise RuntimeError(
                    f"internal: guide start {start} was moved to {frame_idx} by core indexing."
                )
            if latent_idx + t.shape[2] > main_latent_frames:
                raise RuntimeError(f"internal: guide at frame {start} runs past the end of the video.")

            # Same two steps core's attach_guide_latent performs (without IC-LoRA dilation),
            # but with integrity and attention strength kept separate.
            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx, latent_image, noise_mask, t, integrity,
                scale_factors, latent_downscale_factor=1, causal_fix=causal_fix,
            )
            pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
            positive, negative = _append_guide_attention_entry(
                positive, negative, pre_filter_count, list(t.shape[2:]),
                strength=attention_strength, attention_mask=attention_mask,
            )

            desc = f"frame {start}" if count == 1 else f"frames {start}-{start + count - 1} ({t.shape[2]} latent frames)"
            if attention_mask is not None:
                desc += ", attention mask"
            plan_lines.append(desc)

        plan = (f"{len(guides)} guide(s), integrity {integrity}, attention strength "
                f"{attention_strength}:\n" + "\n".join(plan_lines))
        LOG.info("LTXV Add Frame Guides: %s", plan.replace("\n", "; "))

        out_latent = dict(latent)
        out_latent["samples"] = latent_image
        out_latent["noise_mask"] = noise_mask
        return io.NodeOutput(positive, negative, out_latent, plan)


class _Extension(ComfyExtension):
    async def get_node_list(self):
        return [LTXVAddFrameGuides]


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
