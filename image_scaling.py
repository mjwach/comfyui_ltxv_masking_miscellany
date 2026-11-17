"""
Image and mask scaling shared by this pack's nodes, plus three nodes that use
it: LTXV Halve Dimensions, Match Image Size and Match Image or Mask Size.

Filters are chosen per axis from that axis's scale factor, and applied one
axis at a time (the filters are separable, so this is the same as a 2-D
resize, but lets the two axes use different filters):

  images, shrinking by a whole number (2x, 3x, ...): box (block average).
      Exact for integer factors, no ringing.
  images, any other change (up, or down by a non-integer factor): Lanczos.
      Sharpest; scaled correctly for any factor. Results are clipped to 0..1.
  masks, shrinking: box (area-weighted), so each output value is the fraction
      of its area that was covered. Never rings or overshoots.
  masks, growing: bilinear. Smooth, never rings or overshoots.

Everything is done in 32-bit float (no 8-bit rounding). A batch that shrinks
by whole numbers on both axes (the common exact halving) takes a fast path
that averages blocks directly; it gives the same result as the general path.

Masks can optionally be made binary afterwards by thresholding: >= 0.5
becomes 1, the rest 0.
"""

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

MASK_THRESHOLD = 0.5


def _image_filter(src, dst):
    if dst < src and src % dst == 0:
        return Image.BOX
    return Image.LANCZOS


def _mask_filter(src, dst):
    return Image.BOX if dst < src else Image.BILINEAR


def _block_factors(h, w, size):
    """(ky, kx) if (h, w) -> size shrinks by whole numbers on both axes."""
    nw, nh = size
    if h % nh or w % nw:
        return None
    return h // nh, w // nw


def _block_mean(arr, ky, kx):
    """Average ky x kx blocks over axes 1 and 2 of (N, H, W, ...) arrays."""
    out = None
    for dy in range(ky):
        for dx in range(kx):
            part = arr[:, dy::ky, dx::kx]
            if out is None:
                out = part.astype(np.float32, copy=True)
            else:
                out += part
    out *= np.float32(1.0 / (ky * kx))
    return out


def _resize_plane(plane, size, choose):
    """Resize one 2-D float32 plane to size=(w, h), one axis at a time."""
    h, w = plane.shape
    nw, nh = size
    img = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32), mode="F")
    if nw != w:
        img = img.resize((nw, h), choose(w, nw))
    if nh != h:
        img = img.resize((nw, nh), choose(h, nh))
    return np.asarray(img, dtype=np.float32)


def scale_image_array(arr, size):
    """Scale (H, W, C) or (N, H, W, C) float32 images to size=(w, h)."""
    batched = arr.ndim == 4
    a = arr if batched else arr[None]
    n, h, w, c = a.shape
    if (w, h) == tuple(size):
        return arr
    blocks = _block_factors(h, w, size)
    if blocks is not None:
        out = _block_mean(a.astype(np.float32, copy=False), *blocks)
    else:
        out = np.empty((n, size[1], size[0], c), dtype=np.float32)
        for i in range(n):
            for k in range(c):
                out[i, ..., k] = _resize_plane(a[i, ..., k], size, _image_filter)
        np.clip(out, 0.0, 1.0, out=out)
    return out if batched else out[0]


def scale_mask_array(arr, size, threshold):
    """Scale (H, W) or (N, H, W) float32 masks to size=(w, h), then threshold."""
    batched = arr.ndim == 3
    a = arr if batched else arr[None]
    n, h, w = a.shape
    if (w, h) != tuple(size):
        blocks = _block_factors(h, w, size)
        if blocks is not None:
            a = _block_mean(a.astype(np.float32, copy=False), *blocks)
        else:
            out = np.empty((n, size[1], size[0]), dtype=np.float32)
            for i in range(n):
                out[i] = _resize_plane(a[i], size, _mask_filter)
            a = out
    if threshold:
        a = (a >= MASK_THRESHOLD).astype(np.float32)
    return a if batched else a[0]


def _to_numpy(t):
    return t.detach().cpu().float().numpy()


def scale_images(images, size):
    """IMAGE tensor (N, H, W, C) -> size=(w, h). Returns the input if unchanged."""
    if (images.shape[2], images.shape[1]) == tuple(size):
        return images
    return torch.from_numpy(scale_image_array(_to_numpy(images), size))


def scale_masks(masks, size, threshold=False):
    """MASK tensor (N, H, W) -> size=(w, h). Returns the input if nothing to do."""
    if (masks.shape[2], masks.shape[1]) == tuple(size) and not threshold:
        return masks
    return torch.from_numpy(scale_mask_array(_to_numpy(masks), size, threshold))


def halved(w, h):
    """Halved dimensions, rounding up."""
    return (w + 1) // 2, (h + 1) // 2


class LTXVHalveDimensions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVHalveDimensions",
            display_name="LTXV Halve Dimensions",
            category="mask/ltxv",
            description=(
                "Halves the width and height of images and masks (odd sizes round up). "
                "Images are averaged in 2x2 blocks (Lanczos for odd sizes); masks are "
                "area-averaged and optionally thresholded. The passthrough outputs give "
                "back the inputs unchanged."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks", tooltip="Same size and frame count as images."),
                io.Boolean.Input(
                    "active",
                    default=True,
                    optional=True,
                    force_input=True,
                    tooltip="If false, the images and masks outputs pass the inputs through "
                            "unchanged. Default true when not connected.",
                ),
                io.Boolean.Input(
                    "force_binary_masks",
                    default=True,
                    tooltip="Threshold the halved masks at 0.5: values >= 0.5 become 1, the "
                            "rest 0.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Mask.Output(display_name="masks"),
                io.Image.Output(display_name="images passthrough"),
                io.Mask.Output(display_name="masks passthrough"),
            ],
        )

    @classmethod
    def execute(cls, images, masks, force_binary_masks, active=True) -> io.NodeOutput:
        if images is None or images.ndim != 4:
            raise ValueError("LTXV Halve Dimensions: images must be an IMAGE batch.")
        if masks is None:
            raise ValueError("LTXV Halve Dimensions: masks is required.")
        m = masks.unsqueeze(0) if masks.ndim == 2 else masks
        if m.ndim != 3:
            raise ValueError(f"LTXV Halve Dimensions: masks must be a MASK batch, got shape "
                             f"{tuple(masks.shape)}.")
        n, h, w = images.shape[:3]
        if tuple(m.shape) != (n, h, w):
            raise ValueError(
                f"LTXV Halve Dimensions: images are {n} frames of {w}x{h} but masks are "
                f"{m.shape[0]} frames of {m.shape[2]}x{m.shape[1]}; they must match.")
        if active is None or active:
            size = halved(w, h)
            out_images = scale_images(images, size)
            out_masks = scale_masks(m, size, bool(force_binary_masks))
        else:
            out_images, out_masks = images, masks
        return io.NodeOutput(out_images, out_masks, images, masks)


class LTXVMatchImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVMatchImageSize",
            display_name="Match Image Size",
            category="mask/ltxv",
            description=(
                "Scales every frame of images to the width and height of reference (whole "
                "frame to whole frame: no cropping or padding; the aspect ratio follows "
                "reference). Box for whole-number reductions, Lanczos otherwise. Images that "
                "already match are passed through unchanged."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Image.Input("reference",
                               tooltip="Only its width and height are used; its frame count "
                                       "doesn't matter."),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, reference) -> io.NodeOutput:
        for name, t in (("images", images), ("reference", reference)):
            if t is None or t.ndim != 4 or t.shape[1] < 1 or t.shape[2] < 1:
                raise ValueError(f"Match Image Size: {name} must be a non-empty IMAGE batch.")
        size = (int(reference.shape[2]), int(reference.shape[1]))
        return io.NodeOutput(scale_images(images, size))


def reference_size(t):
    """(w, h) of an IMAGE (N, H, W, C) or MASK (N, H, W) / (H, W) tensor, or None."""
    if t is None or t.ndim not in (2, 3, 4):
        return None
    h, w = (t.shape[0], t.shape[1]) if t.ndim == 2 else (t.shape[1], t.shape[2])
    if h < 1 or w < 1:
        return None
    return int(w), int(h)


class LTXVMatchImageOrMaskSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVMatchImageOrMaskSize",
            display_name="Match Image or Mask Size",
            category="mask/ltxv",
            description=(
                "Scales every frame of images and of masks to the width and height of "
                "reference (an image or a mask), whole frame to whole frame: no cropping or "
                "padding. Either images or masks may be left unconnected; that output is then "
                "empty. Inputs that already match are passed through unchanged."
            ),
            inputs=[
                io.Image.Input("images", optional=True),
                io.Mask.Input("masks", optional=True,
                              tooltip="Needn't be the same size or frame count as images."),
                # Declared optional only so it's shown after images and masks (ComfyUI
                # shows required inputs first); execute() requires it.
                io.MultiType.Input("reference", types=[io.Image, io.Mask], optional=True,
                                   tooltip="Required. An image or a mask. Only its width and "
                                           "height are used; its frame count doesn't matter."),
                io.Boolean.Input(
                    "force_binary_masks",
                    default=False,
                    tooltip="Threshold the scaled masks at 0.5: values >= 0.5 become 1, the "
                            "rest 0. Applied even when the masks are already the right size.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Mask.Output(display_name="masks"),
            ],
        )

    @classmethod
    def execute(cls, force_binary_masks, images=None, masks=None, reference=None) -> io.NodeOutput:
        name = "Match Image or Mask Size"
        if reference is None:
            raise ValueError(f"{name}: reference is required (an image or a mask).")
        size = reference_size(reference)
        if size is None:
            raise ValueError(f"{name}: reference must be a non-empty IMAGE or MASK.")
        if images is None and masks is None:
            raise ValueError(f"{name}: connect images, masks, or both.")
        out_images = out_masks = None
        if images is not None:
            if images.ndim != 4 or images.shape[1] < 1 or images.shape[2] < 1:
                raise ValueError(f"{name}: images must be a non-empty IMAGE batch, got shape "
                                 f"{tuple(images.shape)}.")
            out_images = scale_images(images, size)
        if masks is not None:
            m = masks.unsqueeze(0) if masks.ndim == 2 else masks
            if m.ndim != 3 or m.shape[1] < 1 or m.shape[2] < 1:
                raise ValueError(f"{name}: masks must be a non-empty MASK batch, got shape "
                                 f"{tuple(masks.shape)}.")
            out_masks = scale_masks(m, size, bool(force_binary_masks))
            if out_masks is m:
                out_masks = masks
        return io.NodeOutput(out_images, out_masks)
