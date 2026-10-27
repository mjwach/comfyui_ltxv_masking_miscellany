"""
LTXV Pool Mask Blocks
=====================

Pools a mask frame batch within LTX's latent blocks, in space and in time, and
returns a batch with the same dimensions as the input.

LTX's video VAE turns each 32x32-pixel area of each group of video frames into
one latent cell. The frame groups are: frame 0 on its own, then frames 1-8,
9-16, 17-24, and so on. A latent noise mask holds one value per cell, so a
pixel-level mask has to be reduced to that grid somehow; this node does the
reduction explicitly, with a chosen operation, and writes the pooled value back
to every pixel of every frame in the block.

Outputs:
  masks         same shape as the input, every pixel of each block set to the
                block's pooled value (for pixel-space uses: blending, guide
                preprocessing, attention masks)
  latent_masks  one mask per latent frame at latent resolution (width/32 x
                height/32): the block values themselves (mask i -> latent
                frame i). Apply with LTXV Set Video Latent Noise Masks on a
                latent that has no noise mask yet, before adding any guides
                (see README.md for what does and doesn't combine).

Operations:
  max      a block is as masked as its most-masked pixel (default; use this
           when anything touched by the mask must be fully regenerated)
  average  a block gets the mean of its pixels
  min      a block is only as masked as its least-masked pixel

require_ltx_dimensions (default true): the width and height must be multiples
of 32 and the frame count must be of the form 8n+1, or the node stops with an
error. When false, other sizes are accepted: blocks are aligned to the top-left
corner and to frame 0, the last row/column of blocks and the last frame group
may be smaller, and partial blocks are pooled over the pixels they actually
contain (padding never affects the result).
"""

import torch
from comfy_api.latest import io

# LTX video VAE compression factors: 8 frames per latent frame (after frame 0),
# 32x32 pixels per latent cell.
LTX_FRAMES_PER_LATENT = 8
LTX_PIXELS_PER_LATENT = 32

OPERATIONS = ["max", "average", "min"]


def _ceil_div(a, b):
    return -(-a // b)


def check_ltx_dimensions(frames, height, width, frames_per_block=LTX_FRAMES_PER_LATENT,
                         pixels_per_block=LTX_PIXELS_PER_LATENT):
    """Raise ValueError describing every way the size doesn't fit LTX's grid."""
    problems = []
    if width % pixels_per_block:
        problems.append(f"width {width} is not a multiple of {pixels_per_block}")
    if height % pixels_per_block:
        problems.append(f"height {height} is not a multiple of {pixels_per_block}")
    if (frames - 1) % frames_per_block:
        problems.append(
            f"frame count {frames} is not of the form {frames_per_block}n+1 "
            f"(e.g. {frames - (frames - 1) % frames_per_block} or "
            f"{frames - (frames - 1) % frames_per_block + frames_per_block})"
        )
    if problems:
        raise ValueError(
            "LTXV Pool Mask Blocks: masks don't fit LTX's latent grid: "
            + "; ".join(problems)
            + ". Resize/trim the masks, or turn off require_ltx_dimensions to pool "
              "partial edge blocks anyway."
        )


def pool_block_values(masks, operation, frames_per_block=LTX_FRAMES_PER_LATENT,
                      pixels_per_block=LTX_PIXELS_PER_LATENT):
    """Pool a (T, H, W) mask tensor within LTX latent blocks.

    Returns the block values as a float32 tensor of shape (T_lat, Hb, Wb):
    one entry per latent frame (frame 0, then each group of frames) and per
    32x32 block. Any size is accepted; partial edge blocks are pooled over the
    pixels they contain.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"operation must be one of {OPERATIONS}, got {operation!r}.")
    if masks.ndim != 3:
        raise ValueError(f"expected a (frames, height, width) mask batch, got shape {tuple(masks.shape)}.")

    t, h, w = masks.shape
    if t == 0 or h == 0 or w == 0:
        raise ValueError(f"mask batch is empty (shape {tuple(masks.shape)}).")

    tf, s = frames_per_block, pixels_per_block
    x = masks.to(torch.float32)

    hb, wb = _ceil_div(h, s), _ceil_div(w, s)
    groups = _ceil_div(t - 1, tf)  # frame groups after frame 0
    tp, hp, wp = 1 + groups * tf, hb * s, wb * s

    # Pad up to whole blocks with a value that cannot influence the result.
    if operation == "max":
        fill = float("-inf")
    elif operation == "min":
        fill = float("inf")
    else:
        fill = 0.0
    padded = torch.full((tp, hp, wp), fill, dtype=torch.float32, device=x.device)
    padded[:t, :h, :w] = x

    def reduce(blocks, dims):
        if operation == "max":
            return blocks.amax(dim=dims)
        if operation == "min":
            return blocks.amin(dim=dims)
        return blocks.sum(dim=dims)

    # Frame 0: its own block in time, pooled in space only.
    first = reduce(padded[0:1].reshape(1, hb, s, wb, s), (2, 4))  # (1, hb, wb)
    # Remaining frames: groups of tf frames.
    rest = None
    if groups > 0:
        rest = reduce(padded[1:].reshape(groups, tf, hb, s, wb, s), (1, 3, 5))  # (groups, hb, wb)

    if operation == "average":
        # Divide by the number of real (unpadded) pixels in each block.
        count = torch.full((tp, hp, wp), 0.0, dtype=torch.float32, device=x.device)
        count[:t, :h, :w] = 1.0
        first = first / count[0:1].reshape(1, hb, s, wb, s).sum(dim=(2, 4))
        if rest is not None:
            rest = rest / count[1:].reshape(groups, tf, hb, s, wb, s).sum(dim=(1, 3, 5))

    return first if rest is None else torch.cat([first, rest], dim=0)


def expand_block_values(blocks, frames, height, width, frames_per_block=LTX_FRAMES_PER_LATENT,
                        pixels_per_block=LTX_PIXELS_PER_LATENT):
    """Expand (T_lat, Hb, Wb) block values back to a (frames, height, width) mask."""
    tf, s = frames_per_block, pixels_per_block
    first = blocks[0:1].repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
    parts = [first]
    if blocks.shape[0] > 1:
        rest = blocks[1:].repeat_interleave(tf, dim=0).repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
        parts.append(rest)
    return torch.cat(parts, dim=0)[:frames, :height, :width]


def pool_mask_blocks(masks, operation, frames_per_block=LTX_FRAMES_PER_LATENT,
                     pixels_per_block=LTX_PIXELS_PER_LATENT):
    """Pool a (T, H, W) mask tensor within LTX latent blocks.

    Returns a tensor of the same shape and dtype.
    """
    blocks = pool_block_values(masks, operation, frames_per_block, pixels_per_block)
    t, h, w = masks.shape
    return expand_block_values(blocks, t, h, w, frames_per_block, pixels_per_block).to(masks.dtype)


class LTXVPoolMaskBlocks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVPoolMaskBlocks",
            display_name="LTXV Pool Mask Blocks",
            category="mask/ltxv",
            description=(
                "Pools a mask frame batch within LTX latent blocks (32x32 pixels; frame 0 alone, "
                "then groups of 8 frames) using max, average or min, and returns a mask batch of "
                "the same size with every pixel of each block set to the block's pooled value."
            ),
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip="Mask frame batch (one mask per video frame).",
                ),
                io.Combo.Input(
                    "operation",
                    options=OPERATIONS,
                    default="max",
                    tooltip="max: a block is as masked as its most-masked pixel. "
                            "average: the block's mean. min: as masked as its least-masked pixel.",
                ),
                io.Boolean.Input(
                    "require_ltx_dimensions",
                    default=True,
                    tooltip="On: stop with an error unless width and height are multiples of 32 "
                            "and the frame count is 8n+1. Off: accept any size; partial edge "
                            "blocks are pooled over the pixels they contain.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="masks",
                    tooltip="Same size as the input; every pixel of each block set to the "
                            "block's pooled value.",
                ),
                io.Mask.Output(
                    display_name="latent_masks",
                    tooltip="One mask per latent frame at latent resolution (width/32 x "
                            "height/32). Apply with LTXV Set Video Latent Noise Masks on a "
                            "latent with no noise mask yet, before adding any guides.",
                ),
            ],
        )

    @classmethod
    def execute(cls, masks, operation, require_ltx_dimensions) -> io.NodeOutput:
        squeeze = masks.ndim == 2  # a single (H, W) mask
        batch = masks.unsqueeze(0) if squeeze else masks
        if batch.ndim != 3:
            raise ValueError(f"masks must be a mask batch, got shape {tuple(masks.shape)}.")
        if require_ltx_dimensions:
            check_ltx_dimensions(*batch.shape)
        t, h, w = batch.shape
        blocks = pool_block_values(batch, operation)
        out = expand_block_values(blocks, t, h, w).to(batch.dtype)
        latent_out = blocks.to(batch.dtype)
        if squeeze:
            out, latent_out = out.squeeze(0), latent_out.squeeze(0)
        return io.NodeOutput(out, latent_out)
