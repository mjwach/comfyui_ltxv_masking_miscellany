# comfyui_ltxv_masking_miscellany

ComfyUI custom nodes for mask-based video work (inpainting and similar) with
LTX video models.

## Installation

Clone or copy this repository into `ComfyUI/custom_nodes/` and restart
ComfyUI. There are no extra Python dependencies.

Requires a 2026 ComfyUI build. Each node loads independently: LTXV Add Frame
Guides relies on ComfyUI's built-in LTX helpers (`comfy_extras/nodes_lt.py`),
and if a future ComfyUI update renames or removes the ones it needs, the
console says which ones when ComfyUI starts, and the other nodes are still
available.

## Nodes

### LTXV Pool Mask Blocks

*Category: mask/ltxv. File: `mask_block_pooling.py`*

Pools a mask frame batch within LTX's latent blocks, in space and in time, and
returns a mask batch with the same dimensions.

LTX's video VAE turns each 32×32-pixel area of each group of frames into one
latent cell. The frame groups are frame 0 on its own, then frames 1–8, 9–16,
17–24, and so on. A latent noise mask can only hold one value per cell, so a
pixel-level mask always gets reduced to that grid somewhere; this node does it
explicitly, with an operation you choose, and writes each block's pooled value
back to every pixel of every frame in that block.

| Input | Description |
|---|---|
| `masks` | Mask frame batch, one mask per video frame. |
| `operation` | `max` (default): a block is as masked as its most-masked pixel. `average`: the block's mean. `min`: a block is as masked as its least-masked pixel. |
| `require_ltx_dimensions` | Default on. The width and height must be multiples of 32 and the frame count must be of the form 8n+1, or the node stops with an error saying what doesn't fit. When off, any size is accepted (see below). |

| Output | Description |
|---|---|
| `masks` | The pooled batch, same shape as the input: every pixel of each block set to the block's pooled value. For pixel-space uses (blending, guide preprocessing, attention masks, LTXV Add Frame Guides). |
| `latent_masks` | One mask per latent frame at latent resolution (width/32 × height/32): the block values themselves. Apply it with **LTXV Set Video Latent Noise Masks** (see below). |

Notes:

- Use `max` when everything the mask touches must be fully regenerated
  (removing or replacing objects): it guarantees that no latent cell is only
  partly marked, which could otherwise leave remnants of the original content.
  It enlarges the masked area by up to one block (32 px, 8 frames).
- Blocks are aligned to the top-left corner of the frame and to frame 0. With
  `require_ltx_dimensions` off, if the width or height isn't a multiple of 32
  or the frame count isn't 8n+1, the edge blocks are smaller and are pooled
  over the pixels they actually contain.
- The node does nothing else: no growing, thresholding or value remapping. Do
  those before or after it as needed.

#### Setting a latent noise mask

Connect the `latent_masks` output to **LTXV Set Video Latent Noise Masks**. It
takes mask *i* for latent frame *i*, and since `latent_masks` is already at the
latent's resolution, its resize step leaves the values unchanged.

Order matters:

1. Start from a latent with **no noise mask yet**: a freshly VAE-encoded
   video, or an empty latent.
2. **LTXV Set Video Latent Noise Masks** with `latent_masks`. If frame 0 must
   stay as it is in the latent, **frame 0 of your mask must be 0**; if you
   remap mask values (for example so that unmasked areas get some partial
   strength), force frame 0 back to 0 after remapping.
3. **Guide nodes** (IC-LoRA guides, LTXV Add Guide, LTXV Add Frame Guides)
   after that. Each extends the noise mask correctly for the frames it
   appends.

Things that don't work:

- **A latent that already has a noise mask.** LTXV Set Video Latent Noise
  Masks writes into an existing noise mask instead of replacing it. If that
  mask only has one value per frame (Img To Video Condition Only and the
  default guide masks create that kind), it fails with a tensor-size error
  ("The expanded size of the tensor (1) must match the existing size …").
- **Img To Video Condition Only together with a spatial noise mask.** Used
  before the mask node it causes the error above; used after it, it replaces
  the whole noise mask with its own per-frame one. To pin frame 0 to an image
  in the same workflow, use a frame-0 guide instead (LTXV Add Guide with
  frame_idx 0, or LTXV Add Frame Guides with frame 0 selected), which appends
  a guide and leaves the noise mask alone. When the latent is the encoded
  source video, you can also put the image into the source's frame 0 before
  encoding and keep frame 0 of the mask at 0.
- **Setting the noise mask after adding guides.** The latent then also
  contains the appended guide frames, so the mask node either fails with the
  size error above (guide nodes give the latent a per-frame mask) or, if a
  full-size mask was already there, fills the guide frames with your last
  mask, so the guides are no longer frozen.
- **Core Set Latent Noise Mask.** It stores the mask in a 4-dimensional shape,
  which LTX's guide nodes can't extend, so any guide node after it fails. (With
  the full-length `masks` output it would also misplace masks in time, since it
  resamples frames evenly without knowing LTX's frame grouping.)
- **The full-length `masks` output** with LTXV Set Video Latent Noise Masks: it
  would take the first video frames as if they were latent frames.

### LTXV Add Frame Guides

*Category: conditioning/ltxv. File: `frame_guides.py`*

Adds LTX keyframe guides for chosen frames of a full-length image batch. These
are the same kind of guides core LTXV Add Guide adds: appended guide frames,
recorded in the conditioning, and removed after sampling with LTXV Crop Guides.
The node never changes the video part of the latent or its noise mask.

Which frames become guides:

- frames whose mask is completely black (if `guides_from_empty_masks`), and/or
- frames listed in `frame_list` (if `guides_from_frame_list`).

A frame selected by both is used once. Frame 0 always becomes a single-frame
guide. With `coalesce_latent_frames` on (the default), every complete, aligned
8-frame group whose frames are all selected becomes part of a multi-frame
guide, and consecutive complete groups are merged into one guide; all other
selected frames become single-frame guides. Multi-frame guides are encoded
without losing any frames.

| Input | Description |
|---|---|
| `positive`, `negative` | Conditioning. |
| `latent` | Video-only LTX latent (add guides before combining with audio). |
| `vae` | The LTX video VAE. |
| `images` | One image per video frame, exactly the latent's pixel size. |
| `masks` | One mask per frame, same size as the images. |
| `integrity` | 0–1, default 1. Each guide's noise mask is 1 − integrity: 1 keeps the reference exactly as encoded; lower values start it partly noised so the model can reinterpret it. |
| `attention_strength` | 0–10, default 1. How much the generated video attends to the guides: 1 is normal, 0 ignores them, above 1 amplifies them (outside the training range). |
| `frame_list` | Frames to make guides, e.g. `40, 57-64, 120`. Commas only; empty means none. Malformed entries, backwards ranges, negative numbers and frames past the end are errors. |
| `options` | JSON text, or a connected DICT (see below). |

Options:

| Key | Required | Default | Meaning |
|---|---|---|---|
| `guides_from_empty_masks` | yes | | Frames with completely black masks become guides. |
| `guides_from_frame_list` | yes | | Frames in `frame_list` become guides (if false, `frame_list` is ignored). |
| `use_attention_masks` | no | false | Use each guide frame's mask as its attention mask. Frames with empty masks get full attention. |
| `invert_attention_masks` | no | false | Invert those attention masks. |
| `coalesce_latent_frames` | no | true | Merge complete aligned 8-frame groups into multi-frame guides. If false, every guide frame is its own single-frame guide. |

Unknown keys, missing required keys and non-boolean values are errors.

Outputs: `positive`, `negative`, `latent`, and `plan`, a text summary of the
guides that were added.

## License

Public domain (The Unlicense). See `LICENSE`.
