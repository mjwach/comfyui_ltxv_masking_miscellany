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

### LTX Guidance Reroute

*Category: mask/ltxv. File: `guidance_reroute.py`*

Passes three values through unchanged: `images` (IMAGE), `masks` (MASK) and
`guides` (STRING), each to the output of the same name and type. No input is
required; an unconnected input comes out as None. It does nothing else.

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

### Build Patching Mask

*Category: mask/ltxv. File: `build_patching_mask.py`*

Plans where a patch (a stretch of frames to regenerate) goes in a video, and
picks a working section of the video around it. The working section comes out
as a DICT shaped like a VideoToExtend from the
[comfyui_video_to_extend](https://github.com/mjwach/comfyui_video_to_extend)
pack, so it can go straight into that pack's nodes; this pack doesn't depend
on it.

| Input | Description |
|---|---|
| `video` | A video in ComfyUI's input folder, with an upload button (the same list as VideoHelperSuite's Load Video (Upload)). |
| `patch_start` | First frame of the patch, counted from the start of the whole video. |
| `patch_length` | Number of frames in the patch (at least 1). |
| `patch_padding` | Frames added to each end of the patch before anything else happens: start 35, length 3, padding 4 is exactly the same as start 31, length 11, padding 0. |
| `working_size` | Length of the working section, default 5. |
| `working_size_unit` | `seconds` (default; multiplied by the video's frame rate) or `frames`. |

| Output | Description |
|---|---|
| `vte` | DICT with `video_file`, `is_null` (false), `prefix_to_sustain` and `working_frame_count`. |
| `patch_start` | First frame of the patch, counted from the start of the working frames. |
| `patch_length` | Frames in the patch. |

How the numbers are chosen:

- **Working frame count:** `working_size` in frames, moved to the nearest
  8n+1 count (halfway rounds up). If that's longer than the video, the next
  smaller 8n+1 count is used if it fits; otherwise it's an error.
- **Patch:** the smallest run of whole latent frames of the working set (as
  LTX's VAE would group them if the working frames were encoded: frame 0
  alone, then groups of 8) that holds the padded patch, with the padded patch
  about centered in it. So `patch_start` is always 8k+1 and `patch_length` a
  multiple of 8.
- **Prefix:** any length, chosen so the patch sits as near the middle of the
  working section as the video allows. There's always at least one latent
  frame of the working section before the patch and one after it.
- If no placement fits (the patch is too close to the start or end of the
  video, or too long for the working section), one more latent frame of
  patch length is tried, which gives more freedom of placement; if that fails
  too, it's an error.

Errors stop the workflow when it's queued, before anything runs, as long as
the widgets hold values rather than links (linked values are checked when the
node runs). Frames are counted the same way comfyui_video_to_extend counts
them, so the two packs agree on a video's length.

### Configure Mask and Reference Guide Videos / Build Mask and Reference Guide Video

*Category: mask/ltxv. File: `mask_reference_guide.py`*

Build guide videos in the layout used by Alissonerdx's `ltx23_inpaint_masked_r2v`,
a reference-based inpainting LoRA (this is also the layout BFSNodes'
Reserved Region Frame Composer makes): a strip along one side of the frame
filled with chroma green, with the reference image on a white card in it, and
the rest of the frame holding the video scaled to fit, with black bars where
it doesn't fill the space and the masked area painted solid magenta. Pass the
result to LTXVAddGuideMulti as the LoRA's documentation describes.

**Configure Mask and Reference Guide Videos** makes a configuration that any
number of Build nodes can share (for example the guides of both stages of a
two-stage workflow).

| Input | Description |
|---|---|
| `reference_image` | The object to insert: a single image. |
| `reference_alpha` | Optional MASK, 1 = object, 0 = background. If not connected, `reference_image` must have an alpha channel. Load Image's MASK output is *inverted* alpha; put Invert Mask in between. |
| `mask_color`, `zone_color` | Optional INT, 0xRRGGBB (the `rgb_int` output of Color to RGB Int). Defaults: magenta (255, 0, 255) and chroma green (0, 255, 0). They must differ. |
| `auto_resize` | Default off. Grow the frame, in 32 px steps and keeping the video's aspect ratio, so the video keeps its full resolution beside the zone (see below). |
| `zone_location` | `left` (default), `right`, `top`, `bottom`, or a pair of sides meeting at a corner with one preferred, e.g. `left or top (left preferred)`. |
| `target_thickness`, `min_thickness`, `max_thickness` | Zone thickness, default 256 each. Without auto-resize the thickness is the target clamped to [min, max]; min > max is an error. Auto-resize uses only multiples of 32 within [min, max] (or the clamped target if there are none), preferring those near the target. |
| `allow_rotation` | Default off. Allow turning the reference a quarter turn when that shows it substantially larger. |
| `rotate_ccw` | Default on: rotation is counterclockwise; off: clockwise. |
| `max_pixels` | Default -1 (no limit). Largest frame (width × height) auto-resize may grow to. It never makes a frame smaller than the input; 0 effectively turns auto-resize off. |

**Build Mask and Reference Guide Video**

| Input | Description |
|---|---|
| `video_images` | The video frames. |
| `video_masks` | One mask per frame, or a single mask for every frame, the same size as the frames; 1 = regenerate. |
| `configuration` | From Configure Mask and Reference Guide Videos. |
| `pool_mask_blocks` | Default on. After layout, pool the mask to LTX latent blocks exactly as LTXV Pool Mask Blocks does with `max` (32×32 px on the final frame's grid; frame 0, then groups of 8 frames), clipped to the video's rectangle. |

| Output | Description |
|---|---|
| `guide_images` | The laid-out guide video. |
| `resized` | True if auto-resize changed the frame size. |
| `report` | Text: the candidate layouts, their scores, and what was chosen. |

With the defaults, the geometry (zone, video and card rectangles) is the same
as Reserved Region Frame Composer's with region left, 256 px, scale 90%,
padding 12, centered. Differences:

- The reference is composited onto white with its alpha and **cropped to the
  bounding box of its visible pixels** before the 16 px white border is added,
  so the object fills more of the card. A fully transparent reference gives a
  blank white card, which the LoRA treats as "no reference" (text-only
  inpainting).
- The mask is never Lanczos-resampled: it's area-averaged to the video's new
  size and thresholded at 0.5, so the painted area is solid with clean edges.
- Video frames are resampled (Lanczos) in floating point, and copied 1:1 when
  their size doesn't change.

How the layout is chosen. The LoRA turns the video area of the guide into the
whole output frame, and in training the two always had the same aspect ratio:
the video was shrunk uniformly to fit beside the zone, with black bars above
and below. Every layout here keeps that.

- Without auto-resize, the frame is the input's size and the thickness is the
  clamped target.
- With auto-resize, the frame grows in 32 px steps (within `max_pixels`, and
  never giving the video less room than the input's own frame size would) to
  hold the video at full resolution or slightly more, with a thickness that's
  a multiple of 32 so the zone's edge is on a latent-cell boundary, as in
  training. The frame's aspect ratio matches the video's as closely as 32 px
  steps allow; the small mismatch left over (usually under 1%) means the
  output comes out stretched by that much. 1280×720 becomes 1536×864 with the
  video at 1280×720 and 72 px bars, exactly. Resize the sampled output back to
  the input's size; the report gives the factors and the aspect error.

Then each allowed side, with and without rotation, is scored: video detail
kept (most weight), reference detail kept, a small bonus for the preferred
side (a tiebreaker) and a penalty for rotation. The weights are constants at
the top of `mask_reference_guide.py`.

Errors (wrong shapes or sizes, mismatched frame counts, a missing alpha, equal
colors, min > max, no room for the video) stop the workflow before anything is
drawn. The report warns when the frame or the zone thickness isn't a multiple
of 32 (without auto-resize, the input's size and the target are used as
given) or the frame count isn't 8n+1.

## License

Public domain (The Unlicense). See `LICENSE`.
