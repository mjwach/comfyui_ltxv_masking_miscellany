"""
Build Patching Mask
===================

Plans where a "patch" (a stretch of frames to regenerate) goes, for a video in
ComfyUI's input folder, and outputs:

  vte           a DICT usable as a VideoToExtend by the comfyui_video_to_extend
                pack: video_file, is_null (false), prefix_to_sustain and
                working_frame_count. This pack doesn't depend on that one; the
                DICT is just shaped the same way.
  patch_start   first frame of the patch, counted from the start of the
                working frames (not of the whole video)
  patch_length  frame count of the patch

Steps:
  1. The requested patch (patch_start, patch_length, in whole-video frames) is
     grown by patch_padding frames at each end. (Start 35, length 3, padding 4
     is exactly the same as start 31, length 11, padding 0.)
  2. The working frame count comes from working_size, in seconds (times the
     video's frame rate) or in frames, moved to the nearest LTX-VAE-friendly
     8n+1 count. If that's longer than the video, the next smaller 8n+1 count
     is used, if it fits; otherwise it's an error.
  3. The output patch is the smallest run of whole LTX latent frames of the
     working set (if the working set were VAE-encoded: frame 0 alone, then
     groups of 8) that holds the padded patch, about centered on it. The
     prefix (any length) is chosen so that this lands as near the middle of
     the working set as possible, with at least one latent frame of the
     working set before and after it. If no placement within the video
     allows that, one more latent frame is tried; if that fails too, it's an
     error.

All checks run when the workflow is queued (if the widgets are values, not
links), so impossible settings stop the workflow before anything executes.
Frames are counted the same way comfyui_video_to_extend counts them (by
reading video packets, without decoding), so the two packs agree.
"""

import logging
import os

from comfy_api.latest import io

import folder_paths

LOG = logging.getLogger("BuildPatchingMask")

# Same list VideoHelperSuite and comfyui_video_to_extend use for the picker.
VIDEO_EXTENSIONS = ("webm", "mp4", "mkv", "gif", "mov")
UNITS = ("seconds", "frames")
BIG = 1 << 30


# --------------------------------------------------------------------------
# Video file helpers
# --------------------------------------------------------------------------

def list_input_videos():
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            parts = f.split(".")
            if len(parts) > 1 and parts[-1].lower() in VIDEO_EXTENSIONS:
                files.append(f)
    return sorted(files)


def resolve_video_path(video):
    name = str(video).strip().strip('"')
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    path = folder_paths.get_annotated_filepath(name)
    if path and os.path.isfile(path):
        return path
    raise ValueError(f"video file not found: {video!r}")


def file_signature(path):
    st = os.stat(path)
    return f"{path}|{st.st_mtime_ns}|{st.st_size}"


_probe_cache = {}


def probe_video(path):
    """(frame_count, fps) of the first video stream, cached per file version."""
    key = file_signature(path)
    hit = _probe_cache.get(key)
    if hit is not None:
        return hit
    import av
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path!r} has no video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        frames = 0
        for packet in container.demux(stream):
            if packet.size and not getattr(packet, "is_discard", False):
                frames += 1
    if len(_probe_cache) >= 64:
        _probe_cache.clear()
    _probe_cache[key] = (frames, fps)
    return frames, fps


# --------------------------------------------------------------------------
# Planning (pure arithmetic)
# --------------------------------------------------------------------------

def nearest_8n1(target):
    """The 8n+1 count nearest to target (halfway rounds up), at least 1."""
    n = int((target - 1) / 8 + 0.5) if target > 1 else 0
    return 8 * max(0, n) + 1


def plan_patch(total, fps, patch_start, patch_length, patch_padding,
               working_size, working_size_unit):
    """Return (prefix, working, patch_start, patch_length) or raise ValueError.

    total and fps describe the video; the rest are the node's widget values.
    The returned patch_start is relative to the working frames.
    """
    if patch_start < 0:
        raise ValueError(f"patch_start must be >= 0 (got {patch_start}).")
    if patch_length < 1:
        raise ValueError(f"patch_length must be >= 1 (got {patch_length}).")
    if patch_padding < 0:
        raise ValueError(f"patch_padding must be >= 0 (got {patch_padding}).")
    if working_size < 1:
        raise ValueError(f"working_size must be >= 1 (got {working_size}).")
    if working_size_unit not in UNITS:
        raise ValueError(f"working_size_unit must be one of {UNITS}.")

    # 1. Padding, before anything else.
    start = patch_start - patch_padding
    length = patch_length + 2 * patch_padding
    end = start + length
    padded = (f"the patch (frames {start}-{end - 1} after padding)"
              if patch_padding else f"the patch (frames {start}-{end - 1})")
    if start < 0:
        raise ValueError(f"{padded} starts before the first frame of the video.")
    if end > total:
        raise ValueError(
            f"{padded} ends past the end of the video ({total} frames, "
            f"last frame {total - 1}).")

    # 2. Working frame count, on the 8n+1 grid.
    if working_size_unit == "seconds":
        if fps <= 0:
            raise ValueError(
                "the video's frame rate is unknown, so working_size can't be "
                "given in seconds; use frames instead.")
        target = working_size * fps
    else:
        target = working_size
    working = nearest_8n1(target)
    if working > total:
        smaller = working - 8
        if smaller < 1 or smaller > total:
            raise ValueError(
                f"the working set ({working} frames, from working_size "
                f"{working_size} {working_size_unit}) is longer than the "
                f"video ({total} frames).")
        working = smaller
    groups = (working - 1) // 8  # latent frames after latent frame 0

    # 3. Latent-aligned patch near the middle of the working set.
    m_min = max(1, -(-length // 8))
    if groups - m_min - 1 < 0:
        raise ValueError(
            f"the working set ({working} frames) is too short for {padded}: "
            f"a {8 * m_min}-frame aligned patch with a latent frame on each "
            f"side needs a working set of at least {8 * m_min + 9} frames.")
    room = total - working  # largest possible prefix
    for m in (m_min, m_min + 1):
        L = 8 * m
        slack = L - length
        ideal = (working - L - 2) / 16  # a that centers the patch
        best = None
        for o in range(slack + 1):  # padded patch's offset inside the patch
            # s = 1 + 8a, prefix = start - o - s must be in [0, room],
            # and a + m <= groups - 1 (one latent frame after the patch).
            lo = max(0, -(-(start - o - room - 1) // 8))
            hi = min(groups - m - 1, (start - o - 1) // 8)
            if lo > hi:
                continue
            for a in {min(max(int(ideal), lo), hi),
                      min(max(int(ideal) + 1, lo), hi)}:
                s = 1 + 8 * a
                imbalance = abs(s - (working - s - L)) / 2
                cost = (imbalance + abs(2 * o - slack) / 2, a, o)
                if best is None or cost < best[0]:
                    best = (cost, start - o - s, s)
        if best is not None:
            _, prefix, s = best
            return prefix, working, s, L
    raise ValueError(
        f"{padded} can't be placed inside a {working}-frame working set "
        f"within this {total}-frame video with at least one latent frame of "
        "the working set before and after it. Move the patch away from the "
        "start or end of the video, or use a smaller working_size.")


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class BuildPatchingMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BuildPatchingMask",
            display_name="Build Patching Mask",
            category="mask/ltxv",
            description=(
                "Picks a working section of a video around a patch, sized to an LTX-friendly "
                "8n+1 frame count, and outputs it as a VideoToExtend-compatible DICT plus the "
                "patch's position within the working frames, aligned to LTX latent frames."
            ),
            inputs=[
                io.Combo.Input(
                    "video",
                    options=list_input_videos(),
                    upload=io.UploadType.video,
                    tooltip="Video in ComfyUI's input folder.",
                ),
                io.Int.Input(
                    "patch_start", default=0, min=0, max=BIG,
                    tooltip="First frame of the patch, counted from the start of the whole video.",
                ),
                io.Int.Input(
                    "patch_length", default=1, min=1, max=BIG,
                    tooltip="Number of frames in the patch.",
                ),
                io.Int.Input(
                    "patch_padding", default=0, min=0, max=BIG,
                    tooltip="Frames added to each end of the patch before anything else.",
                ),
                io.Int.Input(
                    "working_size", default=5, min=1, max=BIG,
                    tooltip="Length of the working section, in the unit below. Moved to the "
                            "nearest 8n+1 frame count.",
                ),
                io.Combo.Input(
                    "working_size_unit", options=list(UNITS), default="seconds",
                    tooltip="Whether working_size is in seconds (using the video's frame rate) "
                            "or in frames.",
                ),
            ],
            outputs=[
                io.Dict.Output(
                    display_name="vte",
                    tooltip="VideoToExtend-compatible DICT: video_file, is_null, "
                            "prefix_to_sustain, working_frame_count.",
                ),
                io.Int.Output(
                    display_name="patch_start",
                    tooltip="First frame of the patch, counted from the start of the working "
                            "frames. Always on an LTX latent-frame boundary.",
                ),
                io.Int.Output(
                    display_name="patch_length",
                    tooltip="Frames in the patch: a whole number of LTX latent frames (a "
                            "multiple of 8).",
                ),
            ],
        )

    @classmethod
    def _plan(cls, video, patch_start, patch_length, patch_padding,
              working_size, working_size_unit):
        total, fps = probe_video(resolve_video_path(video))
        return plan_patch(total, fps, int(patch_start), int(patch_length),
                          int(patch_padding), int(working_size),
                          working_size_unit)

    @classmethod
    def validate_inputs(cls, video=None, patch_start=None, patch_length=None,
                        patch_padding=None, working_size=None,
                        working_size_unit=None):
        # Runs when the workflow is queued. Inputs fed by links arrive as None
        # and are checked again in execute().
        values = (video, patch_start, patch_length, patch_padding,
                  working_size, working_size_unit)
        try:
            if video is not None:
                resolve_video_path(video)
            if all(v is not None for v in values):
                cls._plan(*values)
        except Exception as e:
            return str(e)
        return True

    @classmethod
    def fingerprint_inputs(cls, video=None, **kwargs):
        # Re-run when the file itself changes on disk.
        try:
            return file_signature(resolve_video_path(video))
        except Exception:
            return ""

    @classmethod
    def execute(cls, video, patch_start, patch_length, patch_padding,
                working_size, working_size_unit) -> io.NodeOutput:
        try:
            prefix, working, start, length = cls._plan(
                video, patch_start, patch_length, patch_padding,
                working_size, working_size_unit)
        except ValueError as e:
            raise ValueError(f"Build Patching Mask: {e}") from None
        vte = {
            "video_file": video,
            "is_null": False,
            "prefix_to_sustain": prefix,
            "working_frame_count": working,
        }
        return io.NodeOutput(vte, start, length)
