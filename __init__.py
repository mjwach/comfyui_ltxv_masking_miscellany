"""
comfyui_ltxv_masking_miscellany
===============================

ComfyUI custom nodes for mask-based video work with LTX video models.
See README.md.

Install: put this folder in ComfyUI/custom_nodes/ and restart ComfyUI.

Each node is loaded independently: if one can't load (for example because a
ComfyUI update removed LTX helpers that LTXV Add Frame Guides needs), the error
is logged and the other nodes are still available.
"""

import logging

from comfy_api.latest import ComfyExtension

LOG = logging.getLogger("comfyui_ltxv_masking_miscellany")

_NODES = []

try:
    from .frame_guides import LTXVAddFrameGuides
except ImportError as e:
    LOG.error("LTXV Add Frame Guides is unavailable: %s", e)
else:
    _NODES.append(LTXVAddFrameGuides)

try:
    from .guidance_reroute import LTXGuidanceReroute
except ImportError as e:
    LOG.error("LTX Guidance Reroute is unavailable: %s", e)
else:
    _NODES.append(LTXGuidanceReroute)

try:
    from .mask_block_pooling import LTXVPoolMaskBlocks
except ImportError as e:
    LOG.error("LTXV Pool Mask Blocks is unavailable: %s", e)
else:
    _NODES.append(LTXVPoolMaskBlocks)

try:
    from .build_patching_mask import BuildPatchingMask
except ImportError as e:
    LOG.error("Build Patching Mask is unavailable: %s", e)
else:
    _NODES.append(BuildPatchingMask)

try:
    from .mask_reference_guide import (
        BuildMaskAndReferenceGuideVideo,
        ConfigureMaskAndReferenceGuideVideos,
    )
except ImportError as e:
    LOG.error("Mask and Reference Guide Video nodes are unavailable: %s", e)
else:
    _NODES.append(ConfigureMaskAndReferenceGuideVideos)
    _NODES.append(BuildMaskAndReferenceGuideVideo)


class _Extension(ComfyExtension):
    async def get_node_list(self):
        return list(_NODES)


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
