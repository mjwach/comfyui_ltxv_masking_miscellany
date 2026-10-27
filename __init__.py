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
    from .mask_block_pooling import LTXVPoolMaskBlocks
except ImportError as e:
    LOG.error("LTXV Pool Mask Blocks is unavailable: %s", e)
else:
    _NODES.append(LTXVPoolMaskBlocks)


class _Extension(ComfyExtension):
    async def get_node_list(self):
        return list(_NODES)


async def comfy_entrypoint() -> ComfyExtension:
    return _Extension()
