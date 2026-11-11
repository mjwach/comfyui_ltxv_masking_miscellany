"""
LTX Guidance Reroute
====================

Passes images, masks and a guides string straight through, unchanged. Every
input is optional; an unconnected input comes out as None.
"""

from comfy_api.latest import io


class LTXGuidanceReroute(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXGuidanceReroute",
            display_name="LTX Guidance Reroute",
            category="mask/ltxv",
            description="Passes images, masks and guides through unchanged.",
            inputs=[
                io.Image.Input("images", optional=True),
                io.Mask.Input("masks", optional=True),
                io.String.Input("guides", optional=True, force_input=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Mask.Output(display_name="masks"),
                io.String.Output(display_name="guides"),
            ],
        )

    @classmethod
    def execute(cls, images=None, masks=None, guides=None) -> io.NodeOutput:
        return io.NodeOutput(images, masks, guides)
