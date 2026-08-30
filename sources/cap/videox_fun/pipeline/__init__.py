import importlib

from .pipeline_cogvideox_fun import CogVideoXFunPipeline
from .pipeline_cogvideox_fun_control import CogVideoXFunControlPipeline
from .pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline
from .pipeline_flux import FluxPipeline
from .pipeline_flux2 import Flux2Pipeline
from .pipeline_flux2_control import Flux2ControlPipeline
from .pipeline_hunyuanvideo import HunyuanVideoPipeline
from .pipeline_hunyuanvideo_i2v import HunyuanVideoI2VPipeline
from .pipeline_ltx2_i2v import LTX2I2VPipeline
from .pipeline_ltx2 import LTX2Pipeline
from .pipeline_qwenimage import QwenImagePipeline
from .pipeline_qwenimage_control import QwenImageControlPipeline
from .pipeline_qwenimage_instantx import QwenImageControlNetPipeline
from .pipeline_qwenimage_edit import QwenImageEditPipeline
from .pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from .pipeline_qwenimage_layered import QwenImageLayeredPipeline
from .pipeline_wan import WanPipeline
from .pipeline_wan2_2 import Wan2_2Pipeline
from .pipeline_wan2_2_animate import Wan2_2AnimatePipeline
from .pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline
from .pipeline_wan2_2_fun_inpaint import Wan2_2FunInpaintPipeline
from .pipeline_wan2_2_ti2v import Wan2_2TI2VPipeline
from .pipeline_wan2_2_vace_fun import Wan2_2VaceFunPipeline
from .pipeline_wan_fun_control import WanFunControlPipeline
from .pipeline_wan_fun_inpaint import WanFunInpaintPipeline
from .pipeline_wan_phantom import WanFunPhantomPipeline
from .pipeline_wan_vace import WanVacePipeline
from .pipeline_z_image import ZImagePipeline
from .pipeline_z_image_control import ZImageControlPipeline


_OPTIONAL_AUDIO_PIPELINES = {
    "FantasyTalkingPipeline": (".pipeline_fantasy_talking", "FantasyTalkingPipeline"),
    "LongCatVideoPipeline": (".pipeline_longcatvideo", "LongCatVideoPipeline"),
    "LongCatVideoAvatarPipeline": (
        ".pipeline_longcatvideo_avatar",
        "LongCatVideoAvatarPipeline",
    ),
    "Wan2_2S2VPipeline": (".pipeline_wan2_2_s2v", "Wan2_2S2VPipeline"),
}


def __getattr__(name):
    """Load audio pipelines only when callers explicitly request them."""
    target = _OPTIONAL_AUDIO_PIPELINES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

WanFunPipeline = WanPipeline
WanI2VPipeline = WanFunInpaintPipeline

Wan2_2FunPipeline = Wan2_2Pipeline
Wan2_2I2VPipeline = Wan2_2FunInpaintPipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan2.1
    WanFunInpaintPipeline.__call__ = sparse_reset(WanFunInpaintPipeline.__call__)
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanFunControlPipeline.__call__ = sparse_reset(WanFunControlPipeline.__call__)
    WanI2VPipeline.__call__ = sparse_reset(WanI2VPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)
    WanVacePipeline.__call__ = sparse_reset(WanVacePipeline.__call__)

    # Phantom
    WanFunPhantomPipeline.__call__ = sparse_reset(WanFunPhantomPipeline.__call__)

    # Wan2.2
    Wan2_2FunInpaintPipeline.__call__ = sparse_reset(Wan2_2FunInpaintPipeline.__call__)
    Wan2_2FunPipeline.__call__ = sparse_reset(Wan2_2FunPipeline.__call__)
    Wan2_2FunControlPipeline.__call__ = sparse_reset(Wan2_2FunControlPipeline.__call__)
    Wan2_2Pipeline.__call__ = sparse_reset(Wan2_2Pipeline.__call__)
    Wan2_2I2VPipeline.__call__ = sparse_reset(Wan2_2I2VPipeline.__call__)
    Wan2_2TI2VPipeline.__call__ = sparse_reset(Wan2_2TI2VPipeline.__call__)
    Wan2_2S2VPipeline.__call__ = sparse_reset(Wan2_2S2VPipeline.__call__)
    Wan2_2VaceFunPipeline.__call__ = sparse_reset(Wan2_2VaceFunPipeline.__call__)
    Wan2_2AnimatePipeline.__call__ = sparse_reset(Wan2_2AnimatePipeline.__call__)
