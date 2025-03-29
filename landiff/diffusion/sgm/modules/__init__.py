from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "landiff.diffusion.sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
