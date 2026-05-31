from .film import FiLMConditioning
from .adain import AdaINConditioning
from .cross_attention import CrossAttentionConditioning

FUSION_REGISTRY = {
    "film": FiLMConditioning,
    "adain": AdaINConditioning,
    "cross_attention": CrossAttentionConditioning,
}


def build_fusion(name: str, **kwargs):
    """Factory function to build a fusion module by name."""
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion type '{name}'. Choose from {list(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](**kwargs)
