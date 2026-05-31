"""Loss functions for EmotiLip training."""

from .emotion_consistency import EmotionConsistencyLoss
from .prosody_loss import ProsodyLoss

__all__ = ["EmotionConsistencyLoss", "ProsodyLoss"]
