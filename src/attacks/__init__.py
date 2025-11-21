"""Adversarial attacks and Byzantine client simulation."""

from .byzantine_attacks import (
    ByzantineClient,
    RandomAttack,
    LabelFlippingAttack,
    ModelPoisoningAttack,
    GradientAttack,
    create_byzantine_client
)
from .attack_detection import AttackDetector, detect_outliers

__all__ = [
    "ByzantineClient",
    "RandomAttack",
    "LabelFlippingAttack",
    "ModelPoisoningAttack",
    "GradientAttack",
    "create_byzantine_client",
    "AttackDetector",
    "detect_outliers"
]