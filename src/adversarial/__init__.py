"""Adversarial attack simulation for federated learning."""

from .byzantine_attacks import (
    ByzantineClient,
    RandomAttack,
    SignFlippingAttack,
    LabelFlippingAttack,
    GaussianNoiseAttack,
    create_byzantine_client
)
from .attack_simulator import AttackSimulator, simulate_attack

__all__ = [
    "ByzantineClient",
    "RandomAttack",
    "SignFlippingAttack",
    "LabelFlippingAttack",
    "GaussianNoiseAttack",
    "create_byzantine_client",
    "AttackSimulator",
    "simulate_attack"
]