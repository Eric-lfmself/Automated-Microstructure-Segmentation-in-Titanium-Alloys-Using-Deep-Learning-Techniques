"""Shared segmentation training objectives."""

from .weighted_ce import make_weighted_cross_entropy

__all__ = ["make_weighted_cross_entropy"]
