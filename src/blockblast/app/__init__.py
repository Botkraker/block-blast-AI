"""Graphical game (pygame): human play, AI hints and AI autoplay.

Top of the dependency graph next to training/evaluation: imports agents, env, engine.
``BlockBlastApp`` is imported from ``blockblast.app.game_app`` so the display-free
``PlaySession`` works without pygame initialisation.
"""

from blockblast.app.session import PlaySession

__all__ = ["PlaySession"]
