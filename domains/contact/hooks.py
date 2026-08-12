# domains/contact/hooks.py
"""Builds executor.DomainHooks for the planar-fingertip contact domain --
the contact sibling of option_graph.executor.nav_hooks(). Nothing under
option_graph/ changes to support this beyond the small, additive widening
described in docs/stage1_env_spec.md's Guards section (run_option's guard_ok
call now also passes `leg`, and can terminate on a str return -- nav's own
guard_region is unaffected, see that file's docstring).

region_of and resolve_target are now real, built from a domains.contact.board
.Board derived from params.portals. With zero portals (today's single-room
default) Board degenerates to one region and resolve_target correctly
refuses any edge -- the same placeholder behaviour this file used to hardcode
directly, now one case of the general logic instead of a separate stub.

This module needs no pymunk import itself (Board is pure geometry, and both
score_arrival and guard come from domains.contact_templates, which is pure
numpy) -- only domains/contact/physics.py and
domains/contact/planar_fingertips.py touch pymunk.
"""

from __future__ import annotations

from typing import Any, FrozenSet

from option_graph.executor import DomainHooks

from domains.contact.board import Board


def guard_cells(node: Any) -> FrozenSet[Any]:
    """No cell-grid concept in the contact domain (unlike nav's doorway
    cells) -- guard_ok ignores its `allowed`/`cell_size` arguments entirely,
    kept only for call-compatibility with run_option's generic contract."""
    return frozenset()


def contact_hooks(params, *, template: str = "push",
                  default_finger: str = "L") -> DomainHooks:
    """Build DomainHooks for the contact domain. `template` selects which
    contact-template score_arrival and guard are used, mirroring nav_hooks'
    own `template` parameter. `params` is a PlanarFingertipParams: it carries
    the portal list Board is built from, the board dimensions the guard's
    off_board check needs, and (optionally) a force-abort threshold.

    `default_finger`: run_episode's terminal leg (the final push to the
    episode goal, not a portal) never calls resolve_target, so leg.direction
    is never set there -- only doorway-crossing legs get one. This corridor
    only ever uses one finger, so defaulting the terminal leg to it is
    correct here; a board that pushes with both fingers would need the
    terminal leg to carry a real direction some other way, not a default.
    """
    from types import SimpleNamespace

    from domains.contact_templates import TEMPLATES

    tmpl = TEMPLATES[template]
    board = Board(params.board_w_cm, params.portals)

    def guard_ok(x, allowed, cell_size, leg=None):
        direction = (leg.direction if leg is not None and leg.direction is not None
                    else default_finger)
        return tmpl.guard(x, allowed, cell_size, SimpleNamespace(direction=direction),
                          params=params)

    return DomainHooks(
        region_of=board.region_of,
        resolve_target=board.resolve_target,
        guard_cells=guard_cells,
        guard_ok=guard_ok,
        score_arrival=tmpl.score_arrival,
        cell_size=1.0,  # unused: guard_ok has no grid concept to scale
        template=str(template))
