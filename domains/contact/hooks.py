# domains/contact/hooks.py
"""Builds executor.DomainHooks for the planar-fingertip domain -- the contact
sibling of option_graph.executor.nav_hooks().

region_of and resolve_target come from a Board built off params.portals. With
zero portals (the single-room default) Board degenerates to one region and
resolve_target refuses every edge, so the placeholder case falls out of the
general logic. No pymunk import here: Board and contact_templates are pure.
"""

from __future__ import annotations

from typing import Any, FrozenSet

from option_graph.executor import DomainHooks

from domains.contact.board import Board


def guard_cells(node: Any) -> FrozenSet[Any]:
    """No cell grid in this domain; guard_ok ignores `allowed`/`cell_size`,
    which exist only for run_option's generic contract."""
    return frozenset()


def contact_hooks(params, *, template: str = "push",
                  default_finger: str = "L") -> DomainHooks:
    """`template` picks the score_arrival and guard; `params` carries the portal
    list, board dimensions, and optional force-abort threshold.

    `default_finger` covers the terminal leg, which never calls resolve_target
    and so has no leg.direction. Correct only because this corridor pushes with
    one finger -- a two-finger board would need a real direction there.
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
