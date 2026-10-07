"""Shared machinery for the fork-specific space/project guards.

``confluence.space_guard`` and ``jira.project_guard`` each define a mixin that
sits first in its fetcher's MRO and a table of rules naming which arguments of
which fetcher methods must be checked. ``install_guards`` turns that table into
wrapper methods on the mixin. Each wrapper binds the call's arguments to the
real method's signature, lets the mixin enforce the rule, calls the real
method and lets the mixin post-process the result.

A guard mixin must implement:

- ``_scope_guard_active() -> bool``: False makes every wrapper a passthrough.
- ``_enforce_rule(rule, arguments)``: raise to deny the call.
- ``_guard_result(rule, result)``: return the (possibly filtered) result.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def iter_values(value: Any) -> Iterable[str]:
    """Normalize an argument that may be a single ID or a collection of IDs.

    ``None`` and empty strings are skipped; upstream code validates them.
    """
    if value is None or value == "":
        return ()
    if isinstance(value, str | int):
        return (str(value),)
    return tuple(str(v) for v in value if v not in (None, ""))


def configured_scope_filter(config: Any, attr: str, env_var: str) -> Any:
    """Return the space/project allowlist that applies to a request.

    Uses the global (lifespan) config when one exists; otherwise falls back to
    the environment variable it would have been read from, so header-only
    deployments without a global config are filtered too.

    Args:
        config: The global JiraConfig/ConfluenceConfig, or None.
        attr: Config attribute holding the filter (e.g. ``projects_filter``).
        env_var: Environment variable holding the filter.

    Returns:
        The filter value, or None when no filter is configured.
    """
    value = getattr(config, attr, None) if config is not None else os.getenv(env_var)
    if isinstance(value, str):
        value = value.strip()
    return value or None


def install_guards(mixin: type, rules: Mapping[str, Any]) -> None:
    """Install one guarded wrapper per rule on ``mixin``.

    Args:
        mixin: The guard mixin class; must come first in the fetcher's MRO.
        rules: Mapping of fetcher method name to the product-specific rule.
    """
    for name, rule in rules.items():
        setattr(mixin, name, _make_guarded(mixin, name, rule))


def _make_guarded(mixin: type, name: str, rule: Any) -> Callable[..., Any]:
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        target = getattr(super(mixin, self), name)
        if not self._scope_guard_active():
            return target(*args, **kwargs)
        bound = inspect.signature(target).bind_partial(*args, **kwargs)
        self._enforce_rule(rule, bound.arguments)
        return self._guard_result(rule, target(*args, **kwargs))

    guarded.__name__ = name
    guarded.__qualname__ = f"{mixin.__name__}.{name}"
    guarded.__doc__ = f"Scope-guarded wrapper around ``{name}``."
    return guarded
