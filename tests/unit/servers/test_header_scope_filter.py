"""Header-PAT configs inherit the space/project allowlist (fork-specific)."""

from unittest.mock import MagicMock

import pytest

from mcp_atlassian.servers.context import MainAppContext
from mcp_atlassian.servers.dependencies import (
    _confluence_spec,
    _header_filter_kwargs,
    _jira_spec,
)


def _ctx(app_context: MainAppContext | None) -> MagicMock:
    ctx = MagicMock()
    ctx.request_context.lifespan_context = (
        {"app_lifespan_context": app_context} if app_context else {}
    )
    return ctx


def test_global_config_filter_wins_over_env(monkeypatch):
    monkeypatch.setenv("JIRA_PROJECTS_FILTER", "ENV")
    jira_config = MagicMock(projects_filter="GLOBAL")
    ctx = _ctx(MainAppContext(full_jira_config=jira_config))
    assert _header_filter_kwargs(ctx, _jira_spec()) == {"projects_filter": "GLOBAL"}


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("DEV,OPS", "DEV,OPS"), ("  ", None), (None, None)],
)
def test_header_only_mode_reads_env(monkeypatch, env_value, expected):
    for var in ("JIRA_PROJECTS_FILTER", "CONFLUENCE_SPACES_FILTER"):
        if env_value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, env_value)
    for ctx in (_ctx(None), _ctx(MainAppContext())):
        assert _header_filter_kwargs(ctx, _jira_spec()) == {"projects_filter": expected}
        assert _header_filter_kwargs(ctx, _confluence_spec()) == {
            "spaces_filter": expected
        }
