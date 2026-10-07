"""User-lookup tools are unavailable while a space/project allowlist is active."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError

from mcp_atlassian.confluence.config import ConfluenceConfig
from mcp_atlassian.jira.config import JiraConfig
from mcp_atlassian.servers.context import MainAppContext
from mcp_atlassian.servers.main import main_mcp

pytestmark = pytest.mark.anyio

USER_TOOLS = {
    "jira_get_user_profile",
    "jira_search_assignable_users",
    "confluence_search_user",
}


def _context(projects_filter: str | None, spaces_filter: str | None) -> MagicMock:
    jira_config = MagicMock(spec=JiraConfig)
    jira_config.is_cloud = True
    jira_config.projects_filter = projects_filter
    confluence_config = MagicMock(spec=ConfluenceConfig)
    confluence_config.spaces_filter = spaces_filter
    app_context = MainAppContext(
        full_jira_config=jira_config, full_confluence_config=confluence_config
    )
    request_context = MagicMock()
    request_context.request = None
    request_context.lifespan_context = {"app_lifespan_context": app_context}
    return request_context


async def _listed(projects_filter: str | None, spaces_filter: str | None) -> set[str]:
    with patch.object(main_mcp, "_mcp_server") as mcp_server:
        mcp_server.request_context = _context(projects_filter, spaces_filter)
        return {tool.name for tool in await main_mcp._list_tools_mcp()}


async def test_user_tools_listed_without_filters():
    assert USER_TOOLS <= await _listed(None, None)


@pytest.mark.parametrize(
    ("projects_filter", "spaces_filter", "hidden"),
    [
        ("DEV", None, {"jira_get_user_profile", "jira_search_assignable_users"}),
        (None, "DEV", {"confluence_search_user"}),
        ("DEV", "DEV", USER_TOOLS),
    ],
)
async def test_user_tools_hidden_while_filter_active(
    projects_filter, spaces_filter, hidden
):
    listed = await _listed(projects_filter, spaces_filter)
    assert not listed & hidden
    assert (USER_TOOLS - hidden) <= listed
    # Watchers are bound to a (guarded) issue and stay available.
    assert "jira_get_issue_watchers" in listed


@pytest.mark.security_regression
async def test_blocked_user_tool_cannot_be_called_by_name():
    with (
        patch.object(main_mcp, "_mcp_server") as mcp_server,
        patch.object(FastMCP, "_call_tool_mcp", new_callable=AsyncMock) as executor,
    ):
        mcp_server.request_context = _context("DEV", "DEV")
        for name in USER_TOOLS:
            with pytest.raises(NotFoundError):
                await main_mcp._call_tool_mcp(name, {})
        executor.assert_not_called()


async def test_user_tools_hidden_in_header_only_mode(monkeypatch):
    """Without a global config the allowlist comes from the environment."""
    monkeypatch.setenv("JIRA_PROJECTS_FILTER", "DEV")
    monkeypatch.setenv("CONFLUENCE_SPACES_FILTER", "DEV")
    request_context = MagicMock()
    request_context.request = None
    request_context.lifespan_context = {"app_lifespan_context": MainAppContext()}
    with patch.object(main_mcp, "_mcp_server") as mcp_server:
        mcp_server.request_context = request_context
        ctx = main_mcp._tool_filter_context()
    assert ctx["blocked_toolsets"] == {
        "toolset:jira_users",
        "toolset:confluence_users",
    }
