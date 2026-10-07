"""Tests for the fork-specific Confluence space guard."""

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import mcp_atlassian.servers.confluence as confluence_server
from mcp_atlassian.confluence import ConfluenceFetcher
from mcp_atlassian.confluence.config import ConfluenceConfig
from mcp_atlassian.confluence.labels import LabelsMixin
from mcp_atlassian.confluence.pages import PagesMixin
from mcp_atlassian.confluence.space_guard import (
    GUARD_RULES,
    UNGUARDED_METHODS,
    SpaceAccessDeniedError,
    SpaceGuardMixin,
)

# content id -> v1 /rest/api/content/{id}?expand=space,container response
CONTENT: dict[str, dict[str, Any]] = {
    "100": {"id": "100", "space": {"key": "DEV"}},
    "101": {"id": "101", "space": {"key": "dev"}},  # case differs
    "200": {"id": "200", "space": {"key": "SECRET"}},
    # attachment without space, resolved through its container
    "300": {"id": "300", "container": {"id": "100"}},
    "301": {"id": "301", "container": {"id": "200"}},
}


def _content_get(url: str, **_: Any) -> dict[str, Any]:
    content_id = url.rstrip("/").rsplit("/", 1)[1]
    if content_id not in CONTENT:
        raise RuntimeError("404 Not Found")
    return CONTENT[content_id]


def _make_fetcher(spaces_filter: str | None = "DEV") -> ConfluenceFetcher:
    fetcher = ConfluenceFetcher.__new__(ConfluenceFetcher)
    fetcher.config = ConfluenceConfig(
        url="https://example.atlassian.net/wiki",
        auth_type="basic",
        username="user",
        api_token="token",
        spaces_filter=spaces_filter,
    )
    fetcher.confluence = MagicMock()
    fetcher.confluence.get.side_effect = _content_get
    fetcher.confluence.url = "https://example.atlassian.net/wiki"
    return fetcher


@pytest.fixture
def fetcher() -> ConfluenceFetcher:
    return _make_fetcher()


@pytest.fixture
def page_content():
    with patch.object(PagesMixin, "get_page_content", autospec=True) as mock:
        mock.return_value = "PAGE"
        yield mock


# --- completeness (protects against upstream additions/renames) -------------


def _public_fetcher_methods() -> set[str]:
    return {
        name
        for name, member in inspect.getmembers(ConfluenceFetcher)
        if not name.startswith("_") and inspect.isfunction(member)
    }


def test_every_public_method_is_classified():
    unclassified = _public_fetcher_methods() - set(GUARD_RULES) - UNGUARDED_METHODS
    assert not unclassified, (
        "New ConfluenceFetcher methods must be added to GUARD_RULES or "
        f"UNGUARDED_METHODS in space_guard.py: {sorted(unclassified)}"
    )


def test_guard_rules_reference_existing_methods_and_parameters():
    for name, rule in GUARD_RULES.items():
        target = None
        for cls in ConfluenceFetcher.__mro__:
            if cls is not SpaceGuardMixin and name in vars(cls):
                target = vars(cls)[name]
                break
        assert target is not None, f"{name} no longer exists upstream"
        params = set(inspect.signature(target).parameters)
        referenced = (
            rule.content_ids + rule.space_keys + rule.space_ids + rule.download_urls
        )
        missing = set(referenced) - params
        assert not missing, f"{name}: parameters renamed upstream: {missing}"


def test_servers_do_not_bypass_the_fetcher():
    """Tools must not call the raw client except where reviewed."""
    reviewed = {
        # metadata via raw session; the download itself goes through the
        # guarded _resolve_attachment_download_url / fetch_attachment_content
        "download_attachment",
    }
    tree = ast.parse(Path(confluence_server.__file__).read_text(encoding="utf-8"))
    offenders = set()
    for func in ast.walk(tree):
        if not isinstance(func, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in {"confluence", "_v2_adapter"}
                and isinstance(node.value, ast.Name)
                and node.value.id == "confluence_fetcher"
            ):
                offenders.add(func.name)
    unreviewed = offenders - reviewed
    assert not unreviewed, f"Unreviewed raw client access: {unreviewed}"


# --- behaviour --------------------------------------------------------------


def test_no_filter_is_a_passthrough(page_content):
    fetcher = _make_fetcher(spaces_filter=None)
    assert fetcher.get_page_content("200") == "PAGE"
    fetcher.confluence.get.assert_not_called()


def test_allowed_page_passes(fetcher, page_content):
    assert fetcher.get_page_content("100", convert_to_markdown=False) == "PAGE"
    page_content.assert_called_once_with(fetcher, "100", convert_to_markdown=False)


def test_space_keys_are_case_insensitive(fetcher, page_content):
    assert fetcher.get_page_content("101") == "PAGE"


def test_foreign_page_is_denied(fetcher, page_content):
    with pytest.raises(SpaceAccessDeniedError):
        fetcher.get_page_content("200")
    page_content.assert_not_called()


def test_unresolvable_content_is_denied(fetcher, page_content):
    with pytest.raises(SpaceAccessDeniedError, match="Could not verify"):
        fetcher.get_page_content("999")
    page_content.assert_not_called()


@pytest.mark.parametrize("bad_id", ["abc", "100/../200", "100?expand=x", ""])
def test_non_numeric_ids_are_denied_without_lookup(fetcher, page_content, bad_id):
    if bad_id == "":
        # empty IDs are skipped by the guard; upstream validates them
        fetcher.get_page_content(bad_id)
        return
    with pytest.raises(SpaceAccessDeniedError, match="Invalid"):
        fetcher.get_page_content(bad_id)
    fetcher.confluence.get.assert_not_called()


def test_lookups_are_cached(fetcher, page_content):
    fetcher.get_page_content("100")
    fetcher.get_page_content("100")
    assert fetcher.confluence.get.call_count == 1


def test_guard_runs_before_real_upstream_code(fetcher):
    with pytest.raises(SpaceAccessDeniedError):
        fetcher.get_page_labels("200")
    fetcher.confluence.get_page_labels.assert_not_called()


def test_attachment_resolved_through_container(fetcher):
    with patch(
        "mcp_atlassian.confluence.attachments.AttachmentsMixin.delete_attachment",
        autospec=True,
        return_value={"success": True},
    ) as delete:
        assert fetcher.delete_attachment("att300") == {"success": True}
        with pytest.raises(SpaceAccessDeniedError):
            fetcher.delete_attachment("att301")
        assert delete.call_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_space_key": "SECRET"},
        {"target_parent_id": "200"},
    ],
)
def test_move_page_checks_the_target(fetcher, kwargs):
    with patch.object(PagesMixin, "move_page", autospec=True) as move:
        with pytest.raises(SpaceAccessDeniedError):
            fetcher.move_page("100", **kwargs)
        move.assert_not_called()


def test_create_page_in_foreign_space_is_denied(fetcher):
    with patch.object(PagesMixin, "create_page", autospec=True) as create:
        with pytest.raises(SpaceAccessDeniedError):
            fetcher.create_page("SECRET", "t", "b")
        create.assert_not_called()


def test_iterable_arguments_are_checked_element_wise(fetcher):
    with patch(
        "mcp_atlassian.confluence.analytics.AnalyticsMixin.batch_get_page_views",
        autospec=True,
    ) as batch:
        with pytest.raises(SpaceAccessDeniedError):
            fetcher.batch_get_page_views(["100", "200"])
        batch.assert_not_called()


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://x/wiki/download/attachments/100/a.png?api=v2", True),
        ("https://x/wiki/download/attachments/200/a.png", False),
        ("https://x/wiki/rest/api/content/100/child/attachment/att1/download", True),
        ("https://x/wiki/rest/api/content/200/child/attachment/att1/download", False),
        ("https://x/some/other/url", False),
    ],
)
def test_download_urls(fetcher, url, allowed):
    with patch(
        "mcp_atlassian.confluence.attachments.AttachmentsMixin.fetch_attachment_content",
        autospec=True,
        return_value=b"data",
    ):
        if allowed:
            assert fetcher.fetch_attachment_content(url) == b"data"
        else:
            with pytest.raises(SpaceAccessDeniedError):
                fetcher.fetch_attachment_content(url)


def test_search_results_from_foreign_spaces_are_dropped(fetcher):
    def page(pid: str, key: str | None) -> MagicMock:
        p = MagicMock(id=pid)
        p.space = MagicMock(key=key) if key is not None else None
        return p

    results = [page("1", "DEV"), page("2", "SECRET"), page("3", None)]
    with patch(
        "mcp_atlassian.confluence.search.SearchMixin.search",
        autospec=True,
        return_value=results,
    ):
        assert [p.id for p in fetcher.search("text ~ x")] == ["1", "3"]


def test_get_spaces_is_filtered(fetcher):
    with patch(
        "mcp_atlassian.confluence.spaces.SpacesMixin.get_spaces",
        autospec=True,
        return_value={"results": [{"key": "DEV"}, {"key": "SECRET"}], "size": 2},
    ):
        assert fetcher.get_spaces() == {"results": [{"key": "DEV"}], "size": 1}


def test_space_template_from_foreign_space_is_denied(fetcher):
    with patch(
        "mcp_atlassian.confluence.templates.TemplatesMixin.get_page_template",
        autospec=True,
        return_value={"templateId": "1", "space": {"key": "SECRET"}},
    ):
        with pytest.raises(SpaceAccessDeniedError):
            fetcher.get_page_template("1")


def test_global_template_is_allowed(fetcher):
    with patch(
        "mcp_atlassian.confluence.templates.TemplatesMixin.get_page_template",
        autospec=True,
        return_value={"templateId": "1"},
    ):
        assert fetcher.get_page_template("1") == {"templateId": "1"}


def test_space_permissions_resolve_space_id(fetcher):
    with (
        patch(
            "mcp_atlassian.confluence.space_guard.ConfluenceV2Adapter"
            "._get_space_key_from_id",
            side_effect=lambda sid: {"1": "DEV", "2": "SECRET"}[sid],
        ),
        patch(
            "mcp_atlassian.confluence.permissions.PermissionsMixin.get_space_permissions",
            autospec=True,
            return_value={"ok": True},
        ),
    ):
        assert fetcher.get_space_permissions("1") == {"ok": True}
        with pytest.raises(SpaceAccessDeniedError):
            fetcher.get_space_permissions("2")


def test_labels_mixin_still_reachable_for_allowed_pages(fetcher):
    with patch.object(LabelsMixin, "get_page_labels", autospec=True, return_value=[]):
        assert fetcher.get_page_labels("100") == []
