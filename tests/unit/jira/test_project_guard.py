"""Tests for the fork-specific Jira project guard."""

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import mcp_atlassian.servers.jira as jira_server
from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.project_guard import (
    GUARD_RULES,
    UNGUARDED_METHODS,
    ProjectAccessDeniedError,
    ProjectGuardMixin,
)
from mcp_atlassian.models.jira import JiraIssue, JiraSearchResult
from mcp_atlassian.models.jira.link import JiraIssueLink, JiraLinkedIssue

# issue key/id -> current project (DEV-5 was moved to SECRET)
ISSUES = {"DEV-1": "DEV", "DEV-2": "DEV", "DEV-5": "SECRET", "10001": "DEV"}
ISSUES |= {"10002": "SECRET", "SECRET-1": "SECRET"}
PROJECTS = {"1": "DEV", "2": "SECRET"}
BOARDS = {"1": "DEV", "2": "SECRET"}  # board 3 has no project location
SPRINTS = {"11": 1, "12": 2}
SERVICE_DESKS = {"5": "DEV", "6": "SECRET"}
VERSIONS = {"100": 1, "200": 2}
LINKS = {
    "7": {"inwardIssue": {"key": "DEV-1"}, "outwardIssue": {"key": "DEV-2"}},
    "8": {"inwardIssue": {"key": "DEV-1"}, "outwardIssue": {"key": "SECRET-1"}},
}


def _not_found(*_: Any, **__: Any) -> None:
    raise RuntimeError("404 Not Found")


def _make_jira_client() -> MagicMock:
    client = MagicMock()
    client.issue.side_effect = lambda key, fields=None: (
        {"fields": {"project": {"key": ISSUES[key]}}} if key in ISSUES else _not_found()
    )
    client.get_project.side_effect = lambda pid: {"key": PROJECTS[pid]}
    client.get_agile_board.side_effect = lambda bid: (
        {"location": {"projectKey": BOARDS[bid]}} if bid in BOARDS else {"id": bid}
    )
    client.get_sprint.side_effect = lambda sid: {"originBoardId": SPRINTS[sid]}
    client.get.side_effect = lambda path: {
        "projectKey": SERVICE_DESKS[path.rsplit("/", 1)[1]]
    }
    client.get_version.side_effect = lambda vid: {"projectId": VERSIONS[vid]}
    client.get_issue_link.side_effect = lambda lid: LINKS[lid]
    return client


def _make_fetcher(projects_filter: str | None = "DEV") -> JiraFetcher:
    fetcher = JiraFetcher.__new__(JiraFetcher)
    fetcher.config = MagicMock(projects_filter=projects_filter)
    fetcher.jira = _make_jira_client()
    return fetcher


@pytest.fixture
def fetcher() -> JiraFetcher:
    return _make_fetcher()


def _patch(method: str, **kwargs: Any) -> Any:
    """Patch the real (upstream) implementation behind the guard."""
    for cls in JiraFetcher.__mro__:
        if cls is not ProjectGuardMixin and method in vars(cls):
            return patch.object(cls, method, autospec=True, **kwargs)
    raise AssertionError(method)


# --- completeness (protects against upstream additions/renames) -------------


def test_every_public_method_is_classified():
    public = {
        name
        for name, member in inspect.getmembers(JiraFetcher)
        if not name.startswith("_") and inspect.isfunction(member)
    }
    unclassified = public - set(GUARD_RULES) - UNGUARDED_METHODS
    assert not unclassified, (
        "New JiraFetcher methods must be added to GUARD_RULES or "
        f"UNGUARDED_METHODS in project_guard.py: {sorted(unclassified)}"
    )


def test_guard_rules_reference_existing_methods_and_parameters():
    param_fields = (
        "issues",
        "projects",
        "boards",
        "sprints",
        "service_desks",
        "versions",
        "links",
        "link_payloads",
        "issue_payloads",
    )
    for name, rule in GUARD_RULES.items():
        target = next(
            (
                vars(cls)[name]
                for cls in JiraFetcher.__mro__
                if cls is not ProjectGuardMixin and name in vars(cls)
            ),
            None,
        )
        assert target is not None, f"{name} no longer exists upstream"
        params = set(inspect.signature(target).parameters)
        referenced = {p for field in param_fields for p in getattr(rule, field)}
        missing = referenced - params
        assert not missing, f"{name}: parameters renamed upstream: {missing}"


def test_servers_do_not_bypass_the_fetcher():
    """Tools must not use the raw client or private fetcher helpers."""
    reviewed = {"_is_internal_only_project"}  # pure config check, no API call
    tree = ast.parse(Path(jira_server.__file__).read_text(encoding="utf-8"))
    offenders = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "jira"
        and (node.attr == "jira" or node.attr.startswith("_"))
    }
    unreviewed = offenders - reviewed
    assert not unreviewed, f"Unreviewed raw client access: {unreviewed}"


# --- issues -----------------------------------------------------------------


def test_no_filter_is_a_passthrough():
    fetcher = _make_fetcher(projects_filter=None)
    with _patch("get_issue_comments", return_value=["c"]):
        assert fetcher.get_issue_comments("SECRET-1") == ["c"]
    fetcher.jira.issue.assert_not_called()


def test_allowed_issue_passes(fetcher):
    with _patch("get_issue_comments", return_value=["c"]) as real:
        assert fetcher.get_issue_comments("DEV-1", limit=5) == ["c"]
        real.assert_called_once_with(fetcher, "DEV-1", limit=5)


def test_foreign_prefix_is_denied_without_lookup(fetcher):
    with _patch("get_issue_comments") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.get_issue_comments("SECRET-1")
        real.assert_not_called()
    fetcher.jira.issue.assert_not_called()


def test_moved_issue_old_key_is_denied(fetcher):
    """DEV-5 has an allowed prefix but now lives in SECRET."""
    with _patch("add_comment") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.add_comment("DEV-5", "hi")
        real.assert_not_called()


@pytest.mark.parametrize(("issue", "allowed"), [("10001", True), ("10002", False)])
def test_numeric_issue_ids_are_resolved(fetcher, issue, allowed):
    with _patch("get_worklogs", return_value=[]):
        if allowed:
            assert fetcher.get_worklogs(issue) == []
        else:
            with pytest.raises(ProjectAccessDeniedError):
                fetcher.get_worklogs(issue)


@pytest.mark.parametrize("issue", ["SECRET%2D1", " secret-1 ", "​SECRET-1"])
def test_obfuscated_foreign_keys_are_denied(fetcher, issue):
    with _patch("delete_issue"), pytest.raises(ProjectAccessDeniedError):
        fetcher.delete_issue(issue)


@pytest.mark.parametrize("issue", ["DEV-1/../../SECRET-1", "DEV-1?x=1", "nope"])
def test_malformed_issue_references_are_denied(fetcher, issue):
    with _patch("delete_issue"), pytest.raises(ProjectAccessDeniedError):
        fetcher.delete_issue(issue)
    fetcher.jira.issue.assert_not_called()


def test_unresolvable_issue_is_denied(fetcher):
    with _patch("get_issue"), pytest.raises(ProjectAccessDeniedError):
        fetcher.get_issue("DEV-404")


def test_lookups_are_cached(fetcher):
    with _patch("get_worklogs", return_value=[]):
        fetcher.get_worklogs("DEV-1")
        fetcher.get_worklogs("DEV-1")
    assert fetcher.jira.issue.call_count == 1


def test_move_issue_checks_target_project(fetcher):
    with _patch("move_issue") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.move_issue("DEV-1", "SECRET")
        real.assert_not_called()


def test_batch_create_checks_every_payload(fetcher):
    payloads = [{"project_key": "DEV"}, {"project_key": "SECRET"}]
    with _patch("batch_create_issues") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.batch_create_issues(payloads)
        real.assert_not_called()


def test_iterables_are_checked_element_wise(fetcher):
    with _patch("move_issues_to_backlog") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.move_issues_to_backlog(["DEV-1", "SECRET-1"])
        real.assert_not_called()


# --- other references -------------------------------------------------------


@pytest.mark.parametrize(("project", "allowed"), [("1", True), ("2", False)])
def test_numeric_project_ids_are_resolved(fetcher, project, allowed):
    with _patch("create_version", return_value={}):
        if allowed:
            fetcher.create_version(project, "v1")
        else:
            with pytest.raises(ProjectAccessDeniedError):
                fetcher.create_version(project, "v1")


@pytest.mark.parametrize(
    ("method", "args", "allowed"),
    [
        ("get_all_sprints_from_board", ("1",), True),
        ("get_all_sprints_from_board", ("2",), False),
        ("get_all_sprints_from_board", ("3",), False),  # board without project
        ("get_sprint_issues", ("11",), True),
        ("get_sprint_issues", ("12",), False),
        ("get_service_desk_queues", ("5",), True),
        ("get_service_desk_queues", ("6",), False),
        ("update_project_version", ("100",), True),
        ("update_project_version", ("200",), False),
        ("remove_issue_link", ("7",), True),
        ("remove_issue_link", ("8",), False),
        ("get_all_sprints_from_board", ("1 OR 2",), False),
    ],
)
def test_container_references_are_resolved(fetcher, method, args, allowed):
    with _patch(method, return_value="OK"):
        if allowed:
            assert getattr(fetcher, method)(*args) == "OK"
        else:
            with pytest.raises(ProjectAccessDeniedError):
                getattr(fetcher, method)(*args)


def test_create_issue_link_checks_both_sides(fetcher):
    data = {
        "type": {"name": "Blocks"},
        "inwardIssue": {"key": "DEV-1"},
        "outwardIssue": {"key": "SECRET-1"},
    }
    with _patch("create_issue_link") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.create_issue_link(data)
        real.assert_not_called()


def test_assignable_user_search_scoping_args_are_checked(fetcher):
    with _patch("search_assignable_users", return_value=[]):
        assert fetcher.search_assignable_users("max", project_key="DEV") == []
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.search_assignable_users("max", issue_key="SECRET-1")


# --- result filtering -------------------------------------------------------


def _issue(key: str, **kwargs: Any) -> JiraIssue:
    return JiraIssue(key=key, **kwargs)


def test_search_results_drop_foreign_issues_and_references(fetcher):
    dev = _issue(
        "DEV-1",
        issuelinks=[
            JiraIssueLink(outward_issue=JiraLinkedIssue(key="DEV-2")),
            JiraIssueLink(outward_issue=JiraLinkedIssue(key="SECRET-3")),
        ],
        subtasks=[{"key": "DEV-3"}, {"key": "SECRET-4"}],
        parent={"key": "SECRET-9"},
        epic_key="SECRET-1",
        epic_name="Secret epic",
    )
    result = JiraSearchResult(issues=[dev, _issue("SECRET-2")])
    with _patch("search_issues", return_value=result):
        filtered = fetcher.search_issues("text ~ x")

    assert [i.key for i in filtered.issues] == ["DEV-1"]
    issue = filtered.issues[0]
    assert [link.outward_issue.key for link in issue.issuelinks] == ["DEV-2"]
    assert [s["key"] for s in issue.subtasks] == ["DEV-3"]
    assert issue.parent is None
    assert issue.epic_key is None and issue.epic_name is None


def test_project_lists_are_filtered(fetcher):
    with _patch("get_all_projects", return_value=[{"key": "DEV"}, {"key": "SECRET"}]):
        assert fetcher.get_all_projects() == [{"key": "DEV"}]
    with _patch("get_project_keys", return_value=["DEV", "SECRET"]):
        assert fetcher.get_project_keys() == ["DEV"]


def test_boards_are_filtered_by_location(fetcher):
    boards = [
        {"id": 1, "location": {"projectKey": "DEV"}},
        {"id": 2, "location": {"projectKey": "SECRET"}},
        {"id": 3},
    ]
    with _patch("get_all_agile_boards", return_value=boards):
        assert [b["id"] for b in fetcher.get_all_agile_boards()] == [1]


def test_cross_project_dependencies_hide_foreign_projects():
    fetcher = _make_fetcher("DEV,OPS")
    data = {
        "project_key": "DEV",
        "total_cross_project_links": 3,
        "by_project": {
            "OPS": {"total_links": 1, "by_link_type": {}},
            "SECRET": {"total_links": 2, "by_link_type": {}},
        },
    }
    with _patch("get_cross_project_dependencies", return_value=data):
        result = fetcher.get_cross_project_dependencies("DEV")
    assert list(result["by_project"]) == ["OPS"]
    assert result["total_cross_project_links"] == 1


def test_epic_hierarchy_hides_foreign_parents(fetcher):
    data = {
        "project_key": "DEV",
        "groups": [
            {"parent": {"key": "SECRET-1", "summary": "s"}, "epics": [{"key": "DEV-7"}]}
        ],
    }
    with _patch("get_project_epic_hierarchy", return_value=data):
        group = fetcher.get_project_epic_hierarchy("DEV")["groups"][0]
    assert group["parent"] is None
    assert group["epics"] == [{"key": "DEV-7"}]


# --- known gaps (xfail until fixed; remove the xfail marker with the fix) -----

# create_issue and update_issue only check the issue's own project. A parent
# or epic reference in the fields can point into a foreign project, which
# changes that project's hierarchy (a new child appears under its epic).
# Reads are not affected: foreign parents are hidden by _sanitize_issue.
_FOREIGN_REFERENCES = [
    pytest.param({"parent": "SECRET-1"}, id="parent-key"),
    pytest.param({"parent": "DEV-5"}, id="parent-moved-key"),  # now SECRET
    pytest.param({"parent": "10002"}, id="parent-numeric-id"),
    pytest.param({"epic_link": "SECRET-1"}, id="epic-link-alias"),
    pytest.param({"epicKey": "SECRET-1"}, id="epic-key-alias"),
]


@pytest.mark.security_regression
@pytest.mark.parametrize("reference", _FOREIGN_REFERENCES)
def test_create_issue_cannot_reference_foreign_issues(fetcher, reference):
    with _patch("create_issue") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.create_issue("DEV", "summary", "Task", **reference)
        real.assert_not_called()


@pytest.mark.security_regression
@pytest.mark.parametrize(
    ("fields", "kwargs"),
    [
        pytest.param({"parent": {"key": "SECRET-1"}}, {}, id="fields-parent-key"),
        pytest.param({"parent": {"id": "10002"}}, {}, id="fields-parent-id"),
        pytest.param(None, {"parent": "SECRET-1"}, id="kwarg-parent"),
        pytest.param(None, {"epic_link": "SECRET-1"}, id="kwarg-epic-link"),
    ],
)
def test_update_issue_cannot_reference_foreign_issues(fetcher, fields, kwargs):
    with _patch("update_issue") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.update_issue("DEV-1", fields=fields, **kwargs)
        real.assert_not_called()


@pytest.mark.security_regression
def test_batch_create_cannot_reference_foreign_issues(fetcher):
    # batch_create_issues sends jira.create_issues directly, so the
    # create_issue guard never sees these payloads.
    payloads = [{"project_key": "DEV", "summary": "s", "parent": "SECRET-1"}]
    with _patch("batch_create_issues") as real:
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.batch_create_issues(payloads)
        real.assert_not_called()


def test_create_issue_with_allowed_parent_still_works(fetcher):
    # Guards the fix for the tests above against blocking same-project parents.
    with _patch("create_issue", return_value=MagicMock()) as real:
        fetcher.create_issue("DEV", "summary", "Sub-task", parent="DEV-2")
        real.assert_called_once()


_REFERENCE_FIELDS = {
    "epic link": "customfield_10014",
    "parent link": "customfield_10020",
}


@pytest.fixture
def reference_fields():
    with (
        patch.object(
            JiraFetcher, "_generate_field_map", return_value=_REFERENCE_FIELDS
        ),
        patch.object(
            JiraFetcher,
            "get_field_ids_to_epic",
            return_value={"epic_link": "customfield_10014"},
        ),
    ):
        yield


@pytest.mark.parametrize(
    ("extra", "allowed"),
    [
        pytest.param({"customfield_10014": "SECRET-1"}, False, id="epic-link-id"),
        pytest.param(
            {"customfield_10020": {"key": "SECRET-1"}}, False, id="parent-link-id"
        ),
        pytest.param({"Parent Link": "SECRET-1"}, False, id="parent-link-name"),
        pytest.param({"customfield_10014": "DEV-2"}, True, id="epic-link-same-project"),
        pytest.param({"customfield_10030": "SECRET-1"}, True, id="unrelated-field"),
    ],
)
def test_issue_reference_custom_fields_are_checked(
    fetcher, reference_fields, extra, allowed
):
    with _patch("create_issue", return_value=MagicMock()) as real:
        if allowed:
            fetcher.create_issue("DEV", "summary", "Task", **extra)
            real.assert_called_once()
        else:
            with pytest.raises(ProjectAccessDeniedError):
                fetcher.create_issue("DEV", "summary", "Task", **extra)
            real.assert_not_called()


def test_custom_fields_are_denied_when_field_lookup_fails(fetcher):
    with (
        patch.object(
            JiraFetcher, "_generate_field_map", side_effect=RuntimeError("503")
        ),
        _patch("update_issue") as real,
    ):
        with pytest.raises(ProjectAccessDeniedError):
            fetcher.update_issue("DEV-1", fields={"customfield_10014": "DEV-2"})
        real.assert_not_called()


def test_clearing_a_parent_is_allowed(fetcher):
    with _patch("update_issue", return_value=MagicMock()) as real:
        fetcher.update_issue("DEV-1", fields={"parent": None})
        real.assert_called_once()
