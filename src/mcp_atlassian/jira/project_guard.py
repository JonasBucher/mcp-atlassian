"""Hard project allowlist for every Jira operation (fork-specific).

Upstream applies ``JIRA_PROJECTS_FILTER`` to JQL searches and checks the
issue-key prefix in ``get_issue`` only. Every other tool (comments, worklogs,
transitions, attachments, sprints, versions, links, ...) can still reach
other projects. A key prefix is not a reliable boundary either: tools accept
numeric issue IDs, and Jira resolves the old key of a moved issue
(``DEV-5`` may now be ``SECRET-12``).

``ProjectGuardMixin`` sits first in the ``JiraFetcher`` MRO and wraps the
methods listed in ``GUARD_RULES``. Before the real method runs, every issue,
project, board, sprint, service desk, version and issue-link reference in its
arguments is resolved to the project it actually belongs to and checked
against the allowlist. Anything that cannot be resolved is denied (fail
closed). Results are post-processed so that issues and embedded references
(issue links, parent, subtasks, epic) from other projects are dropped.

The guard is a no-op when ``JIRA_PROJECTS_FILTER`` is unset.

Every public method of ``JiraFetcher`` must appear either in ``GUARD_RULES``
or in ``UNGUARDED_METHODS``; a unit test enforces this so that methods added
upstream cannot silently bypass the guard after a merge.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import unquote

from ..models.jira import JiraIssue, JiraSearchResult
from ..utils.scope_guard import install_guards, iter_values
from .client import JiraClient
from .config import normalize_project_key
from .issues import _EPIC_LINK_ALIASES
from .protocols import FieldsOperationsProto

logger = logging.getLogger("mcp-atlassian")

_ISSUE_KEY = re.compile(r"^([A-Z][A-Z0-9_]*)-\d+$")
_NUMERIC_ID = re.compile(r"^\d+$")
_PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Field names (lowercase) whose value points at another issue. Writing one
# attaches the new or updated issue to that issue (as child, epic member or
# Advanced Roadmaps child), which changes the referenced project too.
_ISSUE_REFERENCE_NAMES = frozenset({"parent", "parent link"} | _EPIC_LINK_ALIASES)


class ProjectAccessDeniedError(ValueError):
    """Raised when an operation targets a project outside the allowlist."""


@dataclass(frozen=True)
class GuardRule:
    """Which arguments of a fetcher method must be checked, and how.

    Each field names parameters of the wrapped method. A parameter may hold a
    single value or an iterable of values; ``None`` is skipped. ``result``
    optionally post-filters the return value (issue results are always
    sanitized afterwards).
    """

    issues: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    boards: tuple[str, ...] = ()
    sprints: tuple[str, ...] = ()
    service_desks: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    link_payloads: tuple[str, ...] = ()
    issue_payloads: tuple[str, ...] = ()
    issue_fields: tuple[str, ...] = ()
    result: Callable[[ProjectGuardMixin, Any], Any] | None = None


# --- result filters ---------------------------------------------------------


def _filter_project_dicts(guard: ProjectGuardMixin, projects: Any) -> Any:
    if not isinstance(projects, list):
        return projects
    return [
        p
        for p in projects
        if isinstance(p, dict) and guard._is_project_allowed(str(p.get("key", "")))
    ]


def _filter_project_key_list(guard: ProjectGuardMixin, keys: Any) -> Any:
    if not isinstance(keys, list):
        return keys
    return [k for k in keys if guard._is_project_allowed(str(k))]


def _filter_project_key_dict(guard: ProjectGuardMixin, data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if guard._is_project_allowed(str(k))}


def _filter_boards(guard: ProjectGuardMixin, boards: Any) -> Any:
    if not isinstance(boards, list):
        return boards
    return [
        b
        for b in boards
        if isinstance(b, dict)
        and guard._is_project_allowed(
            str((b.get("location") or {}).get("projectKey", ""))
        )
    ]


def _filter_cross_project_dependencies(guard: ProjectGuardMixin, data: Any) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get("by_project"), dict):
        return data
    by_project = _filter_project_key_dict(guard, data["by_project"])
    total = sum(int(v.get("total_links", 0)) for v in by_project.values())
    return {**data, "by_project": by_project, "total_cross_project_links": total}


def _filter_epic_hierarchy(guard: ProjectGuardMixin, data: Any) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        return data
    groups = []
    for group in data["groups"]:
        parent = group.get("parent") if isinstance(group, dict) else None
        if isinstance(parent, dict) and not guard._is_issue_key_allowed(
            str(parent.get("key", ""))
        ):
            # Keep the (allowed) epics, hide the foreign parent.
            group = {
                **group,
                "parent": None,
                "group_name": "Parent outside allowed projects",
            }
        groups.append(group)
    return {**data, "groups": groups}


# --- rule table -------------------------------------------------------------

_ISSUE = GuardRule(issues=("issue_key",))

GUARD_RULES: dict[str, GuardRule] = {
    # issues
    "get_issue": _ISSUE,
    "create_issue": GuardRule(projects=("project_key",), issue_fields=("kwargs",)),
    "update_issue": GuardRule(issues=("issue_key",), issue_fields=("fields", "kwargs")),
    "assign_issue": _ISSUE,
    "delete_issue": _ISSUE,
    "move_issue": GuardRule(issues=("issue_key",), projects=("target_project_key",)),
    "transition_issue": _ISSUE,
    "batch_create_issues": GuardRule(issue_payloads=("issues",)),
    "batch_get_changelogs": GuardRule(issues=("issue_ids_or_keys",)),
    "get_available_transitions": _ISSUE,
    "get_transitions": _ISSUE,
    "get_transitions_models": _ISSUE,
    # comments / worklogs / watchers
    "get_issue_comments": _ISSUE,
    "add_comment": _ISSUE,
    "edit_comment": _ISSUE,
    "add_worklog": _ISSUE,
    "get_worklog": _ISSUE,
    "get_worklog_models": _ISSUE,
    "get_worklogs": _ISSUE,
    "get_issue_watchers": _ISSUE,
    "add_watcher": _ISSUE,
    "remove_watcher": _ISSUE,
    # attachments
    "get_issue_attachments": _ISSUE,
    "get_issue_attachment_contents": _ISSUE,
    "download_issue_attachments": _ISSUE,
    "upload_attachment": _ISSUE,
    "upload_attachments": _ISSUE,
    "upload_attachment_from_content": _ISSUE,
    "upload_attachments_from_content": _ISSUE,
    # forms
    "get_issue_forms": _ISSUE,
    "get_form_details": _ISSUE,
    "update_form_answers": _ISSUE,
    "add_form_template": _ISSUE,
    "delete_form": _ISSUE,
    "get_form_attachments": _ISSUE,
    # links
    "create_issue_link": GuardRule(link_payloads=("data",)),
    "create_remote_issue_link": _ISSUE,
    "get_remote_issue_links": _ISSUE,
    "remove_issue_link": GuardRule(links=("link_id",)),
    # epics
    "prepare_epic_fields": GuardRule(projects=("project_key",)),
    "link_issue_to_epic": GuardRule(issues=("issue_key", "epic_key")),
    "get_epic_issues": GuardRule(issues=("epic_key",)),
    "update_epic_fields": _ISSUE,
    # metrics / SLA / development
    "get_issue_dates": _ISSUE,
    "batch_get_issue_dates": GuardRule(issues=("issue_keys",)),
    "get_issue_sla": _ISSUE,
    "batch_get_issue_sla": GuardRule(issues=("issue_keys",)),
    "get_issue_development_info": _ISSUE,
    "get_issues_development_info": GuardRule(issues=("issue_keys",)),
    # projects
    "get_all_projects": GuardRule(result=_filter_project_dicts),
    "search_projects": GuardRule(result=_filter_project_dicts),
    "get_user_accessible_projects": GuardRule(result=_filter_project_dicts),
    "get_project_keys": GuardRule(result=_filter_project_key_list),
    "get_project_leads": GuardRule(result=_filter_project_key_dict),
    "get_project": GuardRule(projects=("project_key",)),
    "get_project_model": GuardRule(projects=("project_key",)),
    "project_exists": GuardRule(projects=("project_key",)),
    "get_project_components": GuardRule(projects=("project_key",)),
    "get_project_versions": GuardRule(projects=("project_key",)),
    "get_project_roles": GuardRule(projects=("project_key",)),
    "get_project_role_members": GuardRule(projects=("project_key",)),
    "get_project_permission_scheme": GuardRule(projects=("project_key",)),
    "get_project_notification_scheme": GuardRule(projects=("project_key",)),
    "get_project_issue_types": GuardRule(projects=("project_key",)),
    "get_create_fields": GuardRule(projects=("project_key",)),
    "get_project_fields": GuardRule(projects=("project_key",)),
    "get_project_issues_count": GuardRule(projects=("project_key",)),
    "get_project_issues": GuardRule(projects=("project_key",)),
    "get_project_epic_hierarchy": GuardRule(
        projects=("project_key",), result=_filter_epic_hierarchy
    ),
    "get_cross_project_dependencies": GuardRule(
        projects=("project_key",), result=_filter_cross_project_dependencies
    ),
    "get_required_fields": GuardRule(projects=("project_key",)),
    "get_field_options": GuardRule(projects=("project_key",)),
    # versions
    "create_version": GuardRule(projects=("project",)),
    "create_project_version": GuardRule(projects=("project_key",)),
    "update_version": GuardRule(versions=("version_id",)),
    "update_project_version": GuardRule(versions=("version_id",)),
    # search (JQL is already filtered upstream; results are sanitized)
    "search_issues": GuardRule(),
    "get_board_issues": GuardRule(),
    "get_sprint_issues": GuardRule(sprints=("sprint_id",)),
    # agile
    "get_all_agile_boards": GuardRule(projects=("project_key",), result=_filter_boards),
    "get_all_agile_boards_model": GuardRule(projects=("project_key",)),
    "get_all_sprints_from_board": GuardRule(boards=("board_id",)),
    "get_all_sprints_from_board_model": GuardRule(boards=("board_id",)),
    "create_sprint": GuardRule(boards=("board_id",)),
    "update_sprint": GuardRule(sprints=("sprint_id",)),
    "add_issues_to_sprint": GuardRule(sprints=("sprint_id",), issues=("issue_keys",)),
    "move_issues_to_backlog": GuardRule(issues=("issue_keys",)),
    # service management
    "get_service_desk_for_project": GuardRule(projects=("project_key",)),
    "get_service_desk_queues": GuardRule(service_desks=("service_desk_id",)),
    "get_queue_issues": GuardRule(service_desks=("service_desk_id",)),
    "get_request_types": GuardRule(service_desks=("service_desk_id",)),
    "get_request_type_fields": GuardRule(service_desks=("service_desk_id",)),
    "attach_temporary_files": GuardRule(service_desks=("service_desk_id",)),
    "create_customer_request": GuardRule(service_desks=("service_desk_id",)),
    # users: only the scoping arguments are checked here. The user-search
    # tools themselves are blocked at tool level (see servers/main.py),
    # because these methods are also used internally to resolve assignees.
    "search_assignable_users": GuardRule(
        projects=("project_key",), issues=("issue_key",)
    ),
}

# Public fetcher methods that deliberately stay unguarded.
UNGUARDED_METHODS: frozenset[str] = frozenset(
    {
        # global, non-project metadata
        "get_fields",
        "get_field_id",
        "get_field_by_id",
        "get_custom_fields",
        "get_field_ids_to_epic",
        "is_custom_field",
        "format_field_value",
        "search_fields",
        "get_field_contexts",
        "get_issue_link_types",
        # identity of the configured user (needed for auth validation)
        "get_current_user_account_id",
        # user lookup by identifier; blocked at tool level, used internally
        "get_user_profile_by_identifier",
        # URL-based downloads: tools only pass URLs obtained from the
        # guarded get_issue_attachments / get_issue_attachment_contents
        "fetch_attachment_content",
        "download_attachment",
        # generic internal pagination helper, not called by tools
        "get_paged",
        # pure formatting/transformation helpers (no API access of their own;
        # extract_epic_information's internal get_issue call is guarded)
        "markdown_to_jira",
        "format_issue_content",
        "create_issue_metadata",
        "extract_epic_information",
        "sanitize_html",
        "sanitize_transition_fields",
        "add_comment_to_transition_data",
    }
)


class ProjectGuardMixin(JiraClient):
    """Enforces ``config.projects_filter`` on every guarded fetcher method."""

    # --- allowlist ----------------------------------------------------------

    @property
    def _project_allowlist(self) -> frozenset[str] | None:
        raw = self.config.projects_filter
        if not raw:
            return None
        keys = frozenset(normalize_project_key(k) for k in raw.split(","))
        keys = frozenset(k for k in keys if k)
        return keys or None

    def _is_project_allowed(self, project_key: str) -> bool:
        allowlist = self._project_allowlist
        if allowlist is None:
            return True
        return normalize_project_key(project_key) in allowlist

    def _is_issue_key_allowed(self, issue_key: str) -> bool:
        """Prefix check for keys taken from API responses (current keys)."""
        match = _ISSUE_KEY.match(normalize_project_key(issue_key))
        if match is None:
            return False
        return self._is_project_allowed(match.group(1))

    @staticmethod
    def _deny(kind: str, value: str) -> ProjectAccessDeniedError:
        msg = (
            f"{kind} '{value}' is outside the projects allowed by JIRA_PROJECTS_FILTER"
        )
        return ProjectAccessDeniedError(msg)

    # --- resolution ---------------------------------------------------------

    @property
    def _project_guard_cache(self) -> dict[str, str]:
        cache: dict[str, str] | None = self.__dict__.get("_project_guard_cache_store")
        if cache is None:
            cache = {}
            self.__dict__["_project_guard_cache_store"] = cache
        return cache

    def _cached(self, cache_key: str, resolve: Callable[[], str]) -> str:
        cache = self._project_guard_cache
        if cache_key not in cache:
            try:
                value = resolve()
            except ProjectAccessDeniedError:
                raise
            except Exception as e:  # noqa: BLE001 - any lookup failure denies access
                logger.warning(f"Project guard could not resolve {cache_key}: {e}")
                value = ""
            if not value:
                msg = f"Could not verify the Jira project of {cache_key}"
                raise ProjectAccessDeniedError(msg)
            cache[cache_key] = normalize_project_key(value)
        return cache[cache_key]

    @staticmethod
    def _numeric(kind: str, value: str) -> str:
        cleaned = unquote(str(value)).strip()
        if not _NUMERIC_ID.match(cleaned):
            msg = f"Invalid Jira {kind} ID '{value}'"
            raise ProjectAccessDeniedError(msg)
        return cleaned

    def _project_of_issue(self, issue: str) -> str:
        cleaned = normalize_project_key(unquote(str(issue)))
        if match := _ISSUE_KEY.match(cleaned):
            # Cheap pre-check; the API lookup below still runs because a
            # moved issue's old key resolves to its new project.
            if not self._is_project_allowed(match.group(1)):
                raise self._deny("Issue", issue)
        elif not _NUMERIC_ID.match(cleaned):
            msg = f"Invalid Jira issue key or ID '{issue}'"
            raise ProjectAccessDeniedError(msg)

        def resolve() -> str:
            data = self.jira.issue(cleaned, fields="project")
            return str(
                ((data or {}).get("fields") or {}).get("project", {}).get("key", "")
            )

        return self._cached(f"issue {cleaned}", resolve)

    def _project_key(self, project: str) -> str:
        cleaned = normalize_project_key(unquote(str(project)))
        if _PROJECT_KEY.match(cleaned):
            return cleaned
        pid = self._numeric("project", project)
        return self._cached(
            f"project {pid}",
            lambda: str((self.jira.get_project(pid) or {}).get("key", "")),
        )

    def _project_of_board(self, board_id: str) -> str:
        bid = self._numeric("board", board_id)
        return self._cached(
            f"board {bid}",
            lambda: str(
                ((self.jira.get_agile_board(bid) or {}).get("location") or {}).get(
                    "projectKey", ""
                )
            ),
        )

    def _project_of_sprint(self, sprint_id: str) -> str:
        sid = self._numeric("sprint", sprint_id)

        def resolve() -> str:
            board = (self.jira.get_sprint(sid) or {}).get("originBoardId")
            return self._project_of_board(str(board)) if board else ""

        return self._cached(f"sprint {sid}", resolve)

    def _project_of_service_desk(self, service_desk_id: str) -> str:
        sid = self._numeric("service desk", service_desk_id)
        return self._cached(
            f"service desk {sid}",
            lambda: str(
                (self.jira.get(f"rest/servicedeskapi/servicedesk/{sid}") or {}).get(
                    "projectKey", ""
                )
            ),
        )

    def _project_of_version(self, version_id: str) -> str:
        vid = self._numeric("version", version_id)

        def resolve() -> str:
            project_id = (self.jira.get_version(vid) or {}).get("projectId")
            return self._project_key(str(project_id)) if project_id else ""

        return self._cached(f"version {vid}", resolve)

    # --- checks -------------------------------------------------------------

    def _assert_issue_allowed(self, issue: str) -> None:
        if not self._is_project_allowed(self._project_of_issue(issue)):
            raise self._deny("Issue", issue)

    def _assert_project_allowed(self, project: str) -> None:
        if not self._is_project_allowed(self._project_key(project)):
            raise self._deny("Project", project)

    def _assert_board_allowed(self, board_id: str) -> None:
        if not self._is_project_allowed(self._project_of_board(board_id)):
            raise self._deny("Board", board_id)

    def _assert_sprint_allowed(self, sprint_id: str) -> None:
        if not self._is_project_allowed(self._project_of_sprint(sprint_id)):
            raise self._deny("Sprint", sprint_id)

    def _assert_service_desk_allowed(self, service_desk_id: str) -> None:
        if not self._is_project_allowed(self._project_of_service_desk(service_desk_id)):
            raise self._deny("Service desk", service_desk_id)

    def _assert_version_allowed(self, version_id: str) -> None:
        if not self._is_project_allowed(self._project_of_version(version_id)):
            raise self._deny("Version", version_id)

    def _assert_link_allowed(self, link_id: str) -> None:
        lid = self._numeric("issue link", link_id)
        try:
            link = self.jira.get_issue_link(lid) or {}
        except Exception as e:  # noqa: BLE001 - any lookup failure denies access
            logger.warning(f"Project guard could not resolve issue link {lid}: {e}")
            link = {}
        self._assert_link_payload_allowed(link)

    def _assert_link_payload_allowed(self, data: Any) -> None:
        if not isinstance(data, dict):
            msg = "Could not verify the issues of the Jira issue link"
            raise ProjectAccessDeniedError(msg)
        for side in ("inwardIssue", "outwardIssue"):
            key = (data.get(side) or {}).get("key")
            if not key:
                msg = "Could not verify the issues of the Jira issue link"
                raise ProjectAccessDeniedError(msg)
            self._assert_issue_allowed(str(key))

    def _assert_issue_payloads_allowed(self, payloads: Any) -> None:
        for payload in payloads or ():
            project = payload.get("project_key") if isinstance(payload, dict) else None
            if not project:
                msg = "Could not verify the project of an issue to create"
                raise ProjectAccessDeniedError(msg)
            self._assert_project_allowed(str(project))
            self._assert_issue_fields_allowed(payload)

    def _reference_field_ids(self) -> frozenset[str]:
        """Custom field IDs that hold issue references (epic and parent link)."""
        cached: frozenset[str] | None = self.__dict__.get("_reference_field_ids_store")
        if cached is not None:
            return cached
        # The field helpers come from FieldsMixin, composed into JiraFetcher.
        fields = cast(FieldsOperationsProto, self)
        try:
            field_map = fields._generate_field_map()
            ids = {
                fields.get_field_ids_to_epic().get("epic_link"),
                field_map.get("epic link"),
                field_map.get("parent link"),
            }
        except Exception as e:  # noqa: BLE001 - any lookup failure denies access
            logger.warning(f"Project guard could not load Jira fields: {e}")
            msg = "Could not verify which custom fields reference issues"
            raise ProjectAccessDeniedError(msg) from e
        resolved = frozenset(i.lower() for i in ids if i)
        self.__dict__["_reference_field_ids_store"] = resolved
        return resolved

    def _is_issue_reference_field(self, name: str) -> bool:
        lowered = name.strip().lower()
        if lowered in _ISSUE_REFERENCE_NAMES:
            return True
        # Custom field IDs are only resolved when present, so payloads without
        # them never need the field lookup.
        return (
            lowered.startswith("customfield_")
            and lowered in self._reference_field_ids()
        )

    @staticmethod
    def _referenced_issues(value: Any) -> tuple[str, ...]:
        """Issue keys/IDs in a reference field value; empty when clearing."""
        if value is None or value == "":
            return ()
        if isinstance(value, dict):
            ref = value.get("key") or value.get("id")
            if ref in (None, ""):
                msg = "Could not verify the issue referenced by a field"
                raise ProjectAccessDeniedError(msg)
            return (str(ref),)
        if isinstance(value, str | int):
            return (str(value),)
        msg = "Could not verify the issue referenced by a field"
        raise ProjectAccessDeniedError(msg)

    def _assert_issue_fields_allowed(self, fields: Any) -> None:
        """Deny writes whose fields point at issues in other projects."""
        if not isinstance(fields, dict):
            return
        for name, value in fields.items():
            if self._is_issue_reference_field(str(name)):
                for issue in self._referenced_issues(value):
                    self._assert_issue_allowed(issue)

    # --- enforcement --------------------------------------------------------

    def _scope_guard_active(self) -> bool:
        return self._project_allowlist is not None

    def _enforce_rule(self, rule: GuardRule, arguments: dict[str, Any]) -> None:
        checks: tuple[tuple[tuple[str, ...], Callable[[str], None]], ...] = (
            (rule.issues, self._assert_issue_allowed),
            (rule.projects, self._assert_project_allowed),
            (rule.boards, self._assert_board_allowed),
            (rule.sprints, self._assert_sprint_allowed),
            (rule.service_desks, self._assert_service_desk_allowed),
            (rule.versions, self._assert_version_allowed),
            (rule.links, self._assert_link_allowed),
        )
        for params, check in checks:
            for param in params:
                for value in iter_values(arguments.get(param)):
                    check(value)
        for param in rule.link_payloads:
            self._assert_link_payload_allowed(arguments.get(param))
        for param in rule.issue_payloads:
            self._assert_issue_payloads_allowed(arguments.get(param))
        for param in rule.issue_fields:
            self._assert_issue_fields_allowed(arguments.get(param))

    def _guard_result(self, rule: GuardRule, result: Any) -> Any:
        if rule.result:
            result = rule.result(self, result)
        return self._sanitize(result)

    def _sanitize(self, result: Any) -> Any:
        """Drop foreign issues and foreign references embedded in issues."""
        if isinstance(result, JiraSearchResult):
            result.issues = [
                self._sanitize_issue(i)
                for i in result.issues
                if self._is_issue_key_allowed(i.key)
            ]
        elif isinstance(result, JiraIssue):
            self._sanitize_issue(result)
        elif isinstance(result, list) and any(isinstance(i, JiraIssue) for i in result):
            return [
                self._sanitize_issue(i) if isinstance(i, JiraIssue) else i
                for i in result
                if not isinstance(i, JiraIssue) or self._is_issue_key_allowed(i.key)
            ]
        return result

    def _sanitize_issue(self, issue: JiraIssue) -> JiraIssue:
        issue.issuelinks = [
            link
            for link in issue.issuelinks
            if all(
                self._is_issue_key_allowed(linked.key)
                for linked in (link.inward_issue, link.outward_issue)
                if linked is not None
            )
        ]
        issue.subtasks = [
            s
            for s in issue.subtasks
            if self._is_issue_key_allowed(str((s or {}).get("key", "")))
        ]
        if issue.parent and not self._is_issue_key_allowed(
            str(issue.parent.get("key", ""))
        ):
            issue.parent = None
        if issue.epic_key and not self._is_issue_key_allowed(issue.epic_key):
            issue.epic_key = None
            issue.epic_name = None
        return issue


install_guards(ProjectGuardMixin, GUARD_RULES)
