"""Live probes for gaps in the fork-specific space/project guards.

The unit tests in tests/unit/confluence/test_space_guard.py and
tests/unit/jira/test_project_guard.py pin the expected behaviour. These tests
answer a different question: can each gap actually be exploited on a real
site? A failure here confirms the leak; a pass means the guard (or the site)
stopped it.

They are skipped unless run with ``--integration --use-real-data`` and the
environment below is set. They write to the allowed space/project and clean
up after themselves. They never write to the foreign space or project.

Required environment, in addition to the usual JIRA_* / CONFLUENCE_* auth:

- ``CONFLUENCE_SPACES_FILTER``: the allowed space(s).
- ``CONFLUENCE_TEST_PAGE_ID``: a page in an allowed space; probe comments are
  added to it and deleted again.
- ``SCOPE_GUARD_FOREIGN_SPACE_KEY``: a space the account can read but that is
  not in the allowlist.
- ``SCOPE_GUARD_FOREIGN_PAGE_TITLE``: a page in that space.
- ``SCOPE_GUARD_FOREIGN_MARKER``: a unique word that appears in that page's
  body and nowhere in the allowed space.
- ``JIRA_PROJECTS_FILTER``: the allowed project(s).
- ``JIRA_TEST_PROJECT_KEY``: an allowed project to create a probe issue in.
- ``SCOPE_GUARD_FOREIGN_ISSUE_KEY``: an epic in a project the account can
  read but that is not in the allowlist.
- ``SCOPE_GUARD_FOREIGN_ISSUE_MARKER`` (optional): a unique word in that
  issue's summary; enables the Jira macro probe.
"""

import os
import uuid
from collections.abc import Callable, Iterator

import pytest

from mcp_atlassian.confluence import ConfluenceFetcher
from mcp_atlassian.confluence.config import ConfluenceConfig
from mcp_atlassian.jira import JiraFetcher
from mcp_atlassian.jira.config import JiraConfig
from mcp_atlassian.jira.project_guard import ProjectAccessDeniedError


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"{name} not set")
    return value


@pytest.fixture(autouse=True)
def _real_data_only(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--use-real-data", default=False):
        pytest.skip("Real API tests only run with --use-real-data flag")


@pytest.fixture
def confluence() -> ConfluenceFetcher:
    _env("CONFLUENCE_URL")
    _env("CONFLUENCE_SPACES_FILTER")
    return ConfluenceFetcher(config=ConfluenceConfig.from_env())


@pytest.fixture
def jira() -> JiraFetcher:
    _env("JIRA_URL")
    _env("JIRA_PROJECTS_FILTER")
    return JiraFetcher(config=JiraConfig.from_env())


def _allowed_spaces() -> set[str]:
    raw = _env("CONFLUENCE_SPACES_FILTER")
    return {k.strip().upper() for k in raw.split(",") if k.strip()}


# --- gap 1: rendered comments ---------------------------------------------------


def _include_macro() -> tuple[str, str]:
    space = _env("SCOPE_GUARD_FOREIGN_SPACE_KEY")
    title = _env("SCOPE_GUARD_FOREIGN_PAGE_TITLE")
    marker = _env("SCOPE_GUARD_FOREIGN_MARKER")
    storage = (
        '<ac:structured-macro ac:name="include"><ac:parameter ac:name="">'
        f'<ac:link><ri:page ri:space-key="{space}" ri:content-title="{title}"/>'
        "</ac:link></ac:parameter></ac:structured-macro>"
    )
    return storage, marker


def _jira_macro() -> tuple[str, str]:
    key = _env("SCOPE_GUARD_FOREIGN_ISSUE_KEY")
    marker = _env("SCOPE_GUARD_FOREIGN_ISSUE_MARKER")
    storage = (
        '<ac:structured-macro ac:name="jira">'
        f'<ac:parameter ac:name="key">{key}</ac:parameter>'
        "</ac:structured-macro>"
    )
    return storage, marker


@pytest.fixture
def probe_comment_ids(confluence: ConfluenceFetcher) -> Iterator[list[str]]:
    ids: list[str] = []
    yield ids
    for comment_id in ids:
        confluence.confluence.remove_content(comment_id)


@pytest.mark.integration
@pytest.mark.parametrize(
    "macro", [_include_macro, _jira_macro], ids=["include", "jira"]
)
def test_comment_macro_does_not_render_foreign_content(
    confluence: ConfluenceFetcher,
    probe_comment_ids: list[str],
    macro: Callable[[], tuple[str, str]],
) -> None:
    page_id = _env("CONFLUENCE_TEST_PAGE_ID")
    storage, marker = macro()

    comment = confluence.add_comment(page_id, f"<p>scope guard probe</p>{storage}")
    assert comment is not None, "probe comment could not be created"
    probe_comment_ids.append(comment.id)

    bodies = [c.body for c in confluence.get_page_comments(page_id)]
    leaked = [b for b in bodies if marker in b]
    assert not leaked, "foreign content was rendered into a comment"


# --- gap 2: CQL escaping the allowlist -------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "template",
    [
        'type = page) OR space = "{space}" OR (type = page',
        'text ~ "{marker}") OR (text ~ "{marker}"',
    ],
    ids=["unbalanced", "closing-first"],
)
def test_search_query_cannot_escape_the_space_allowlist(
    confluence: ConfluenceFetcher, template: str
) -> None:
    cql = template.format(
        space=_env("SCOPE_GUARD_FOREIGN_SPACE_KEY"),
        marker=_env("SCOPE_GUARD_FOREIGN_MARKER"),
    )
    allowed = _allowed_spaces()

    try:
        results = confluence.search(cql, limit=50)
    except Exception:  # noqa: BLE001 - refusing the query is a valid outcome
        return

    # Resolve each result's space with the raw client: search results do not
    # carry it reliably, which is part of the gap.
    foreign = []
    for page in results:
        data = confluence.confluence.get_page_by_id(page.id, expand="space")
        key = str((data.get("space") or {}).get("key", "")).upper()
        if key not in allowed:
            foreign.append((page.id, key))
    assert not foreign, f"search returned content from foreign spaces: {foreign}"


# --- gap 3: cross-project parent on write ----------------------------------------


@pytest.mark.integration
def test_create_issue_with_foreign_parent_is_refused(jira: JiraFetcher) -> None:
    project = _env("JIRA_TEST_PROJECT_KEY")
    foreign_epic = _env("SCOPE_GUARD_FOREIGN_ISSUE_KEY")
    created = None
    # If Jira itself rejects the cross-project parent, this fails with Jira's
    # error instead: the guard still missed it, but the site blocked the write.
    try:
        with pytest.raises(ProjectAccessDeniedError):
            created = jira.create_issue(
                project,
                f"scope guard probe {uuid.uuid4().hex[:8]}",
                "Task",
                parent=foreign_epic,
            )
    finally:
        if created is not None:
            jira.delete_issue(created.key)
