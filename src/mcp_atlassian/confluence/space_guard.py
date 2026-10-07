"""Hard space allowlist for every Confluence operation (fork-specific).

Upstream applies ``CONFLUENCE_SPACES_FILTER`` only to CQL searches, so any tool
that takes a page, comment or attachment ID (``get_page``, ``update_page``,
``download_attachment``, ...) can still reach content in other spaces. This
module closes that gap at the fetcher level: ``SpaceGuardMixin`` sits first in
the ``ConfluenceFetcher`` MRO and wraps the operations listed in
``GUARD_RULES``. Before the real method runs, every content ID, space key,
space ID and download URL in its arguments is resolved to a space key and
checked against the allowlist. Anything that cannot be resolved is denied
(fail closed).

The guard is a no-op when ``CONFLUENCE_SPACES_FILTER`` is unset.

Every public method of ``ConfluenceFetcher`` must appear either in
``GUARD_RULES`` or in ``UNGUARDED_METHODS``; a unit test enforces this so that
methods added upstream cannot silently bypass the guard after a merge.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..utils.scope_guard import install_guards, iter_values
from .client import ConfluenceClient
from .v2_adapter import ConfluenceV2Adapter

logger = logging.getLogger("mcp-atlassian")

_NUMERIC_ID = re.compile(r"^\d+$")
_DOWNLOAD_URL_CONTENT_ID = (
    # Legacy link: /download/attachments/{content_id}/{filename}
    re.compile(r"/download/(?:attachments|thumbnails)/(\d+)/"),
    # v1 endpoint: /rest/api/content/{content_id}/child/attachment/{id}/download
    re.compile(r"/rest/api/content/(\d+)/child/attachment/"),
)


class SpaceAccessDeniedError(ValueError):
    """Raised when an operation targets content outside the allowed spaces."""


@dataclass(frozen=True)
class GuardRule:
    """Which arguments of a fetcher method must be checked, and how.

    Each field names parameters of the wrapped method. A parameter may hold a
    single value or an iterable of values (e.g. ``page_ids``); ``None`` is
    skipped. ``result`` optionally post-filters or validates the return value.
    """

    content_ids: tuple[str, ...] = ()
    space_keys: tuple[str, ...] = ()
    space_ids: tuple[str, ...] = ()
    download_urls: tuple[str, ...] = ()
    result: Callable[[SpaceGuardMixin, Any], Any] | None = None


# --- result filters ---------------------------------------------------------


def _filter_search_results(guard: SpaceGuardMixin, pages: list[Any]) -> list[Any]:
    # CQL already ANDs the allowlist (see SearchMixin.search); this drops any
    # result whose space is known and foreign as a second line of defence.
    # Results without a resolvable space key are kept because the CQL
    # restriction is authoritative for them.
    kept = []
    for page in pages:
        key = page.space.key if getattr(page, "space", None) else ""
        if key and not guard._is_space_allowed(key):
            logger.warning(f"Space guard dropped search result {page.id} ({key})")
            continue
        kept.append(page)
    return kept


def _filter_spaces_response(guard: SpaceGuardMixin, data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        results = [
            s
            for s in data["results"]
            if isinstance(s, dict) and guard._is_space_allowed(s.get("key", ""))
        ]
        return {**data, "results": results, "size": len(results)}
    return data


def _filter_space_key_dict(guard: SpaceGuardMixin, data: Any) -> Any:
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if guard._is_space_allowed(k)}
    return data


def _check_template_space(guard: SpaceGuardMixin, template: Any) -> Any:
    # Global templates carry no space and are allowed; space templates must
    # belong to an allowed space.
    space = template.get("space") if isinstance(template, dict) else None
    if isinstance(space, dict) and space.get("key"):
        guard._assert_space_allowed(space["key"])
    return template


# --- rule table -------------------------------------------------------------

GUARD_RULES: dict[str, GuardRule] = {
    # pages
    "get_page_content": GuardRule(content_ids=("page_id",)),
    "get_page_ancestors": GuardRule(content_ids=("page_id",)),
    "get_page_by_title": GuardRule(space_keys=("space_key",)),
    "get_space_pages": GuardRule(space_keys=("space_key",)),
    "create_page": GuardRule(space_keys=("space_key",), content_ids=("parent_id",)),
    "update_page": GuardRule(content_ids=("page_id", "parent_id")),
    "update_page_section": GuardRule(content_ids=("page_id",)),
    "get_page_children": GuardRule(content_ids=("page_id",)),
    "get_space_page_tree": GuardRule(space_keys=("space_key",)),
    "delete_page": GuardRule(content_ids=("page_id",)),
    "get_page_history": GuardRule(content_ids=("page_id",)),
    "move_page": GuardRule(
        content_ids=("page_id", "target_parent_id"),
        space_keys=("target_space_key",),
    ),
    "get_page_version_diff": GuardRule(content_ids=("page_id",)),
    "copy_page": GuardRule(
        content_ids=("source_page_id", "destination_parent_id"),
        space_keys=("destination_space_key",),
    ),
    # comments
    "get_page_comments": GuardRule(content_ids=("page_id",)),
    "add_comment": GuardRule(content_ids=("page_id",)),
    "reply_to_comment": GuardRule(content_ids=("comment_id",)),
    "get_inline_comments": GuardRule(content_ids=("page_id",)),
    "add_inline_comment": GuardRule(content_ids=("page_id",)),
    # labels
    "get_page_labels": GuardRule(content_ids=("page_id",)),
    "add_page_label": GuardRule(content_ids=("page_id",)),
    # attachments
    "upload_attachment": GuardRule(content_ids=("content_id",)),
    "upload_attachments": GuardRule(content_ids=("content_id",)),
    "upload_attachment_from_content": GuardRule(content_ids=("content_id",)),
    "get_content_attachments": GuardRule(content_ids=("content_id",)),
    "download_content_attachments": GuardRule(content_ids=("content_id",)),
    "delete_attachment": GuardRule(content_ids=("attachment_id",)),
    # The download_attachment tool fetches metadata outside the fetcher, but
    # every download path resolves its URL here and then fetches the bytes
    # through one of the two URL-based methods below.
    "_resolve_attachment_download_url": GuardRule(
        content_ids=("attachment_id", "content_id"),
        download_urls=("download_url",),
    ),
    "fetch_attachment_content": GuardRule(download_urls=("url",)),
    "download_attachment": GuardRule(download_urls=("url",)),
    # restrictions / permissions / analytics
    "get_page_restrictions": GuardRule(content_ids=("page_id",)),
    "set_page_restrictions": GuardRule(content_ids=("page_id",)),
    "check_content_permissions": GuardRule(content_ids=("content_id",)),
    "get_space_permissions": GuardRule(space_ids=("space_id",)),
    "get_page_views": GuardRule(content_ids=("page_id",)),
    "batch_get_page_views": GuardRule(content_ids=("page_ids",)),
    # templates
    "list_page_templates": GuardRule(space_keys=("space_key",)),
    "get_page_template": GuardRule(result=_check_template_space),
    "create_page_from_template": GuardRule(
        space_keys=("space_key",), content_ids=("parent_id",)
    ),
    # search / spaces (result filtering)
    "search": GuardRule(result=_filter_search_results),
    "get_spaces": GuardRule(result=_filter_spaces_response),
    "get_user_contributed_spaces": GuardRule(result=_filter_space_key_dict),
}

# Public fetcher methods that deliberately stay unguarded because they do not
# touch space-scoped content. The user-search tool is blocked at tool level
# instead (servers/main.py), because user lookups cannot be scoped to a space.
UNGUARDED_METHODS: frozenset[str] = frozenset(
    {
        "search_user",
        "get_user_details_by_accountid",
        "get_user_details_by_username",
        "get_user_details_by_userkey",
        "get_current_user_info",
    }
)


class SpaceGuardMixin(ConfluenceClient):
    """Enforces ``config.spaces_filter`` on every guarded fetcher method."""

    # --- allowlist ----------------------------------------------------------

    @property
    def _space_allowlist(self) -> frozenset[str] | None:
        raw = self.config.spaces_filter
        if not raw:
            return None
        keys = frozenset(k.strip().upper() for k in raw.split(",") if k.strip())
        return keys or None

    def _is_space_allowed(self, space_key: str) -> bool:
        allowlist = self._space_allowlist
        if allowlist is None:
            return True
        return bool(space_key) and space_key.strip().upper() in allowlist

    def _assert_space_allowed(self, space_key: str) -> None:
        if not self._is_space_allowed(space_key):
            msg = (
                f"Confluence space '{space_key}' is not allowed by "
                "CONFLUENCE_SPACES_FILTER"
            )
            raise SpaceAccessDeniedError(msg)

    # --- resolution ---------------------------------------------------------

    @property
    def _space_guard_cache(self) -> dict[str, str]:
        cache: dict[str, str] | None = self.__dict__.get("_space_guard_cache_store")
        if cache is None:
            cache = {}
            self.__dict__["_space_guard_cache_store"] = cache
        return cache

    def _space_key_for_content(self, content_id: str) -> str:
        """Resolve any content ID (page, blog post, comment, attachment)."""
        cid = str(content_id).strip()
        if cid.lower().startswith("att"):
            cid = cid[3:]
        # Rejecting non-numeric IDs also blocks path tricks such as
        # "123/../../space/OTHER" in methods that interpolate IDs into URLs.
        if not _NUMERIC_ID.match(cid):
            msg = f"Invalid Confluence content ID '{content_id}'"
            raise SpaceAccessDeniedError(msg)

        cache = self._space_guard_cache
        if cid in cache:
            return cache[cid]

        try:
            data = self.confluence.get(
                f"{self._v1_rest_base_url()}/rest/api/content/{cid}",
                params={"expand": "space,container"},
                absolute=True,
            )
        except Exception as e:  # noqa: BLE001 - any lookup failure denies access
            logger.warning(f"Space guard could not resolve content {cid}: {e}")
            data = None

        key = ""
        if isinstance(data, dict):
            key = (data.get("space") or {}).get("key", "")
            if not key:
                # Some content (e.g. attachments on Server/DC) only exposes
                # its container; resolve the container one level up.
                container_id = str((data.get("container") or {}).get("id", ""))
                if container_id and container_id != cid:
                    return self._space_key_for_content(container_id)

        if not key:
            msg = f"Could not verify the space of Confluence content '{content_id}'"
            raise SpaceAccessDeniedError(msg)
        cache[cid] = key
        return key

    def _space_key_for_space_id(self, space_id: str) -> str:
        sid = str(space_id).strip()
        if not _NUMERIC_ID.match(sid):
            msg = f"Invalid Confluence space ID '{space_id}'"
            raise SpaceAccessDeniedError(msg)
        adapter = ConfluenceV2Adapter(
            session=self.confluence._session, base_url=self.confluence.url
        )
        # _get_space_key_from_id falls back to returning the ID itself on
        # errors, which never matches an allowlisted key -> denied.
        return adapter._get_space_key_from_id(sid)

    def _assert_content_allowed(self, content_id: str) -> None:
        key = self._space_key_for_content(content_id)
        if not self._is_space_allowed(key):
            msg = (
                f"Confluence content '{content_id}' is outside the spaces allowed "
                "by CONFLUENCE_SPACES_FILTER"
            )
            raise SpaceAccessDeniedError(msg)

    def _assert_download_url_allowed(self, url: str) -> None:
        for pattern in _DOWNLOAD_URL_CONTENT_ID:
            if match := pattern.search(str(url)):
                self._assert_content_allowed(match.group(1))
                return
        raise SpaceAccessDeniedError(
            "Could not determine the owning content of the attachment download URL"
        )

    # --- enforcement --------------------------------------------------------

    def _enforce_rule(self, rule: GuardRule, arguments: dict[str, Any]) -> None:
        checks: tuple[tuple[tuple[str, ...], Callable[[str], None]], ...] = (
            (rule.content_ids, self._assert_content_allowed),
            (rule.space_keys, self._assert_space_allowed),
            (
                rule.space_ids,
                lambda sid: self._assert_space_allowed(
                    self._space_key_for_space_id(sid)
                ),
            ),
            (rule.download_urls, self._assert_download_url_allowed),
        )
        for params, check in checks:
            for param in params:
                for value in iter_values(arguments.get(param)):
                    check(value)

    def _scope_guard_active(self) -> bool:
        return self._space_allowlist is not None

    def _guard_result(self, rule: GuardRule, result: Any) -> Any:
        return rule.result(self, result) if rule.result else result


install_guards(SpaceGuardMixin, GUARD_RULES)
