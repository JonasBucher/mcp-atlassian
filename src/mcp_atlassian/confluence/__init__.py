"""Confluence API integration module.

This module provides access to Confluence content through the Model Context Protocol.
"""

from .analytics import AnalyticsMixin
from .attachments import AttachmentsMixin
from .client import ConfluenceClient
from .comments import CommentsMixin
from .config import ConfluenceConfig
from .labels import LabelsMixin
from .pages import PagesMixin
from .permissions import PermissionsMixin
from .restrictions import RestrictionsMixin
from .search import SearchMixin
from .space_guard import SpaceAccessDeniedError, SpaceGuardMixin
from .spaces import SpacesMixin
from .templates import TemplatesMixin
from .users import UsersMixin


class ConfluenceFetcher(
    SpaceGuardMixin,  # must stay first: wraps the methods of all mixins below
    SearchMixin,
    SpacesMixin,
    PagesMixin,
    CommentsMixin,
    LabelsMixin,
    UsersMixin,
    AnalyticsMixin,
    AttachmentsMixin,
    TemplatesMixin,
    PermissionsMixin,
    RestrictionsMixin,
):
    """Main entry point for Confluence operations, providing backward compatibility.

    This class combines functionality from various mixins to maintain the same
    API as the original ConfluenceFetcher class.

    Available mixins:
    - SpaceGuardMixin: Enforces CONFLUENCE_SPACES_FILTER on all operations
    - SearchMixin: CQL search operations
    - SpacesMixin: Space operations
    - PagesMixin: Page operations
    - CommentsMixin: Comment operations
    - LabelsMixin: Label operations
    - UsersMixin: User operations
    - AnalyticsMixin: Page view analytics (Cloud only)
    - AttachmentsMixin: Attachment operations
    - TemplatesMixin: Template operations
    - PermissionsMixin: Permission inspection operations (Cloud only)
    - RestrictionsMixin: Page restriction operations
    """

    pass


__all__ = [
    "ConfluenceFetcher",
    "ConfluenceConfig",
    "ConfluenceClient",
    "AnalyticsMixin",
    "PermissionsMixin",
    "RestrictionsMixin",
    "SpaceAccessDeniedError",
    "SpaceGuardMixin",
    "TemplatesMixin",
]
