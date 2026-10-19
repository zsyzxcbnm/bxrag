"""API request/response contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.main import SearchRequest


def test_search_request_defaults() -> None:
    request = SearchRequest(query="balance transfer fee")
    assert request.top_k == 5
    assert request.mode == "hybrid"
    assert request.use_metadata is False


def test_search_request_rejects_invalid_top_k() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="x", top_k=50)


def test_search_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="")
