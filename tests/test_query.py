"""Query parsing tests."""

from __future__ import annotations

from src.retrieval.query import parse_query


def test_full_product_name() -> None:
    parsed = parse_query("How many days do I have to request a transfer on Apex Platinum?")
    assert parsed.card_tier == "Apex Platinum"


def test_product_alias_without_apex_prefix() -> None:
    parsed = parse_query("Is there an authorized user fee on Everyday Freedom?")
    assert parsed.card_tier == "Apex Everyday Freedom"


def test_generic_question_has_no_tier_filter() -> None:
    parsed = parse_query("What is the maximum deadline for filing a merchant dispute?")
    assert parsed.card_tier is None


def test_year_and_policy_code() -> None:
    parsed = parse_query("Show FRD-101-APEX-2024 rules for 2024")
    assert parsed.effective_year == "2024"
    assert parsed.policy_code == "FRD-101-APEX-2024"

