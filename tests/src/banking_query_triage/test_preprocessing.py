"""The serving cleaner preserves the normalization used during training."""

import pytest

from banking_query_triage.preprocessing import clean_messages


def test_normalizes_unicode_whitespace_and_preserves_negation():
    assert clean_messages(["  Ｉ DO NOT\nrecognize this!  "]) == [
        "i do not recognize this!"
    ]
    assert clean_messages(["  Card\nbill  "], lowercase=False) == ["Card bill"]


@pytest.mark.parametrize("message", ["", " \n", None, 42])
def test_rejects_blank_or_nonstring_messages(message):
    with pytest.raises(ValueError, match="nonempty"):
        clean_messages([message])
