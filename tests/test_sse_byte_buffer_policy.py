from __future__ import annotations

import json

import pytest

from headroom.proxy.sse_byte_buffer_policy import (
    find_sse_event_terminator,
    parse_sse_events_from_byte_buffer,
)


def test_find_sse_event_terminator_returns_earliest_separator() -> None:
    assert find_sse_event_terminator(bytearray(b"data: one\r\n\r\ndata: two\n\n")) == (9, 4)


def test_parse_sse_events_drains_complete_events_and_leaves_tail() -> None:
    buf = bytearray(b": ignored\nevent: delta\ndata: one\ndata: two\n\npartial")

    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "one\ntwo")]
    assert bytes(buf) == b"partial"


def test_parse_sse_events_preserves_split_utf8_tail() -> None:
    smile = "\U0001f642".encode()
    buf = bytearray(b"data: hello " + smile[:2])

    assert parse_sse_events_from_byte_buffer(buf) == []
    buf.extend(smile[2:] + b"\n\n")
    assert parse_sse_events_from_byte_buffer(buf) == [(None, "hello \U0001f642")]


def test_parse_sse_events_splits_crlf_lines() -> None:
    buf = bytearray(b"event: delta\r\ndata: one\r\ndata: two\r\n\r\n")

    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "one\ntwo")]
    assert not buf


def test_parse_sse_events_raises_on_complete_invalid_utf8_event() -> None:
    with pytest.raises(UnicodeDecodeError):
        parse_sse_events_from_byte_buffer(bytearray(b"data: \xff\n\n"))


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_parse_sse_events_preserves_unicode_line_separators_in_json(
    separator: str,
) -> None:
    payload = {"text": f"before{separator}after"}
    wire = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    events = parse_sse_events_from_byte_buffer(bytearray(wire))

    assert len(events) == 1
    assert json.loads(events[0][1]) == payload


def test_parse_sse_events_strips_only_one_leading_space() -> None:
    # SSE spec: "If value starts with a U+0020 SPACE character, remove it".
    # Further leading whitespace is part of the payload.
    buf = bytearray(b"event: delta\ndata:   indented\ndata:\ttab\ndata:none\n\n")

    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "  indented\n\ttab\nnone")]


def test_parse_sse_events_drains_cr_only_stream() -> None:
    # SSE line endings may be CRLF, LF or CR. A CR-only stream used to stay buffered
    # forever because only LF/LF and CRLF/CRLF were recognised.
    buf = bytearray(b"event: delta\rdata: one\rdata: two\r\r")
    assert parse_sse_events_from_byte_buffer(buf) == [("delta", "one\ntwo")]
    assert not buf


def test_parse_sse_events_keeps_mixed_line_endings_separate() -> None:
    # A CR-CR separator followed later by an LF-LF one: without CR support the first two
    # events merged into a single event with a newline-joined payload.
    buf = bytearray(b"data: one\r\rdata: two\n\n")
    assert parse_sse_events_from_byte_buffer(buf) == [(None, "one"), (None, "two")]
    assert not buf


def test_lone_trailing_cr_is_not_a_terminator() -> None:
    # The CR may be the first half of a CRLF arriving in the next chunk; treating it as a
    # complete line ending would invent a blank line.
    buf = bytearray(b"data: one\r")
    assert parse_sse_events_from_byte_buffer(buf) == []
    assert bytes(buf) == b"data: one\r"
    buf.extend(b"\n\r\n")
    assert parse_sse_events_from_byte_buffer(buf) == [(None, "one")]


def test_split_crlf_suffix_completes_the_event() -> None:
    # A CRLF ending split across chunks must still produce exactly one event, whether the
    # parser confirms the terminator before or after the final byte arrives.
    buf = bytearray(b"data: one\r\n\r")
    events = parse_sse_events_from_byte_buffer(buf)
    buf.extend(b"\n")
    events += parse_sse_events_from_byte_buffer(buf)
    assert events == [(None, "one")]
    assert parse_sse_events_from_byte_buffer(buf) == []
