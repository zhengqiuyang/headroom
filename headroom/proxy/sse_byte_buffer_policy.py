"""Pure SSE byte-buffer parsing policy."""

from __future__ import annotations

_CR = 0x0D
_LF = 0x0A

# An SSE event ends with a blank line: two line endings. A line ending is CRLF, LF or CR,
# so every ordered pair is a valid terminator -- including mixed pairs such as CRLF + LF.
# CR + LF is not a pair: those two bytes are themselves a single CRLF ending.
SSE_EVENT_TERMINATORS = (
    b"\r\n\r\n",
    b"\r\n\n",
    b"\r\n\r",
    b"\n\r\n",
    b"\n\n",
    b"\n\r",
    b"\r\r\n",
    b"\r\r",
)
SSE_EVENT_LINE_PREFIX = "event:"
SSE_DATA_LINE_PREFIX = "data:"


def _field_value(line: str, prefix: str) -> str:
    """Value after ``prefix``; per the SSE spec only one leading space is removed."""
    value = line[len(prefix) :]
    return value[1:] if value.startswith(" ") else value


def find_sse_event_terminator(buf: bytearray) -> tuple[int, int] | None:
    """Return the earliest complete SSE event terminator in ``buf``.

    The earliest separator wins; at the same offset the longest one wins, so a CRLF+CRLF
    ending is consumed whole instead of being split into CRLF + CR. A lone trailing CR
    matches nothing here: it may be the first half of a CRLF arriving in the next chunk,
    and waiting one chunk keeps the line ending intact.
    """
    best: tuple[int, int] | None = None
    for terminator in SSE_EVENT_TERMINATORS:
        index = buf.find(terminator)
        if index == -1:
            continue
        if best is None or index < best[0] or (index == best[0] and len(terminator) > best[1]):
            best = (index, len(terminator))
    return best


def parse_sse_events_from_byte_buffer(
    buf: bytearray,
) -> list[tuple[str | None, str]]:
    """Drain complete ``event:`` + ``data:`` events from a bytes buffer.

    Returns list of ``(event_name, data_str)`` tuples for complete events.
    Mutates ``buf`` in-place to leave only partial-event tail bytes.

    Operates on bytes; only decodes complete events as UTF-8 (raises if a
    *complete* event has invalid UTF-8, which is an upstream protocol bug).
    """
    events: list[tuple[str | None, str]] = []
    while True:
        terminator_match = find_sse_event_terminator(buf)
        if terminator_match is None:
            break
        idx, terminator_len = terminator_match
        event_bytes = bytes(buf[:idx])
        del buf[: idx + terminator_len]

        event_name: str | None = None
        data_lines: list[str] = []
        for raw_line in event_bytes.splitlines():
            line = raw_line.decode("utf-8")
            if not line:
                continue
            if line.startswith(":"):
                continue
            if line.startswith(SSE_EVENT_LINE_PREFIX):
                event_name = _field_value(line, SSE_EVENT_LINE_PREFIX)
            elif line.startswith(SSE_DATA_LINE_PREFIX):
                data_lines.append(_field_value(line, SSE_DATA_LINE_PREFIX))

        if data_lines:
            events.append((event_name, "\n".join(data_lines)))
    return events
