"""Per-subscription delivery windows: recurring local-time delivery schedules.

A delivery window restricts when a subscription's head task may be claimed
to recurring local-time intervals on given weekdays in an IANA timezone,
e.g. "weekdays 09:00-18:00 in Asia/Shanghai". Windows may cross midnight
(start > end means "from start until end the following day"); the early
morning portion of such a window belongs to the day the window *started*,
so whether it applies is decided by the previous day's weekday.

The configuration is persisted on the subscription row and every decision
below is a pure function of that persisted config plus the wall clock, so
all workers (and a restarted service) compute the same open/closed verdict
and the same next-opening instant.

DST transitions are resolved through zoneinfo. A window boundary that
falls on a nonexistent local time (spring forward) is interpreted with the
pre-transition offset, i.e. the window opens at the first valid instant at
or after the configured wall time; an ambiguous boundary (fall back) uses
its first occurrence (fold=0). Both choices are deterministic.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Weekdays are ISO 8601 numbers: 1 = Monday ... 7 = Sunday.


def load_timezone(name: str) -> ZoneInfo:
    """Return the ZoneInfo for an IANA timezone name.

    Raises ValueError for unknown names and for malformed keys (absolute
    paths etc.), so API validation can map both to a 422.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone: {name!r}") from exc


def parse_time_of_day(text: str) -> int:
    """Parse a local time 'HH:MM' or 'HH:MM:SS' to seconds since midnight.

    Raises ValueError for anything else (bad shape, out-of-range fields).
    """
    parts = text.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected 'HH:MM' or 'HH:MM:SS', got {text!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        raise ValueError(f"expected 'HH:MM' or 'HH:MM:SS', got {text!r}") from None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"time of day out of range: {text!r}")
    return hour * 3600 + minute * 60 + second


def format_time_of_day(seconds: int) -> str:
    """Inverse of parse_time_of_day: 'HH:MM', or 'HH:MM:SS' when needed."""
    hour, rem = divmod(seconds, 3600)
    minute, second = divmod(rem, 60)
    if second:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{hour:02d}:{minute:02d}"


def is_open(
    now: float,
    tz_name: str,
    days: set[int] | list[int],
    start_seconds: int,
    end_seconds: int,
) -> bool:
    """Whether the window is open at epoch instant `now`.

    The interval is start-inclusive / end-exclusive. For a cross-midnight
    window (start > end) the portion after midnight belongs to the window
    that started the previous day, so it is open exactly when the previous
    day was an allowed weekday.
    """
    local = datetime.fromtimestamp(now, ZoneInfo(tz_name))
    tod = local.hour * 3600 + local.minute * 60 + local.second
    day = local.isoweekday()
    if start_seconds < end_seconds:
        return day in days and start_seconds <= tod < end_seconds
    if tod >= start_seconds:
        return day in days
    if tod < end_seconds:
        return (day - 2) % 7 + 1 in days  # yesterday's window spills over
    return False


def next_open(now: float, tz_name: str, days: set[int] | list[int], start_seconds: int) -> float:
    """Next epoch instant strictly after `now` at which a window opens.

    The result depends only on the persisted window config and the wall
    clock, so concurrent workers compute the same instant. Weekdays repeat
    weekly, so with at least one allowed weekday (guaranteed by API
    validation) an opening always occurs within the next 8 days.
    """
    tz = ZoneInfo(tz_name)
    base_date = datetime.fromtimestamp(now, tz).date()
    allowed = set(days)
    for offset in range(8):
        day = base_date + timedelta(days=offset)
        if day.isoweekday() not in allowed:
            continue
        # Wall-clock arithmetic from local midnight keeps the opening at the
        # configured local time across DST transitions; a nonexistent local
        # boundary resolves via the pre-transition offset (deterministic).
        open_local = datetime.combine(day, dtime(0, 0), tzinfo=tz) + timedelta(
            seconds=start_seconds
        )
        candidate = open_local.timestamp()
        if candidate > now:
            return candidate
    # Unreachable through the API (days is never empty); stay total so the
    # worker loop can never crash on a hand-edited database row.
    return now + 8 * 86400
