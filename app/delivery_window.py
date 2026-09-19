"""Per-subscription delivery-window scheduling.

A delivery window restricts *when* a subscription's tasks may be claimed:

- ``timezone``: an IANA time zone name (e.g. ``Europe/Berlin``) in which the
  weekday and clock-time fields are interpreted;
- ``weekdays``: the ISO weekdays (Monday = 1 ... Sunday = 7) on which the
  window opens;
- ``local_start`` / ``local_end``: local wall-clock seconds after midnight,
  half-open interval ``[start, end)``.

``local_end < local_start`` (or end == 00:00) denotes a window that wraps
past local midnight: it opens on an allowed weekday at ``local_start`` and
closes on the following calendar day at ``local_end``.

All computations are pure functions over epoch seconds plus the (immutable)
window configuration: ``is_open`` answers for one instant, and ``next_open``
returns the earliest *future* opening instant. Both are evaluated against the
real IANA zone database via :mod:`zoneinfo`, so daylight-saving transitions
are handled from first principles:

- membership at an instant is derived from the instant's *real* local wall
  time, so spring-forward gaps are simply never "inside" and fall-back
  repeats follow the actual clock (a wrap window ending inside the repeated
  hour stays open until the second, real 02:00);
- a scheduled opening whose nominal local time falls inside a spring-forward
  gap resolves to the first real instant *after* the gap (the wall clock
  jumped over the configured instant); an opening inside a repeated hour
  resolves to the earlier of the two real instants.

The module deliberately keeps no global mutable state (zone objects are
cached read-only) and uses no process timers; callers persist the absolute
UTC epoch returned by :meth:`DeliveryWindow.next_open`, which makes every
scheduling decision reproducible by any worker and after any restart.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ISO 8601 weekdays, 1 = Monday ... 7 = Sunday (matches date.isoweekday()).
MIN_WEEKDAY = 1
MAX_WEEKDAY = 7

_SECONDS_PER_DAY = 24 * 3600

# Accept HH:MM and HH:MM:SS; output is always the canonical HH:MM:SS.
_CLOCK_RE = re.compile(r"^(\d{2}):(\d{2})(?::(\d{2}))?$")


@lru_cache(maxsize=128)
def get_zone(name: str) -> ZoneInfo:
    """Return the cached ZoneInfo for an IANA name; raise ValueError if unknown."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        # ValueError/OSError cover malformed names and a missing system tzdb.
        raise ValueError(f"unknown IANA time zone: {name!r}") from exc


def parse_clock(value: str) -> int:
    """Parse ``HH:MM`` / ``HH:MM:SS`` local time to seconds after midnight."""
    if not isinstance(value, str):
        raise ValueError("local time must be a string in HH:MM or HH:MM:SS form")
    match = _CLOCK_RE.match(value.strip())
    if match is None:
        raise ValueError("local time must use HH:MM or HH:MM:SS form, e.g. 09:00:00")
    hour, minute, second = (int(g) for g in match.groups("00"))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError("local time is out of range (hour 00-23, minute/second 00-59)")
    return hour * 3600 + minute * 60 + second


def format_clock(seconds: int) -> str:
    """Format seconds-after-midnight as canonical ``HH:MM:SS``."""
    hour, remainder = divmod(int(seconds), 3600)
    minute, second = divmod(remainder, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _is_real(zone: ZoneInfo, naive: datetime, ts: float) -> bool:
    """Whether epoch ``ts`` maps back to exactly the given naive local time.

    ZoneInfo fills non-existent gap times with a neighbouring offset, so a
    round-trip comparison is the reliable way to tell real, ambiguous and
    non-existent local times apart.
    """
    return datetime.fromtimestamp(ts, zone).replace(tzinfo=None) == naive


def _open_instant(day, seconds_after_midnight: int, zone: ZoneInfo) -> float:
    """Epoch second at which a window opening on a local calendar day occurs.

    ``day`` is the local ``date`` whose midnight anchors the time. The result
    is always a *real* instant:

    - ordinary times map directly;
    - a repeated hour (fall-back transition) maps to the earlier instant
      (``fold=0``), i.e. the window opens with the first pass;
    - a skipped hour (spring-forward gap) maps to the first real instant at
      or after the nominal time -- the wall clock jumped straight over it.
    """
    naive = datetime(day.year, day.month, day.day) + timedelta(seconds=seconds_after_midnight)
    ts_fold0 = naive.replace(tzinfo=zone, fold=0).timestamp()
    ts_fold1 = naive.replace(tzinfo=zone, fold=1).timestamp()
    if _is_real(zone, naive, ts_fold0):
        return ts_fold0
    if _is_real(zone, naive, ts_fold1):
        return ts_fold1

    # Non-existent local time (spring-forward gap): binary-search the first
    # real instant whose local wall time is >= the nominal opening time. The
    # local-time function is monotonic non-decreasing, and no DST transition
    # on Earth jumps more than 24h, so a 48h search span is ample.
    low = min(ts_fold0, ts_fold1)
    high = low + 2 * _SECONDS_PER_DAY
    for _ in range(64):  # 48h / 2**64 -- far below a microsecond
        mid = (low + high) / 2
        if datetime.fromtimestamp(mid, zone).replace(tzinfo=None) >= naive:
            high = mid
        else:
            low = mid
    return high


@dataclass(frozen=True)
class DeliveryWindow:
    """Immutable delivery-window configuration (all scheduling is DST-aware)."""

    timezone: str
    weekdays: tuple[int, ...]  # sorted unique ISO weekdays, never empty
    local_start: int  # seconds after local midnight
    local_end: int  # seconds after local midnight; <= local_start means wrap

    @property
    def zone(self) -> ZoneInfo:
        return get_zone(self.timezone)

    @property
    def wraps_past_midnight(self) -> bool:
        """True when the window opens on one day and closes on the next."""
        return self.local_end <= self.local_start

    def is_open(self, ts: float) -> bool:
        """Whether the window is open at the given epoch second."""
        local = datetime.fromtimestamp(ts, self.zone)
        seconds = local.hour * 3600 + local.minute * 60 + local.second + local.microsecond / 1_000_000
        weekday = local.isoweekday()
        if weekday in self.weekdays:
            if self.wraps_past_midnight:
                # The segment anchored on *this* weekday opens at local_start
                # and runs until midnight; its post-midnight tail is covered
                # by the previous-weekday check below.
                if seconds >= self.local_start:
                    return True
            elif self.local_start <= seconds < self.local_end:
                return True
        if self.wraps_past_midnight and (weekday - 2) % 7 + 1 in self.weekdays:
            # Before local_end on this weekday we are inside the tail of the
            # window that opened on the previous calendar day.
            if seconds < self.local_end:
                return True
        return False

    def next_open(self, ts: float) -> float:
        """Earliest opening instant strictly responsible for service after ``ts``.

        Precondition: the window is closed at ``ts`` (callers gate on
        :meth:`is_open` first). The next opening is the start of an allowed
        weekday's segment that is not already in the past; at most the next
        seven calendar days need inspecting because ``weekdays`` is non-empty.
        """
        zone = self.zone
        base_date = datetime.fromtimestamp(ts, zone).date()
        candidates: list[float] = []
        for delta in range(0, 8):
            day = base_date + timedelta(days=delta)
            if day.isoweekday() not in self.weekdays:
                continue
            opens_at = _open_instant(day, self.local_start, zone)
            if opens_at >= ts:
                candidates.append(opens_at)
        return min(candidates)
