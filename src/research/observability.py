from datetime import datetime
from typing import Any


def summarize_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact, serializable summary from an ordered workflow trace."""
    names = [event.get("name") for event in events if isinstance(event.get("name"), str)]
    timestamps = [
        datetime.fromisoformat(event["timestamp"])
        for event in events
        if isinstance(event.get("timestamp"), str)
    ]
    duration_seconds = 0.0
    if len(timestamps) >= 2:
        duration_seconds = round((max(timestamps) - min(timestamps)).total_seconds(), 6)

    event_counts: dict[str, int] = {}
    for name in names:
        event_counts[name] = event_counts.get(name, 0) + 1

    return {
        "event_count": len(events),
        "event_counts": event_counts,
        "first_event": names[0] if names else None,
        "last_event": names[-1] if names else None,
        "started_at": min(timestamps).isoformat() if timestamps else None,
        "finished_at": max(timestamps).isoformat() if timestamps else None,
        "duration_seconds": duration_seconds,
    }