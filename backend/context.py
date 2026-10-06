"""Turns raw 12h data into a compact summary. Math happens here, not in the LLM.
Missing, empty or malformed data never raises: that section is just marked unavailable."""
import logging
from datetime import datetime
import data_client

log = logging.getLogger(__name__)


def _safe_fetch(fn):
    """Call a data_client function; return [] on any failure or non-list result."""
    try:
        data = fn()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("Data fetch failed (%s): %s", getattr(fn, "__name__", fn), e)
        return []


def _parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _num(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clean_glucose(raw):
    """Keep only readings with a valid time and value, sorted by time."""
    out = []
    for g in raw:
        if not isinstance(g, dict):
            continue
        t, v = _parse_time(g.get("time")), _num(g.get("mg_dl"))
        if t is not None and v is not None:
            out.append({"time": t.isoformat(), "mg_dl": round(v), "_t": t})
    out.sort(key=lambda x: x["_t"])
    return out


def _glucose_summary(readings):
    empty = {"available": False, "latest": None, "min": None, "max": None, "avg": None,
             "time_in_range_pct": None, "trend_15min": None, "readings": []}
    if not readings:
        return empty
    vals = [r["mg_dl"] for r in readings]
    trend_label = None
    if len(vals) >= 4:
        diff = vals[-1] - vals[-4]  # ~15 min at 5-min intervals
        trend_label = "rising" if diff > 8 else "falling" if diff < -8 else "steady"
    return {
        "available": True,
        "latest": vals[-1], "min": min(vals), "max": max(vals),
        "avg": round(sum(vals) / len(vals)),
        "time_in_range_pct": round(100 * sum(70 <= v <= 180 for v in vals) / len(vals)),
        "trend_15min": trend_label,
        "readings": [{"time": r["time"], "mg_dl": r["mg_dl"]} for r in readings],
    }


def _meal_summaries(foods, glucose):
    meals = []
    for f in foods:
        if not isinstance(f, dict):
            continue
        ft = _parse_time(f.get("time"))
        meal = {
            "time": ft.isoformat() if ft else None,
            "name": f.get("name") or "Unnamed food",
            "carbs_g": _num(f.get("carbs_g")), "protein_g": _num(f.get("protein_g")),
            "fat_g": _num(f.get("fat_g")), "calories": _num(f.get("calories")),
            "glucose_before": None, "peak_within_2h": None, "peak_time": None,
        }
        if ft and glucose:
            before = [g["mg_dl"] for g in glucose if g["_t"] <= ft]
            after = [g for g in glucose if 0 <= (g["_t"] - ft).total_seconds() <= 7200]
            meal["glucose_before"] = before[-1] if before else None
            if after:
                peak = max(after, key=lambda g: g["mg_dl"])
                meal["peak_within_2h"], meal["peak_time"] = peak["mg_dl"], peak["time"]
        meals.append(meal)
    return meals


def _watch_summary(raw):
    hourly = []
    for w in raw:
        if not isinstance(w, dict):
            continue
        t = _parse_time(w.get("hour_start"))
        steps, hr = _num(w.get("steps")), _num(w.get("avg_hr"))
        if t is not None and (steps is not None or hr is not None):
            hourly.append({"hour_start": t.isoformat(),
                           "steps": int(steps) if steps is not None else None,
                           "avg_hr": round(hr) if hr is not None else None})
    if not hourly:
        return {"available": False, "total_steps": None, "avg_hr": None, "hourly": []}
    steps = [h["steps"] for h in hourly if h["steps"] is not None]
    hrs = [h["avg_hr"] for h in hourly if h["avg_hr"] is not None]
    return {"available": True,
            "total_steps": sum(steps) if steps else None,
            "avg_hr": round(sum(hrs) / len(hrs)) if hrs else None,
            "hourly": hourly}


def build_context():
    glucose_raw = _clean_glucose(_safe_fetch(data_client.get_glucose))
    foods = _safe_fetch(data_client.get_foods)
    watch = _safe_fetch(data_client.get_watch)
    return {
        "glucose": _glucose_summary(glucose_raw),
        "meals": _meal_summaries(foods, glucose_raw),
        "watch": _watch_summary(watch),
    }


def to_prompt_text(ctx):
    """Compact text for the LLM. Missing sections are stated plainly."""
    g, w, meals = ctx["glucose"], ctx["watch"], ctx["meals"]
    lines = []

    if g["available"]:
        trend = f" ({g['trend_15min']})" if g["trend_15min"] else ""
        lines.append(f"Glucose (last 12h): latest {g['latest']} mg/dL{trend}, min {g['min']}, "
                     f"max {g['max']}, avg {g['avg']}, time in range 70-180: {g['time_in_range_pct']}%.")
    else:
        lines.append("Glucose: no data available.")

    if meals:
        for m in meals:
            parts = [f"Meal {m['time'][11:16] if m['time'] else 'unknown time'}: {m['name']}"]
            macros = [f"{m[k]:g}{u}" for k, u in (("carbs_g", "g carbs"), ("protein_g", "g protein"),
                                                   ("fat_g", "g fat"), ("calories", " kcal")) if m[k] is not None]
            if macros:
                parts.append(" - " + ", ".join(macros))
            if m["peak_within_2h"] is not None:
                before = f"before {m['glucose_before']}, " if m["glucose_before"] is not None else ""
                parts.append(f". Glucose {before}peak within 2h {m['peak_within_2h']} at {m['peak_time'][11:16]}")
            lines.append("".join(parts) + ".")
    else:
        lines.append("Meals: no data available.")

    if w["available"]:
        bits = []
        if w["total_steps"] is not None:
            bits.append(f"{w['total_steps']} steps total")
        if w["avg_hr"] is not None:
            bits.append(f"average HR {w['avg_hr']} bpm")
        lines.append("Watch: " + ", ".join(bits) + ".")
        lines.append("Hourly steps/HR: " + "; ".join(
            f"{h['hour_start'][11:13]}h {h['steps'] if h['steps'] is not None else '?'} steps/"
            f"{h['avg_hr'] if h['avg_hr'] is not None else '?'}bpm" for h in w["hourly"]))
    else:
        lines.append("Watch (steps, heart rate): no data available.")

    if not (g["available"] or meals or w["available"]):
        lines.append("Note: no personal data is available right now. Answer in general terms and "
                     "don't invent the user's numbers.")
    return "\n".join(lines)