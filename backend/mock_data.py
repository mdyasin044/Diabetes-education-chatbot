"""Dummy data standing in for your future REST APIs (same shapes you'll implement later)."""
import math, random
from datetime import datetime, timedelta

def _now():
    return datetime.now().replace(second=0, microsecond=0)

def mock_foods():
    n = _now()
    return [
        {"time": (n - timedelta(hours=10, minutes=30)).isoformat(), "name": "Oatmeal with berries", "carbs_g": 45, "protein_g": 10, "fat_g": 6, "calories": 270},
        {"time": (n - timedelta(hours=6)).isoformat(), "name": "Chicken pasta with garlic bread", "carbs_g": 85, "protein_g": 32, "fat_g": 20, "calories": 660},
        {"time": (n - timedelta(hours=2, minutes=30)).isoformat(), "name": "Apple with peanut butter", "carbs_g": 28, "protein_g": 7, "fat_g": 16, "calories": 270},
    ]

def mock_glucose():
    """144 readings, one per 5 minutes, mg/dL, with bumps after each meal."""
    random.seed(7)
    n = _now()
    start = n - timedelta(hours=12)
    meals = [(datetime.fromisoformat(f["time"]), f["carbs_g"]) for f in mock_foods()]
    out = []
    for i in range(144):
        t = start + timedelta(minutes=5 * i)
        g = 112 + 6 * math.sin(i / 20)
        for mt, carbs in meals:
            m = (t - mt).total_seconds() / 60
            if m > 0:
                g += carbs * 0.9 * (m / 60) * math.exp(1 - m / 60)  # peaks ~60 min
        out.append({"time": t.isoformat(), "mg_dl": round(g + random.uniform(-3, 3))})
    return out

def mock_watch():
    """Hourly steps and heart rate for the last 12 hours."""
    random.seed(11)
    n = _now().replace(minute=0)
    out = []
    for h in range(12, 0, -1):
        t = n - timedelta(hours=h)
        asleep = t.hour < 6
        out.append({
            "hour_start": t.isoformat(),
            "steps": 0 if asleep else random.randint(150, 1400),
            "avg_hr": random.randint(54, 60) if asleep else random.randint(68, 98),
        })
    return out
