"""An invented table of food delivery orders, for Lab 05."""

import numpy as np
import pandas as pd


def make_orders(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Return n invented orders. late is 1 when an order arrived late."""
    rng = np.random.default_rng(seed)
    distance = np.round(rng.gamma(2.2, 1.8, n).clip(0.3, 15), 1)
    items = rng.integers(1, 9, n)
    hour = rng.choice(np.arange(10, 23), n)
    weather = rng.choice(["clear", "rain", "snow"], n, p=[0.72, 0.22, 0.06])
    day_type = rng.choice(["weekday", "weekend"], n, p=[0.68, 0.32])
    vehicle = rng.choice(["bike", "scooter", "car"], n, p=[0.35, 0.40, 0.25])
    prep = np.round(rng.normal(14, 5, n).clip(4, 40), 1)
    courier_months = rng.integers(0, 60, n)
    rush = ((hour >= 12) & (hour <= 13)) | ((hour >= 18) & (hour <= 20))
    bad_weather = [weather == "snow", weather == "rain"]
    weather_effect = np.select(bad_weather, [1.3, 0.7], 0)
    vehicle_effect = np.select(
        [vehicle == "bike", vehicle == "scooter"], [0.35, 0.0], -0.15
    )
    score = (
        -4.2
        + 0.38 * distance
        + 0.09 * (prep - 14)
        + 0.08 * items
        + 0.95 * rush
        + 0.60 * (day_type == "weekend")
        + weather_effect
        + vehicle_effect
        - 0.022 * courier_months
    )
    late = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    rating = np.clip(np.round(4.6 - 1.6 * late + rng.normal(0, 0.6, n)), 1, 5)
    frame = pd.DataFrame(
        {
            "order_id": np.arange(100000, 100000 + n),
            "distance_km": distance,
            "items": items,
            "hour": hour,
            "weather": weather,
            "day_type": day_type,
            "vehicle": vehicle,
            "prep_minutes": prep,
            "courier_months": courier_months,
            "customer_rating": rating,
            "late": late,
        }
    )
    # some values are not recorded
    frame.loc[rng.random(n) < 0.06, "prep_minutes"] = np.nan
    frame.loc[rng.random(n) < 0.50, "customer_rating"] = np.nan
    return frame
