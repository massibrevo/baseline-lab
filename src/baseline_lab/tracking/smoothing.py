"""Timestamp-aware adaptive smoothing; never fills a missing measurement."""

import math

import numpy as np


class OneEuro:
    def __init__(self, minimum_cutoff=1.5, beta=0.008):
        self.minimum_cutoff = minimum_cutoff
        self.beta = beta
        self.last = None

    def update(self, value, t):
        raw = np.array(value, dtype=float)
        if self.last is None or not 0 < t - self.last[0] <= 0.15:
            self.last = (t, raw, raw, np.zeros_like(raw))
            return tuple(raw)
        last_t, last_raw, last_filtered, last_derivative = self.last
        dt = t - last_t

        def alpha(cutoff):
            return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

        derivative = (raw - last_raw) / dt
        filtered_derivative = alpha(1.0) * derivative + (1 - alpha(1.0)) * last_derivative
        cutoff = self.minimum_cutoff + self.beta * np.linalg.norm(filtered_derivative)
        filtered = alpha(cutoff) * raw + (1 - alpha(cutoff)) * last_filtered
        self.last = (t, raw, filtered, filtered_derivative)
        return tuple(filtered)
