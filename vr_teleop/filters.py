"""Signal processing filters for VR teleoperation."""

import time
from typing import Optional, Tuple
import numpy as np


class OneEuroFilter:
    """
    One-Euro Filter for adaptive signal smoothing.

    Automatically adjusts filtering based on signal speed:
    more filtering when slow, less when moving fast.
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def filter(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()

        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        # Derivative
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class OneEuroFilterVector:
    """One-Euro Filter for 3D vectors."""

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0
    ):
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]

    def reset(self):
        for f in self.filters:
            f.reset()

    def filter(self, vec: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is None:
            t = time.time()
        return np.array([self.filters[i].filter(vec[i], t) for i in range(3)])


class DeadZoneFilter:
    """Soft dead zone with smooth ramping to prevent acceleration spikes."""

    def __init__(
        self,
        linear_threshold: float = 15.0,
        angular_threshold: float = 2.0,
        ramp_ratio: float = 2.0,
    ):
        self.linear_threshold = linear_threshold
        self.angular_threshold = angular_threshold
        self.linear_outer = linear_threshold * ramp_ratio
        self.angular_outer = angular_threshold * ramp_ratio

    def apply(
        self, linear_vel: np.ndarray, angular_vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        linear_vel = linear_vel.copy()
        angular_vel = angular_vel.copy()

        # Linear (radial)
        mag = np.linalg.norm(linear_vel)
        if mag < self.linear_threshold:
            linear_vel = np.zeros(3)
        elif mag < self.linear_outer:
            t = (mag - self.linear_threshold) / (
                self.linear_outer - self.linear_threshold
            )
            linear_vel *= t * t * (3 - 2 * t)  # smoothstep

        # Angular (per-axis)
        for i in range(3):
            m = abs(angular_vel[i])
            if m < self.angular_threshold:
                angular_vel[i] = 0
            elif m < self.angular_outer:
                t = (m - self.angular_threshold) / (
                    self.angular_outer - self.angular_threshold
                )
                angular_vel[i] *= t * t * (3 - 2 * t)

        return linear_vel, angular_vel


class AccelerationLimiter:
    """Limits rate of velocity change to prevent robot acceleration errors."""

    def __init__(
        self,
        max_linear_accel: float = 500.0,
        max_angular_accel: float = 200.0,
        max_linear_vel: float = 70.0,
        max_angular_vel: float = 35.0,
    ):
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.prev_linear = None
        self.prev_angular = None
        self.prev_time = None

    def reset(self):
        self.prev_linear = None
        self.prev_angular = None
        self.prev_time = None

    def limit(
        self, linear_vel: np.ndarray, angular_vel: np.ndarray, t: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if t is None:
            t = time.time()

        linear_vel = linear_vel.copy()
        angular_vel = angular_vel.copy()

        if self.prev_time is None:
            self.prev_linear = np.zeros(3)
            self.prev_angular = np.zeros(3)
            self.prev_time = t
            return self._clamp_velocity(linear_vel, angular_vel)

        dt = max(t - self.prev_time, 0.005)  # min 5ms to prevent jitter issues

        # Limit acceleration
        max_lin_delta = self.max_linear_accel * dt
        lin_delta = linear_vel - self.prev_linear
        if np.linalg.norm(lin_delta) > max_lin_delta:
            linear_vel = self.prev_linear + lin_delta * (
                max_lin_delta / np.linalg.norm(lin_delta)
            )

        max_ang_delta = self.max_angular_accel * dt
        ang_delta = angular_vel - self.prev_angular
        if np.linalg.norm(ang_delta) > max_ang_delta:
            angular_vel = self.prev_angular + ang_delta * (
                max_ang_delta / np.linalg.norm(ang_delta)
            )

        linear_vel, angular_vel = self._clamp_velocity(linear_vel, angular_vel)

        self.prev_linear = linear_vel.copy()
        self.prev_angular = angular_vel.copy()
        self.prev_time = t
        return linear_vel, angular_vel

    def _clamp_velocity(
        self, linear: np.ndarray, angular: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lin_mag = np.linalg.norm(linear)
        if lin_mag > self.max_linear_vel:
            linear = linear * (self.max_linear_vel / lin_mag)
        ang_mag = np.linalg.norm(angular)
        if ang_mag > self.max_angular_vel:
            angular = angular * (self.max_angular_vel / ang_mag)
        return linear, angular
