"""Robot control, safety monitoring, and velocity calculation."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation

import numpy as np
from xarm.wrapper import XArmAPI

from .config import load_config
from .filters import OneEuroFilterVector, DeadZoneFilter, AccelerationLimiter
from .reset_function import reset_function

logger = logging.getLogger(__name__)


class RobotState(Enum):
    IDLE = "idle"
    VELOCITY_CONTROL = "velocity_control"
    POSITION_HOLD = "position_hold"
    EMERGENCY_STOP = "emergency_stop"


class VelocityCalculator:
    """Converts VR controller movement to robot velocities with filtering."""

    def __init__(self, config: dict):
        vc = config["velocity_calculator"]
        ml = config["motion_limits"]
        margin = ml["safety_margin"]
        accel_margin = ml["accel_safety_margin"]

        self.position_scale = vc["position_scale"]
        self.rotation_scale = vc["rotation_scale"]
        self.max_linear_vel = ml["linear"]["max_velocity"] * margin
        self.max_angular_vel = ml["angular"]["max_velocity"] * margin

        lf = vc["linear_filter"]
        af = vc["angular_filter"]
        self.linear_filter = OneEuroFilterVector(
            lf["min_cutoff"], lf["beta"], lf["d_cutoff"]
        )
        self.angular_filter = OneEuroFilterVector(
            af["min_cutoff"], af["beta"], af["d_cutoff"]
        )

        self.accel_limiter = AccelerationLimiter(
            max_linear_accel=ml["linear"]["max_acceleration"] * accel_margin,
            max_angular_accel=ml["angular"]["max_acceleration"] * accel_margin,
            max_linear_vel=self.max_linear_vel,
            max_angular_vel=self.max_angular_vel,
        )

        self.last_pos = None
        self.last_quat = None
        self.last_time = None

    def reset(self):
        self.last_pos = None
        self.last_quat = None
        self.last_time = None
        self.linear_filter.reset()
        self.angular_filter.reset()
        self.accel_limiter.reset()

    def calculate(
        self, position: Dict, quaternion: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate filtered velocities from VR controller pose."""
        t = time.time()
        pos = np.array([position["x"], position["y"], position["z"]])
        quat = np.array(
            [quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]]
        )

        if self.last_pos is None:
            self.last_pos, self.last_quat, self.last_time = pos, quat, t
            return np.zeros(3), np.zeros(3)

        dt = t - self.last_time
        if dt <= 0 or dt > 0.1:  # Skip on timing issues or large gaps
            self.last_pos, self.last_quat, self.last_time = pos, quat, t
            return np.zeros(3), np.zeros(3)

        # Linear velocity (VR -> robot coordinate transform)
        # VR: X=right, Y=up, Z=toward user (negative Z = away from user)
        # Robot: X=forward, Y=left, Z=up
        vel_vr = (pos - self.last_pos) / dt * self.position_scale
        linear = np.array([-vel_vr[2], vel_vr[0], -vel_vr[1]]) * 1000  # m/s -> mm/s

        # Angular velocity from quaternion difference
        angular = self._quat_to_angular(self.last_quat, quat, dt) * self.rotation_scale
        angular = np.array([angular[2], angular[0], angular[1]])  # VR -> robot

        # Filter and limit
        linear = self.linear_filter.filter(linear, t)
        angular = self.angular_filter.filter(angular, t)
        linear, angular = self.accel_limiter.limit(linear, angular, t)

        self.last_pos, self.last_quat, self.last_time = pos, quat, t
        return linear, angular

    def _quat_to_angular(self, q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
        """Convert quaternion change to angular velocity (deg/s)."""
        r1 = Rotation.from_quat(q1)
        r2 = Rotation.from_quat(q2)
        r_diff = r2 * r1.inv()
        rotvec = r_diff.as_rotvec()

        if np.linalg.norm(rotvec) < 1e-6:
            return np.zeros(3)

        return np.degrees(rotvec) / dt


class SafetyMonitor:
    """Workspace boundary enforcement and command timeout."""

    def __init__(self, config: dict):
        sf = config["safety"]
        self.timeout = sf["command_timeout"]
        self.buffer = sf["boundary_buffer"]
        self.prediction_time = sf["prediction_time"]
        self.workspace = sf["workspace"]
        self.last_cmd_time = time.time()
        self.position = None
        self._violation_logged = False

    def update_command_time(self):
        self.last_cmd_time = time.time()

    def update_position(self, pos: np.ndarray):
        self.position = pos.copy()

    def check_timeout(self) -> bool:
        return (time.time() - self.last_cmd_time) > self.timeout

    def limit_velocity(
        self,
        linear: np.ndarray,
        angular: np.ndarray,
        position: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Limit velocity based on workspace boundaries with predictive checking."""
        pos = position if position is not None else self.position
        if pos is None:
            # No position data - block all movement for safety
            return np.zeros(3), angular.copy(), False

        linear = linear.copy()
        is_safe = True

        for i, axis in enumerate(["x", "y", "z"]):
            lo, hi = self.workspace[axis]
            p, v = pos[i], linear[i]

            # Predict position after prediction_time
            predicted_p = p + v * self.prediction_time

            # Outside lower limit - recovery mode
            if p < lo:
                is_safe = False
                if v < 0:  # Trying to move further out - block completely
                    linear[i] = 0.0
                else:
                    # Allow positive velocity (moving back in) up to recovery speed
                    linear[i] = min(max(linear[i], 0), 30.0)
                if not self._violation_logged:
                    logger.warning(f"Workspace violation: {axis}={p:.1f} < {lo} (min)")
                    self._violation_logged = True

            # Outside upper limit - recovery mode
            elif p > hi:
                is_safe = False
                if v > 0:  # Trying to move further out - block completely
                    linear[i] = 0.0
                else:
                    # Allow negative velocity (moving back in) up to recovery speed
                    linear[i] = max(min(linear[i], 0), -30.0)
                if not self._violation_logged:
                    logger.warning(f"Workspace violation: {axis}={p:.1f} > {hi} (max)")
                    self._violation_logged = True

            # Predicted to cross lower limit - hard stop
            elif predicted_p < lo and v < 0:
                is_safe = False
                # Calculate max safe velocity to not cross boundary
                max_safe_v = (
                    (lo - p) / self.prediction_time if self.prediction_time > 0 else 0
                )
                linear[i] = max(v, max_safe_v)

            # Predicted to cross upper limit - hard stop
            elif predicted_p > hi and v > 0:
                is_safe = False
                # Calculate max safe velocity to not cross boundary
                max_safe_v = (
                    (hi - p) / self.prediction_time if self.prediction_time > 0 else 0
                )
                linear[i] = min(v, max_safe_v)

            # In lower buffer zone - approaching limit, gradual slowdown
            elif p < lo + self.buffer:
                if v < 0:  # Moving toward boundary
                    is_safe = False
                    scale = (p - lo) / self.buffer  # 0 at limit, 1 at buffer edge
                    linear[i] = v * scale * scale  # Quadratic for smoother stop

            # In upper buffer zone - approaching limit, gradual slowdown
            elif p > hi - self.buffer:
                if v > 0:  # Moving toward boundary
                    is_safe = False
                    scale = (hi - p) / self.buffer  # 0 at limit, 1 at buffer edge
                    linear[i] = v * scale * scale  # Quadratic for smoother stop
            else:
                self._violation_logged = False  # Reset when back in safe zone

        return linear, angular, is_safe


class XArmController:
    """Main robot controller for VR teleoperation."""

    # Button indices
    BTN_TRIGGER, BTN_GRIP, BTN_THUMBSTICK, BTN_A, BTN_B = 0, 1, 3, 4, 5

    def __init__(self, robot_ip: str = "192.168.0.244", simulate: bool = False):
        self.config = load_config()
        self.robot_ip = robot_ip
        self.simulate = simulate
        self.arm: Optional[XArmAPI] = None

        self.state = RobotState.IDLE
        self.running = False
        self.velocity_enabled = False

        self.velocity_calc = VelocityCalculator(self.config)
        dz = self.config["dead_zone"]
        self.deadzone = DeadZoneFilter(
            dz["linear_threshold"], dz["angular_threshold"], dz["ramp_ratio"]
        )
        self.safety = SafetyMonitor(self.config)

        ml = self.config["motion_limits"]
        self.cmd_limiter = AccelerationLimiter(
            max_linear_accel=ml["linear"]["max_acceleration"]
            * ml["accel_safety_margin"],
            max_angular_accel=ml["angular"]["max_acceleration"]
            * ml["accel_safety_margin"],
            max_linear_vel=ml["linear"]["max_velocity"] * ml["safety_margin"],
            max_angular_vel=ml["angular"]["max_velocity"] * ml["safety_margin"],
        )

        self.home_position = [370, -21, 114, 180, 0, 0]
        self.gripper_position = 800
        self.previous_buttons = {}
        self._a_press_start = None
        self._a_long_press_triggered = False
        self._cached_position: Optional[np.ndarray] = None
        self._position_lock = asyncio.Lock()

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot")

        # Callbacks (set by server)
        self.on_vr_data = None
        self.on_robot_state = None
        self.on_recording_toggle = None

    # Clear errors and configure
    def _clear_errors_robust(self):
        """Clear errors with retry - handles sticky error states."""
        for _ in range(3):
            self.arm.clean_error()
            self.arm.clean_warn()
            time.sleep(0.3)
            code, state = self.arm.get_state()
            if state != 4:  # Not in error state
                return True
            self.arm.motion_enable(enable=True)
            time.sleep(0.3)
        return False

    async def initialize(self) -> bool:
        """Initialize robot connection and move to home."""
        if self.simulate:
            logger.info("Running in simulation mode")
            return True

        try:
            self.arm = XArmAPI(self.robot_ip, is_radian=False)
            self.arm.connect()
            time.sleep(1)

            if not self.arm.connected:
                raise ConnectionError("Failed to connect to robot")

            self._clear_errors_robust()
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.5)

            rl = self.config["robot_limits"]
            self.arm.set_tcp_maxacc(rl["tcp_max_acceleration"])
            self.arm.set_joint_maxacc(rl["joint_max_acceleration"])
            self.arm.set_reduced_mode(True)
            self.arm.set_reduced_max_tcp_speed(rl["reduced_max_tcp_speed"])
            self.arm.set_reduced_max_joint_speed(rl["reduced_max_joint_speed"])
            self.arm.set_collision_sensitivity(rl["collision_sensitivity"])
            self.arm.set_teach_sensitivity(rl["teach_sensitivity"])
            self.arm.motion_enable(enable=True)

            # Gripper
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_position(800, wait=False)

            # Go home
            self.arm.set_position(*self.home_position, speed=60, wait=True)

            # Velocity control mode
            self.arm.set_mode(5)
            self.arm.set_state(0)
            time.sleep(0.3)

            logger.info("Robot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Robot initialization failed: {e}")
            return False

    async def _run_cmd(self, func, *args, **kwargs):
        """Run blocking robot command in executor."""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, lambda: func(*args, **kwargs)
        )

    async def stop_motion(self):
        """Stop all robot motion."""
        self.cmd_limiter.reset()
        if not self.simulate and self.arm:
            await self._run_cmd(
                self.arm.vc_set_cartesian_velocity, [0, 0, 0, 0, 0, 0], False, -1
            )

    async def set_gripper(self, position: int):
        self.gripper_position = position
        if not self.simulate and self.arm:
            await self._run_cmd(self.arm.set_gripper_position, position, False, 2000)

    def handle_button(self, idx: int, pressed: bool):
        """Handle VR controller button events."""
        was_pressed = self.previous_buttons.get(idx, False)
        self.previous_buttons[idx] = pressed

        # A button: long press (2s) = reset_function, short press = return home
        if idx == self.BTN_A:
            if pressed and not was_pressed:
                self._a_press_start = time.time()
                self._a_long_press_triggered = False
            elif pressed and was_pressed and self._a_press_start:
                if not self._a_long_press_triggered and (
                    time.time() - self._a_press_start >= 2.0
                ):
                    self._a_long_press_triggered = True
                    asyncio.create_task(self._execute_reset_function())
            elif not pressed and was_pressed:
                if not self._a_long_press_triggered and self._a_press_start:
                    asyncio.create_task(self._return_home())
                self._a_press_start = None
            return

        if not pressed or was_pressed:
            return

        if idx == self.BTN_B:
            asyncio.create_task(self._toggle_velocity())
        elif idx == self.BTN_TRIGGER:
            asyncio.create_task(self.set_gripper(0))
        elif idx == self.BTN_GRIP:
            asyncio.create_task(self.set_gripper(800))
        elif idx == self.BTN_THUMBSTICK:
            if self.on_recording_toggle:
                self.on_recording_toggle()

    async def _toggle_velocity(self):
        if self.state == RobotState.VELOCITY_CONTROL:
            logger.info("Stopping velocity control")
            self.state = RobotState.IDLE
            await self.stop_motion()
            self.velocity_calc.reset()
            self.cmd_limiter.reset()
        else:
            logger.info("Starting velocity control")
            self.state = RobotState.VELOCITY_CONTROL
            self.velocity_calc.reset()
            self.cmd_limiter.reset()

    async def _return_home(self):
        logger.info("Returning home")
        self.state = RobotState.POSITION_HOLD
        await self.stop_motion()

        if not self.simulate and self.arm:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.set_position(*self.home_position, speed=80, wait=True)
            self.arm.set_mode(5)
            self.arm.set_state(0)

        self.state = RobotState.IDLE
        logger.info("Returned to home")

    async def _execute_reset_function(self):
        """Execute predefined reset sequence (long press A)."""
        logger.info("Executing reset function")
        self.state = RobotState.POSITION_HOLD
        await self.stop_motion()

        if not self.simulate:
            await reset_function(self)

        self.state = RobotState.IDLE
        logger.info("Reset function completed")

    async def process_vr_data(self, data: Dict):
        """Process incoming VR controller data."""
        self.safety.update_command_time()

        # Notify recording callback
        if self.on_vr_data:
            self.on_vr_data(data)

        position = await self.get_cached_position()

        for ctrl in data.get("controllers", []):
            if ctrl.get("handedness") != "right" or not ctrl.get("connected"):
                continue

            for i, btn in enumerate(ctrl.get("buttons", [])):
                if btn:
                    self.handle_button(i, btn.get("pressed", False))

            if self.state != RobotState.VELOCITY_CONTROL:
                continue

            grip = ctrl.get("grip")
            if not grip:
                continue

            linear, angular = self.velocity_calc.calculate(
                grip["position"], grip["quaternion"]
            )
            linear, angular = self.deadzone.apply(linear, angular)
            linear, angular = self.cmd_limiter.limit(linear, angular)
            # Safety must be LAST - cannot be smoothed by acceleration limiter
            linear, angular, _ = self.safety.limit_velocity(linear, angular, position)

            await self._send_velocity(linear, angular)

    async def _send_velocity(self, linear: np.ndarray, angular: np.ndarray):
        """Send velocity command to robot."""
        if self.simulate:
            return

        if self.arm and self.arm.connected:
            vel = np.concatenate([linear, angular]).tolist()
            code = await self._run_cmd(
                self.arm.vc_set_cartesian_velocity, vel, False, -1
            )
            if code != 0:
                await self._handle_error(code)

    async def _handle_error(self, code: int):
        """Handle robot errors with collision recovery."""
        if not self.arm:
            return

        _, (err, _) = self.arm.get_err_warn_code()
        logger.warning(f"Robot error: api={code}, err={err}")

        # Collision errors need physical recovery
        if err in {1, 2, 10, 11}:
            self._clear_errors_robust()
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.3)

            # Move up slightly to clear collision
            code, pos = self.arm.get_position(is_radian=False)
            if code == 0:
                safe_pos = list(pos)
                safe_pos[2] = min(safe_pos[2] + 30, 380)
                self.arm.set_position(*safe_pos, speed=30, wait=True)
            time.sleep(0.3)
        else:
            self._clear_errors_robust()
            time.sleep(0.1)

        self.velocity_calc.reset()
        self.cmd_limiter.reset()

        if self.arm.mode != 5:
            self.arm.set_mode(5)
        self.arm.set_state(0)
        time.sleep(0.1)

    async def robot_state_poller(self):
        """Background task polling robot state."""
        interval = 1.0 / self.config["recording"]["robot_state_poll_hz"]

        while self.running:
            try:
                if self.arm and self.arm.connected:
                    code, pos = await self._run_cmd(self.arm.get_position, False)
                    joint_code, joints = await self._run_cmd(
                        self.arm.get_servo_angle, False
                    )

                    if code == 0 and isinstance(pos, (list, tuple)) and len(pos) >= 6:
                        async with self._position_lock:
                            self._cached_position = np.array(pos[:3])
                        self.safety.update_position(self._cached_position)

                        joint_positions = None
                        if (
                            joint_code == 0
                            and isinstance(joints, (list, tuple))
                            and len(joints) >= 7
                        ):
                            joint_positions = joints[:7]

                        if self.on_robot_state:
                            self.on_robot_state(
                                pos[:3],
                                pos[3:6],
                                self.gripper_position,
                                joint_positions,
                            )

                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"State poller error: {e}")
                await asyncio.sleep(0.1)

    async def safety_monitor_task(self):
        """Background task checking safety conditions."""
        while self.running:
            if (
                self.safety.check_timeout()
                and self.state == RobotState.VELOCITY_CONTROL
            ):
                logger.warning("Command timeout - stopping")
                await self.stop_motion()
                self.state = RobotState.IDLE
            await asyncio.sleep(0.1)

    async def get_cached_position(self) -> Optional[np.ndarray]:
        async with self._position_lock:
            return (
                self._cached_position.copy()
                if self._cached_position is not None
                else None
            )

    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        self._executor.shutdown(wait=True)

        if self.arm and self.arm.connected:
            try:
                self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0], False, -1)
                self.arm.set_mode(0)
                self.arm.set_state(4)
                self.arm.set_gripper_position(0, wait=True)
                self.arm.disconnect()
                logger.info("Robot disconnected")
            except Exception as e:
                logger.error(f"Shutdown error: {e}")
