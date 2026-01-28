"""Camera capture and data recording for teleoperation."""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Generator
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logger.warning("pyrealsense2 not installed - camera disabled")


def enumerate_cameras() -> list:
    """Get list of connected RealSense cameras."""
    if not REALSENSE_AVAILABLE:
        return []
    try:
        ctx = rs.context()
        return [
            {
                "serial": d.get_info(rs.camera_info.serial_number),
                "name": d.get_info(rs.camera_info.name),
            }
            for d in ctx.query_devices()
        ]
    except Exception as e:
        logger.error(f"Failed to enumerate cameras: {e}")
        return []


def get_camera_configs(camera_name: str) -> list:
    """Get resolution configs to try based on camera model."""
    if "D435" in camera_name:
        # D435/D435i: color 1920x1080, depth 1280x720 or 848x480
        return [
            {"color": (1920, 1080), "depth": (1280, 720), "fps": 30},
            {"color": (1920, 1080), "depth": (848, 480), "fps": 30},
            {"color": (1280, 720), "depth": (1280, 720), "fps": 30},
            {"color": (640, 480), "depth": (640, 480), "fps": 30},
        ]
    elif "D405" in camera_name:
        # D405: close-range camera, 640x480
        return [
            {"color": (640, 480), "depth": (640, 480), "fps": 30},
            {"color": (640, 480), "depth": (640, 480), "fps": 15},
        ]
    else:
        # Generic fallback
        return [
            {"color": (640, 480), "depth": (640, 480), "fps": 30},
            {"color": (640, 480), "depth": (640, 480), "fps": 15},
        ]


class CameraCapture:
    """Non-blocking RealSense camera capture."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        align_depth: bool = True,
    ):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed")

        self.preferred_width = width
        self.preferred_height = height
        self.preferred_fps = fps
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.camera_name: Optional[str] = None
        self.align_depth = align_depth

        self.pipeline: Optional[rs.pipeline] = None
        self.is_running = False
        self.intrinsics: Optional[Dict] = None
        self.depth_scale = 0.001

        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[tuple] = None
        self._frame_lock = threading.Lock()
        self._frame_count = 0

    def _hardware_reset(self):
        """Reset camera hardware."""
        if not self.serial:
            return
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial:
                    logger.info(f"Hardware reset: {self.serial}")
                    dev.hardware_reset()
                    time.sleep(3.0)  # Wait for device to reinitialize
                    break
        except Exception as e:
            logger.warning(f"Hardware reset failed: {e}")

    def _cleanup(self):
        """Clean up pipeline resources."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        self.pipeline = None

    def start(self, max_retries: int = 3) -> bool:
        """Start camera capture with retry logic."""
        # Detect camera model for appropriate configs
        camera_name = ""
        if self.serial:
            for dev in enumerate_cameras():
                if dev["serial"] == self.serial:
                    camera_name = dev["name"]
                    break

        configs = get_camera_configs(camera_name)
        # Prepend user-preferred config from config.yaml
        user_config = {
            "color": (self.preferred_width, self.preferred_height),
            "depth": (self.preferred_width, self.preferred_height),
            "fps": self.preferred_fps,
        }
        configs.insert(0, user_config)

        for config_idx, cfg in enumerate(configs):
            color_res = cfg["color"]
            depth_res = cfg["depth"]
            fps = cfg["fps"]

            for attempt in range(max_retries):
                try:
                    self._cleanup()

                    # Hardware reset on retry
                    if attempt > 0 or config_idx > 0:
                        self._hardware_reset()

                    self.pipeline = rs.pipeline()
                    rs_config = rs.config()

                    if self.serial:
                        rs_config.enable_device(self.serial)

                    rs_config.enable_stream(
                        rs.stream.color, color_res[0], color_res[1], rs.format.bgr8, fps
                    )
                    rs_config.enable_stream(
                        rs.stream.depth, depth_res[0], depth_res[1], rs.format.z16, fps
                    )

                    profile = self.pipeline.start(rs_config)
                    device = profile.get_device()
                    self.serial = device.get_info(rs.camera_info.serial_number)
                    self.camera_name = device.get_info(rs.camera_info.name)
                    self.depth_scale = device.first_depth_sensor().get_depth_scale()
                    self.width, self.height = color_res
                    self.fps = fps

                    # Extract intrinsics
                    color_profile = profile.get_stream(
                        rs.stream.color
                    ).as_video_stream_profile()
                    ci = color_profile.get_intrinsics()
                    self.intrinsics = {
                        "width": ci.width,
                        "height": ci.height,
                        "fx": ci.fx,
                        "fy": ci.fy,
                        "ppx": ci.ppx,
                        "ppy": ci.ppy,
                    }

                    # Warmup with timeout protection
                    warmup_start = time.time()
                    warmup_count = 0
                    while warmup_count < 30 and (time.time() - warmup_start) < 5.0:
                        try:
                            self.pipeline.wait_for_frames(timeout_ms=1000)
                            warmup_count += 1
                        except:
                            break

                    if warmup_count < 5:
                        raise RuntimeError(f"Warmup failed: only {warmup_count} frames")

                    self.is_running = True
                    self._thread = threading.Thread(
                        target=self._capture_loop, daemon=True
                    )
                    self._thread.start()

                    logger.info(
                        f"Camera started: {self.camera_name} - {color_res[0]}x{color_res[1]}@{fps}fps (serial: {self.serial})"
                    )
                    return True

                except Exception as e:
                    logger.warning(
                        f"Camera {self.serial} config {color_res}@{fps}fps attempt {attempt + 1} failed: {e}"
                    )
                    self._cleanup()
                    time.sleep(1.0)

        logger.error(f"Failed to start camera {self.serial} with any configuration")
        return False

    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        self.pipeline = None
        logger.info(f"Camera stopped ({self._frame_count} frames)")

    def _capture_loop(self):
        """Background capture thread."""
        align = rs.align(rs.stream.color) if self.align_depth else None

        while self.is_running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                if align:
                    frames = align.process(frames)

                color = np.asanyarray(frames.get_color_frame().get_data())
                depth = np.asanyarray(frames.get_depth_frame().get_data())

                with self._frame_lock:
                    self._latest_frame = (color, depth, time.time())
                    self._frame_count += 1

            except Exception as e:
                if self.is_running:
                    logger.error(f"Capture error: {e}")
                    time.sleep(0.1)

    def get_frame(self) -> Optional[tuple]:
        """Get latest frame (color, depth, timestamp)."""
        with self._frame_lock:
            return self._latest_frame


@dataclass
class TrajectoryPoint:
    """Single point in recorded trajectory."""

    timestamp: float
    tcp_position: Optional[list] = None
    tcp_orientation: Optional[list] = None
    joint_positions: Optional[list] = None
    gripper: int = 800
    vr_position: Optional[list] = None
    vr_quaternion: Optional[list] = None


class DataRecorder:
    """Records camera frames, robot state, and VR data. Supports multiple cameras."""

    def __init__(
        self,
        base_dir: str = "recordings",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        save_fps: float = 3.0,
        align_depth: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = 1.0 / save_fps
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.align_depth = align_depth

        self.cameras: Dict[str, CameraCapture] = {}  # serial -> camera
        self.session_dir: Optional[Path] = None
        self.is_recording = False
        self.start_time: Optional[float] = None

        self._trajectory: deque = deque(maxlen=100000)
        self._frame_count = 0
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Current state
        self._robot_pos = None
        self._robot_ori = None
        self._joint_positions = None
        self._gripper = 800
        self._vr_pos = None
        self._vr_quat = None
        self._state_lock = threading.Lock()

        # Camera serial to index mapping
        self._camera_index_map: Dict[str, int] = {}

    @property
    def camera_active(self) -> bool:
        return any(cam.is_running for cam in self.cameras.values())

    @property
    def num_cameras(self) -> int:
        return len([c for c in self.cameras.values() if c.is_running])

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def recording_duration(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    def get_tcp_state(self) -> tuple:
        """Get current TCP position and orientation."""
        with self._state_lock:
            return self._robot_pos, self._robot_ori, self._gripper

    def start_camera(self) -> bool:
        """Initialize and start all available cameras with staggered init."""
        if not REALSENSE_AVAILABLE:
            return False

        devices = enumerate_cameras()
        if not devices:
            logger.warning("No cameras found")
            return False

        logger.info(f"Found {len(devices)} camera(s): {[d['name'] for d in devices]}")

        success = False
        for i, dev in enumerate(devices):
            serial = dev["serial"]
            name = dev["name"]

            # Stagger initialization to avoid USB bandwidth conflicts
            if i > 0:
                logger.info(f"Waiting before initializing camera {i + 1}...")
                time.sleep(2.0)

            try:
                cam = CameraCapture(
                    width=self.camera_width,
                    height=self.camera_height,
                    fps=self.camera_fps,
                    serial=serial,
                    align_depth=self.align_depth,
                )
                if cam.start():
                    self.cameras[serial] = cam
                    success = True
                else:
                    logger.warning(f"Failed to start camera {name} ({serial})")
            except Exception as e:
                logger.error(f"Error starting camera {serial}: {e}")

        logger.info(f"Successfully started {len(self.cameras)} camera(s)")
        return success

    def stop_camera(self):
        """Stop all cameras."""
        for serial, cam in list(self.cameras.items()):
            cam.stop()
        self.cameras.clear()

    def update_robot_state(self, position, orientation, gripper, joint_positions=None):
        """Update robot state (called from robot poller)."""
        with self._state_lock:
            self._robot_pos = list(position) if position is not None else None
            self._robot_ori = list(orientation) if orientation is not None else None
            self._joint_positions = (
                list(joint_positions) if joint_positions is not None else None
            )
            self._gripper = gripper

    def update_vr_state(self, data: Dict):
        """Update VR state from controller data."""
        for ctrl in data.get("controllers", []):
            if ctrl.get("handedness") == "right":
                grip = ctrl.get("grip")
                if grip:
                    with self._state_lock:
                        self._vr_pos = [grip["position"][k] for k in ["x", "y", "z"]]
                        self._vr_quat = [
                            grip["quaternion"][k] for k in ["x", "y", "z", "w"]
                        ]
                break

    def start_session(self) -> Optional[Path]:
        """Start a new recording session."""
        if self.is_recording:
            return self.session_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True)

        # Create directories for each camera with numeric indices
        self._camera_index_map.clear()
        for cam_idx, serial in enumerate(self.cameras.keys()):
            self._camera_index_map[serial] = cam_idx
            (self.session_dir / f"rgb_{cam_idx}").mkdir()
            (self.session_dir / f"depth_{cam_idx}").mkdir()

        self._trajectory.clear()
        self._frame_count = 0
        self.start_time = time.time()
        self._stop_event.clear()

        # Start save thread
        self._save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self._save_thread.start()

        self.is_recording = True
        logger.info(
            f"Recording started: {self.session_dir} ({len(self.cameras)} cameras)"
        )
        return self.session_dir

    def stop_session(self) -> Dict:
        """Stop recording and save trajectory in standardized format."""
        if not self.is_recording:
            return {}

        self.is_recording = False
        self._stop_event.set()

        # Wait for save thread
        if self._save_thread:
            self._save_thread.join(timeout=5.0)

        # Convert trajectory to standardized format
        duration = time.time() - self.start_time if self.start_time else 0
        frames_data = []

        for idx, p in enumerate(self._trajectory):
            state_timestamp = datetime.fromtimestamp(p.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            frame = {
                "state_index": idx,
                "state_timestamp": state_timestamp,
            }

            # Joint positions (j1-j7)
            if p.joint_positions is not None:
                for i, joint_val in enumerate(p.joint_positions[:7]):
                    frame[f"j{i + 1}"] = float(joint_val)
            else:
                for i in range(7):
                    frame[f"j{i + 1}"] = 0.0

            # TCP position (x, y, z)
            if p.tcp_position is not None:
                frame["x"] = float(p.tcp_position[0])
                frame["y"] = float(p.tcp_position[1])
                frame["z"] = float(p.tcp_position[2])
            else:
                frame["x"] = 0.0
                frame["y"] = 0.0
                frame["z"] = 0.0

            # TCP orientation (roll, pitch, yaw)
            if p.tcp_orientation is not None:
                frame["roll"] = float(p.tcp_orientation[0])
                frame["pitch"] = float(p.tcp_orientation[1])
                frame["yaw"] = float(p.tcp_orientation[2])
            else:
                frame["roll"] = 0.0
                frame["pitch"] = 0.0
                frame["yaw"] = 0.0

            # Gripper state
            frame["gripper_state"] = {
                "type": "Jaw",
                "pos": p.gripper,
            }

            # Camera paths (reconstructed from frame index)
            for cam_idx in range(len(self.cameras)):
                frame[f"rgb_{cam_idx}_path"] = (
                    f"rgb_{cam_idx}/rgb_{cam_idx}_image_{idx}.png"
                )
                frame[f"depth_{cam_idx}_path"] = (
                    f"depth_{cam_idx}/depth_{cam_idx}_image_{idx}.npy"
                )

            # VR state
            if p.vr_position is not None:
                frame["_vr_right_position"] = p.vr_position
            if p.vr_quaternion is not None:
                frame["_vr_right_quaternion"] = p.vr_quaternion

            frames_data.append(frame)

        # Process trajectory: convert to radians and add actions
        processed_data = self._process_trajectory_data(frames_data)

        if self.session_dir:
            # Save JSON
            with open(self.session_dir / "trajectory.json", "w") as f:
                json.dump(processed_data, f, indent=2)

            # Save NPZ for efficient loading
            self._save_trajectory_npz(processed_data)

            # Save intrinsics for all cameras
            all_intrinsics = {}
            for serial, cam in self.cameras.items():
                cam_idx = self._camera_index_map.get(serial, 0)
                if cam.intrinsics:
                    all_intrinsics[str(cam_idx)] = {
                        "serial": serial,
                        "name": cam.camera_name,
                        "intrinsics": cam.intrinsics,
                    }
            if all_intrinsics:
                with open(self.session_dir / "camera_intrinsics.json", "w") as f:
                    json.dump(all_intrinsics, f, indent=2)

        logger.info(f"Recording stopped: {self._frame_count} frames, {duration:.1f}s")
        return {
            "frames": self._frame_count,
            "duration": duration,
            "session_dir": str(self.session_dir),
        }

    def _process_trajectory_data(self, trajectory_data: list) -> list:
        """
        Process trajectory data to:
        1. Convert angles (roll, pitch, yaw, j1-j7) from degrees to radians
        2. Add action field (next state values, or self for final state)
        """
        deg_to_rad = np.pi / 180.0
        processed = []

        for frame in trajectory_data:
            new_frame = frame.copy()

            # Convert joint angles to radians
            for j in range(1, 8):
                key = f"j{j}"
                if key in new_frame:
                    new_frame[key] = float(new_frame[key] * deg_to_rad)

            # Convert orientation to radians
            for key in ["roll", "pitch", "yaw"]:
                if key in new_frame:
                    new_frame[key] = float(new_frame[key] * deg_to_rad)

            processed.append(new_frame)

        # Add action field to each frame
        for i, frame in enumerate(processed):
            if i < len(processed) - 1:
                next_frame = processed[i + 1]
            else:
                next_frame = frame

            frame["action"] = {
                "x": next_frame.get("x", 0.0),
                "y": next_frame.get("y", 0.0),
                "z": next_frame.get("z", 0.0),
                "roll": next_frame.get("roll", 0.0),
                "pitch": next_frame.get("pitch", 0.0),
                "yaw": next_frame.get("yaw", 0.0),
                "gripper_state": next_frame.get(
                    "gripper_state", {"type": "Jaw", "pos": 800}
                ),
            }

        return processed

    def _save_trajectory_npz(self, processed_data: list):
        """Save trajectory data to NPZ format for efficient loading."""
        if not processed_data:
            return

        arrays = {}

        # State indices
        arrays["state_indices"] = np.array([f["state_index"] for f in processed_data])

        # Joint positions (j1-j7) - already in radians
        joint_data = np.zeros((len(processed_data), 7))
        for i, f in enumerate(processed_data):
            for j in range(7):
                joint_data[i, j] = f.get(f"j{j + 1}", 0.0)
        arrays["joint_positions"] = joint_data

        # TCP position (x, y, z)
        tcp_positions = np.array(
            [
                [f.get("x", 0.0), f.get("y", 0.0), f.get("z", 0.0)]
                for f in processed_data
            ]
        )
        arrays["tcp_positions"] = tcp_positions

        # TCP orientation (roll, pitch, yaw) - already in radians
        tcp_orientations = np.array(
            [
                [f.get("roll", 0.0), f.get("pitch", 0.0), f.get("yaw", 0.0)]
                for f in processed_data
            ]
        )
        arrays["tcp_orientations"] = tcp_orientations

        # Gripper positions
        arrays["gripper_positions"] = np.array(
            [f.get("gripper_state", {}).get("pos", 800) for f in processed_data]
        )

        # Actions array (x, y, z, roll, pitch, yaw, gripper)
        action_data = np.zeros((len(processed_data), 7))
        for i, f in enumerate(processed_data):
            action = f.get("action", {})
            action_data[i, 0] = action.get("x", 0.0)
            action_data[i, 1] = action.get("y", 0.0)
            action_data[i, 2] = action.get("z", 0.0)
            action_data[i, 3] = action.get("roll", 0.0)
            action_data[i, 4] = action.get("pitch", 0.0)
            action_data[i, 5] = action.get("yaw", 0.0)
            action_data[i, 6] = action.get("gripper_state", {}).get("pos", 800)
        arrays["actions"] = action_data

        # VR data if present
        if "_vr_right_position" in processed_data[0]:
            arrays["vr_positions"] = np.array(
                [f.get("_vr_right_position", [0, 0, 0]) for f in processed_data]
            )
        if "_vr_right_quaternion" in processed_data[0]:
            arrays["vr_quaternions"] = np.array(
                [f.get("_vr_right_quaternion", [0, 0, 0, 1]) for f in processed_data]
            )

        np.savez(self.session_dir / "trajectory.npz", **arrays)

    def _save_loop(self):
        """Background thread for saving frames."""
        last_save = 0

        while not self._stop_event.is_set():
            try:
                now = time.time()
                if now - last_save < self.save_interval:
                    time.sleep(0.01)
                    continue

                # Get current state
                with self._state_lock:
                    robot_pos = self._robot_pos
                    robot_ori = self._robot_ori
                    joint_positions = self._joint_positions
                    gripper = self._gripper
                    vr_pos = self._vr_pos
                    vr_quat = self._vr_quat

                # Save frames from all cameras
                idx = self._frame_count
                frame_data = {}
                has_frame = False

                for serial, cam in self.cameras.items():
                    if cam.is_running:
                        frame = cam.get_frame()
                        if frame:
                            color, depth, _ = frame
                            cam_idx = self._camera_index_map.get(serial, 0)

                            # Save color as PNG
                            if color is not None:
                                rgb_filename = f"rgb_{cam_idx}_image_{idx}.png"
                                cv2.imwrite(
                                    str(
                                        self.session_dir
                                        / f"rgb_{cam_idx}"
                                        / rgb_filename
                                    ),
                                    color,
                                )
                                frame_data[f"rgb_{cam_idx}_path"] = (
                                    f"rgb_{cam_idx}/{rgb_filename}"
                                )
                                has_frame = True

                            # Save depth as NPY
                            if depth is not None:
                                depth_filename = f"depth_{cam_idx}_image_{idx}.npy"
                                np.save(
                                    self.session_dir
                                    / f"depth_{cam_idx}"
                                    / depth_filename,
                                    depth,
                                )
                                frame_data[f"depth_{cam_idx}_path"] = (
                                    f"depth_{cam_idx}/{depth_filename}"
                                )

                # Save trajectory point
                if self.session_dir and (has_frame or robot_pos is not None):
                    self._trajectory.append(
                        TrajectoryPoint(
                            timestamp=now,
                            tcp_position=robot_pos,
                            tcp_orientation=robot_ori,
                            joint_positions=joint_positions,
                            gripper=gripper,
                            vr_position=vr_pos,
                            vr_quaternion=vr_quat,
                        )
                    )
                    self._frame_count += 1
                    last_save = now

            except Exception as e:
                logger.error(f"Save error: {e}")
                time.sleep(0.1)

    def _transform_image(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        """Apply camera-specific image transforms."""
        if "D405" in camera_name:
            # D405: rotate 180 degrees (flip both axes)
            return cv2.flip(image, -1)
        elif "D435" in camera_name:
            # D435/D435i: mirror horizontally (flip X only)
            return cv2.flip(image, 1)
        return image

    def generate_mjpeg_frames(
        self, target_fps: int = 15
    ) -> Generator[bytes, None, None]:
        """Generate MJPEG stream combining all cameras side-by-side."""
        interval = 1.0 / target_fps
        last_frame_time = 0

        while True:
            now = time.time()
            if now - last_frame_time < interval:
                time.sleep(0.01)
                continue

            frames = []
            active_cameras = [(s, c) for s, c in self.cameras.items() if c.is_running]

            if not active_cameras:
                # No cameras - show placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "No Camera",
                    (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                )
                _, jpeg = cv2.imencode(".jpg", placeholder)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
                last_frame_time = now
                continue

            # Collect frames from all cameras
            for serial, cam in active_cameras:
                frame = cam.get_frame()
                if frame:
                    color, _, _ = frame

                    color = self._transform_image(color, cam.camera_name or "")

                    # Resize to common height for side-by-side display
                    h, w = color.shape[:2]
                    target_h = 480
                    scale = target_h / h
                    resized = cv2.resize(color, (int(w * scale), target_h))

                    # Add camera label
                    label = cam.camera_name or serial[-4:]
                    cv2.putText(
                        resized,
                        label,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    frames.append(resized)

            if not frames:
                time.sleep(0.01)
                continue

            # Combine frames side by side
            combined = np.hstack(frames) if len(frames) > 1 else frames[0]

            # Add recording indicator
            if self.is_recording:
                cv2.circle(combined, (30, 60), 15, (0, 0, 255), -1)
                cv2.putText(
                    combined,
                    f"REC {self._frame_count}",
                    (55, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            _, jpeg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            last_frame_time = now
