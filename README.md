# VR Teleoperation Framework

VR teleoperation system for controlling a UFactory xArm robot using a Meta Quest headset.

## Prerequisites

1. Python (v3.10.18) and a virtual environment ([uv](https://docs.astral.sh/uv/#installation) or pip venv). Uv is recommended.
2. Android Debug Bridge (ADB) installed.
	- It is recommended to install ADB through Brew (on Linux and MacOS).  
	- Do this by first making sure that [brew ](https://brew.sh/) is installed, and then running `brew install adb-enhanced`.
	- Instructions on how to install ADB on Windows [here](https://www.xda-developers.com/install-adb-windows-macos-linux/#:~:text=How%20to%20set%20up%20ADB%20on%20Microsoft%20Windows).
3. Meta Quest headset with [developer mode](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/) enabled.
4. Allow [USB Debugging](https://knowledge.vr-expert.com/kb/how-to-enable-usb-debugging-on-the-oculus-quest-2/) on the Meta Quest headset.
5. UFactory xArm robot.

## Installation

```bash
# Create and activate virtual environment
uv venv && source .venv/bin/activate  # or use pip venv

# Install dependencies
uv pip install -r requirements.txt  # or pip install -r requirements.txt
```

## Usage


### 1. Connect Quest Headset

Connect the headset to your computer via USB and forward the server port:

```bash
adb reverse tcp:8080 tcp:8080
```

#### First time use with a new computer

When connecting to the headset for the first time with a new computer, you need to approve USB debugging.

Do this by first connecting the VR headset through USB. Doing this will make a notification pop up in the headset that prompts you to allow USB debugging. Click Accept.



### 2. Start the Server

```bash
python -m vr_teleop.server --robot-ip [your robot IP address]
```

### 3. View the cameras

Go to `http://localhost:8080/camera` in your computer browser (not headset).
You should be able to see two camera viewports and information about the gripper.


### 4. Enter VR Mode

1. In the Quest browser, navigate to `localhost:8080`
2. Press **Enter VR Mode**
3. Define the workspace boundary when prompted (If not prompted, the workspace boundary is already defined)

The VR headset will be dark. You don't need to wear the headset while operating the robot. Make sure that the VR Controller is in view of the headset cameras. 

By default, the Quest VR headset will go into power saving mode if you put it down without wearing it. This will cause it to disconnect from the robot controller service. If this happens, you need to go to the browser, close the `localhost:8080` tab, and re-open it. Refreshing the tab is not enough.

## Controller Mapping (Right Controller)

| Button | Action |
|--------|--------|
| B | Toggle velocity control (start/stop) |
| A | Return to home position |
| Trigger | Close gripper |
| Grip | Open gripper |
| Thumbstick (press) | Toggle recording |

## Coordinate System

The VR controller coordinate system locks when **Enter VR Mode** is pressed. The robot coordinate system mirrors the headset orientation at that moment. After activation:

- The headset can be moved freely without affecting the coordinate mapping
- The controller must remain within the headset's camera vision
- The controller must stay within the defined VR workspace boundary

## Troubleshooting

**Robot reaches joint limits and freezes:** Press the A button to return home and reset.

**Headset enters sleep mode:** Keep the headset worn or held during operation to prevent power saving mode from interrupting the connection.

**Connection lost:** Close the `localhost:8080` tab completely and reopen it. Do not reload the page. The server does not need to be restarted.

## Endpoints

Note: the default host and port is `localhost` and `8080` respectively. If customized or remotely teleoperating, you need to update the addresses below.

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8080/ws/controllers` | WebSocket for VR controller data |
| `http://localhost:8080/camera` | Camera feed with TCP position overlay |
| `http://localhost:8080/status` | JSON status endpoint |

## Recording

Recorded sessions are saved to the `recordings/` directory with:
- RGB and depth images from all connected RealSense cameras
- `trajectory.json` — Human-readable trajectory with actions
- `trajectory.npz` — NumPy arrays (efficient, not human-readable)
- `camera_intrinsics.json` — Camera calibration data

## Acknowledgement

This work was supported by the European Commission under the Horizon Europe Framework Programme project SoftEnable, under Grant 101070600.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
