"""Main entry point for VR teleoperation server."""

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from .config import load_config
from .robot import XArmController
from .recording import DataRecorder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VR Teleoperation Server", version="2.0.0")

controller: XArmController = None
recorder: DataRecorder = None


@app.on_event("startup")
async def startup():
    global controller, recorder

    if controller is None:
        logger.error("Controller not initialized")
        return

    if not await controller.initialize():
        logger.error("Robot initialization failed")
        return

    controller.running = True

    # Wire up recording callbacks
    controller.on_vr_data = recorder.update_vr_state
    controller.on_robot_state = recorder.update_robot_state
    controller.on_recording_toggle = toggle_recording_sync  # Recording toggle callback

    # Start background tasks
    asyncio.create_task(controller.safety_monitor_task())
    asyncio.create_task(controller.robot_state_poller())

    logger.info("Server started successfully")


def toggle_recording_sync():
    """Sync wrapper for recording toggle."""
    asyncio.create_task(toggle_recording())


@app.on_event("shutdown")
def shutdown():
    if recorder and recorder.is_recording:
        recorder.stop_session()
    if recorder:
        recorder.stop_camera()
    if controller:
        controller.shutdown()


@app.websocket("/ws/controllers")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for VR controller data."""
    await websocket.accept()
    logger.info("VR client connected")

    try:
        while True:
            try:
                data_str = await asyncio.wait_for(
                    websocket.receive_text(), timeout=10.0
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            data = json.loads(data_str)
            await controller.process_vr_data(data)

    except WebSocketDisconnect:
        logger.info("VR client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if controller:
            await controller.stop_motion()


async def toggle_recording():
    """Toggle data recording on/off."""
    if recorder.is_recording:
        stats = recorder.stop_session()
        logger.info(f"Recording stopped: {stats.get('frames', 0)} frames")
    else:
        session = recorder.start_session()
        logger.info(f"Recording started: {session}")


@app.get("/status")
async def get_status():
    """Get server status."""
    pos, ori, gripper = recorder.get_tcp_state() if recorder else (None, None, 800)
    return {
        "robot_connected": controller is not None and controller.arm is not None,
        "robot_state": controller.state.value if controller else "unknown",
        "camera_active": recorder.camera_active if recorder else False,
        "num_cameras": recorder.num_cameras if recorder else 0,
        "recording": recorder.is_recording if recorder else False,
        "recording_frames": recorder.frame_count if recorder else 0,
        "recording_duration": recorder.recording_duration if recorder else 0,
        "tcp_position": pos,
        "tcp_orientation": ori,
        "gripper": gripper,
    }


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream."""
    if not recorder or not recorder.camera_active:
        return HTMLResponse("<h1>Camera not active</h1>", status_code=503)

    return StreamingResponse(
        recorder.generate_mjpeg_frames(target_fps=15),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera")
async def camera_viewer():
    """HTML page for camera viewing with TCP position display."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Camera</title>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial; background: #1a1a2e; color: white; }
            .container { display: flex; flex-direction: column; align-items: center; }
            img { max-width: 95vw; border: 3px solid #4a4a6a; border-radius: 10px; }
            .info-panel { 
                display: flex; 
                gap: 20px; 
                margin-top: 15px; 
                flex-wrap: wrap;
                justify-content: center;
            }
            .info-box { 
                padding: 15px 20px; 
                background: #2a2a4a; 
                border-radius: 8px; 
                min-width: 200px;
            }
            .info-box h3 { margin: 0 0 10px 0; color: #8899aa; font-size: 14px; }
            .info-box .value { font-size: 18px; font-family: monospace; }
            .recording { color: #ff4444; }
            .connected { color: #44ff44; }
            .disconnected { color: #ff4444; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Robot Camera</h1>
            <img id="stream" src="/video_feed" onerror="setTimeout(() => this.src='/video_feed?' + Date.now(), 2000)">
            <div class="info-panel">
                <div class="info-box">
                    <h3>TCP POSITION (mm)</h3>
                    <div class="value" id="tcp-pos">--</div>
                </div>
                <div class="info-box">
                    <h3>TCP ORIENTATION (deg)</h3>
                    <div class="value" id="tcp-ori">--</div>
                </div>
                <div class="info-box">
                    <h3>GRIPPER</h3>
                    <div class="value" id="gripper">--</div>
                </div>
                <div class="info-box">
                    <h3>STATUS</h3>
                    <div class="value" id="status">--</div>
                </div>
            </div>
        </div>
        <script>
            function fmt(arr) {
                if (!arr) return '--';
                return arr.map(v => v.toFixed(1)).join(', ');
            }
            async function updateStatus() {
                try {
                    const r = await fetch('/status');
                    const d = await r.json();
                    
                    document.getElementById('tcp-pos').textContent = 
                        d.tcp_position ? `X: ${d.tcp_position[0].toFixed(1)}  Y: ${d.tcp_position[1].toFixed(1)}  Z: ${d.tcp_position[2].toFixed(1)}` : '--';
                    document.getElementById('tcp-ori').textContent = 
                        d.tcp_orientation ? `R: ${d.tcp_orientation[0].toFixed(1)}  P: ${d.tcp_orientation[1].toFixed(1)}  Y: ${d.tcp_orientation[2].toFixed(1)}` : '--';
                    document.getElementById('gripper').textContent = d.gripper !== undefined ? d.gripper : '--';
                    
                    let status = d.robot_state || 'unknown';
                    if (d.recording) {
                        status += ` | <span class="recording">REC ${d.recording_frames}</span>`;
                    }
                    status += ` | Cameras: ${d.num_cameras}`;
                    document.getElementById('status').innerHTML = status;
                } catch(e) {
                    document.getElementById('status').textContent = 'Connection error';
                }
            }
            setInterval(updateStatus, 100);  // Update at 10Hz for responsive TCP display
            updateStatus();
        </script>
    </body>
    </html>
    """)


@app.get("/")
async def index():
    """Serve the WebXR client page."""
    vr_ui_path = Path(__file__).parent / "vr_ui.html"
    if vr_ui_path.exists():
        return HTMLResponse(vr_ui_path.read_text())
    return HTMLResponse("<h1>VR UI not found</h1>", status_code=404)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VR Teleoperation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--robot-ip", default="192.168.0.244", help="Robot IP")
    parser.add_argument("--simulate", action="store_true", help="Simulation mode")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera")
    args = parser.parse_args()

    global controller, recorder

    config = load_config()

    controller = XArmController(robot_ip=args.robot_ip, simulate=args.simulate)
    recorder = DataRecorder(
        camera_width=config["camera"]["width"],
        camera_height=config["camera"]["height"],
        camera_fps=config["camera"]["fps"],
        save_fps=config["recording"]["save_fps"],
        align_depth=config["camera"]["align_depth"],
    )

    if not args.no_camera:
        if recorder.start_camera():
            logger.info("Camera initialized")
        else:
            logger.warning("Camera initialization failed")

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"VR endpoint: ws://{args.host}:{args.port}/ws/controllers")
    print(f"Camera viewer: http://{args.host}:{args.port}/camera")
    print(f"Status: http://{args.host}:{args.port}/status")
    if args.simulate:
        print("Running in SIMULATION mode")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

