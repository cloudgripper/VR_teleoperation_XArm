import logging
import time

logger = logging.getLogger(__name__)


async def reset_function(controller):
    arm = controller.arm

    logger.info("Starting reset sequence...")

    # Switch to position mode
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.3)

    # Get current position
    code, current_pos = arm.get_position(is_radian=False)
    if code != 0:
        logger.error(f"Failed to get position: {code}")
        return

    current_x, current_y = current_pos[0], current_pos[1]
    move_speed = 75  # mm/s
    gripper_speed = 40

    code = arm.set_position(
        current_x + 10,
        current_y,
        18,
        current_pos[3],
        current_pos[4],
        current_pos[5],
        speed=move_speed,
        wait=True,
    )

    arm.set_gripper_position(0, wait=True, speed=gripper_speed)
    time.sleep(0.3)

    code = arm.set_position(
        current_x,
        current_y,
        150,
        current_pos[3],
        current_pos[4],
        current_pos[5],
        speed=move_speed,
        wait=True,
    )

    code = arm.set_position(
        320,
        -190,
        5,
        180,
        0,
        0,  # roll, pitch, yaw
        speed=move_speed,
        wait=True,
    )

    arm.set_gripper_position(800, wait=True, speed=gripper_speed)

    # Return to velocity control mode
    arm.set_mode(5)
    arm.set_state(0)
    time.sleep(0.3)

    logger.info("Reset sequence completed")
