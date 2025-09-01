import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List

from airbot_mmk import AIRBOTMMK, AIRBOTMMKConfig
from mmk2_types.types import MMK2Components
from airbot_data_collection.basis import SystemMode
from robot_config import RobotConfig
import numpy as np

logger = logging.getLogger(__name__)


class Robot:
    """Robot wrapper for MMK2 using the existing AIRBOTMMK class."""

    def __init__(self, config: RobotConfig):
        """Initialize MMK2 robot using the existing AIRBOTMMK interface.
        
        Args:
            config: Robot configuration containing ports, camera indices, etc.
        """
        self.config = config
        logger.info("Initializing MMK2 Robot using AIRBOTMMK class...")

        yaml_config = self._get_default_config()
        logger.warning("Using default MMK configuration")
        
        # Create AIRBOTMMK configuration
        mmk_config = AIRBOTMMKConfig(
            ip=yaml_config.get("ip", "192.168.11.200"),
            components=yaml_config.get("components", [
                "left_arm", "left_arm_eef", "right_arm", "right_arm_eef", "head", "spine"
            ]),
            cameras=yaml_config.get("cameras", {}),
            default_action=yaml_config.get("default_action"),
            demonstrate=True  # Enable demonstration mode for action recording
        )
        
        # Initialize the MMK robot
        self.mmk_robot = AIRBOTMMK(mmk_config)
        
        # Configure the robot
        if not self.mmk_robot.configure():
            raise RuntimeError("Failed to configure MMK robot")
        
        logger.info("MMK2 Robot initialization completed successfully")
        
        # Store the default reset position from yaml config
        self.reset_position = yaml_config.get("default_action")
        
    def _get_default_config(self) -> dict:
        """Get default configuration if yaml file is not available."""
        return {
            "ip": "192.168.11.200",
            "components": ["left_arm", "left_arm_eef", "right_arm", "right_arm_eef", "head", "spine"],
            "cameras": {
                "head_camera": {
                    "camera_type": "REALSENSE",
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "false"
                },
                "left_camera": {
                    "camera_type": "USB",
                    "video_device": "/dev/left_camera",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25"
                },
                "right_camera": {
                    "camera_type": "USB", 
                    "video_device": "/dev/right_camera",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25"
                }
            },
            "default_action": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left_arm (6 joints)
                0.0,                             # left_arm_eef (1 joint)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right_arm (6 joints)
                0.0,                             # right_arm_eef (1 joint)
                0.0, -1.0,                       # head (2 joints)
                0.15,                            # spine (1 joint)
            ]
        }

    def capture_observation(self) -> Dict:
        try:
            # Use the existing capture_observation method from AIRBOTMMK
            obs = self.mmk_robot.capture_observation()
            logger.debug(f"Captured observations with {len(obs)} keys")
            return obs
            
        except Exception as e:
            logger.error(f"Failed to capture observations: {e}")
            return {}

    def get_qpos(self, obs: Dict) -> List[float]:
        qpos = []
        components = ["left_arm", "left_arm_eef", "right_arm", "right_arm_eef", "head", "spine"]
        
        for comp in components:
            key = f"observation/{comp}/joint_state"
            if key in obs:
                joint_data = obs[key]
                if isinstance(joint_data, dict) and "data" in joint_data:
                    positions = joint_data["data"].get("position", [])
                    qpos.extend(positions)
                    break                
        
        return qpos

    def send_action(self, action: List[float]) -> None:
        action = list(action)
        if len(action) == 32:
            action = action[:14]
            logger.info(f"截断32维action到前14维: {len(action)}  action: {action}")
            # Get default head and spine values from reset position
            # MMK structure: left_arm(6) + left_eef(1) + right_arm(6) + right_eef(1) + head(2) + spine(1)
            default_head_spine = self.reset_position[-3:]  # Last 3 values: head(2) + spine(1)
            action.extend(default_head_spine)
            logger.info(f"添加默认head和spine值: {default_head_spine}, 最终17维action")
        elif len(action) == 17:
            logger.info(f"17维action: {action}")
        else:
            logger.error(f"action长度错误: {len(action)}")
            return

        try:
            self.mmk_robot.send_action(action)            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")

    def reset_to_home(self) -> None:
        """Reset robot to home position safely."""
        logger.info("Resetting robot to home position...")
        time.sleep(0.5)  # Allow mode switch to complete
        # Use the existing reset method from AIRBOTMMK
        self.mmk_robot.reset(sleep_time=1.0)
        logger.info("Robot reset to home position completed")

    def shutdown(self) -> bool:
        """Safely shutdown the robot.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        logger.info("Shutting down MMK2 Robot...")
        
        try:
            # Reset to safe position before shutdown
            self.reset_to_home()
            
            # Use the existing shutdown method
            success = self.mmk_robot.shutdown()
            
            if success:
                logger.info("MMK2 Robot shutdown completed successfully")
            else:
                logger.warning("MMK2 Robot shutdown completed with errors")
                
            return success
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    def get_system_status(self) -> Dict:
        """Get current system status information.
        
        Returns:
            Dictionary containing system status
        """
        return {
            "robot_type": "MMK2",
            "config": self.mmk_robot.config.model_dump() if hasattr(self.mmk_robot.config, 'model_dump') else str(self.mmk_robot.config),
            "components": [comp.value for comp in self.mmk_robot.config.components],
            "cameras": list(self.mmk_robot.config.cameras.keys()),
            "default_action": self.mmk_robot.config.default_action,
            "demonstrate_mode": self.mmk_robot.config.demonstrate,
            "initialized": True
        }

    def get_robot_info(self) -> Dict:
        """Get robot hardware information."""
        return self.mmk_robot.get_info()