from airbot_data_collection.airbot.robots.airbot_play import AIRBOTPlay

# from airbot_data_collection.airbot.robots.airbot_play_mock import AIRBOTPlayMock as AIRBOTPlay
from airbot_data_collection.airbot.robots.airbot_play import AIRBOTPlayConfig
from airbot_data_collection.airbot.sensors.cameras.v4l2 import BsonV4L2Camera
from airbot_data_collection.airbot.sensors.cameras.v4l2 import V4L2CameraConfig
from robot_config import RobotConfig


class Robot:
    """Robot class for the AIRBOT Play robot."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.robots = {
            name: AIRBOTPlay(AIRBOTPlayConfig(port=port))
            for name, port in zip(self.config.robot_groups, self.config.robot_ports, strict=True)
        }
        self.cameras = {
            name: BsonV4L2Camera(V4L2CameraConfig(camera_index=index))
            for name, index in zip(self.config.camera_names, self.config.camera_index, strict=True)
        }
        self.keys = list(self.robots.keys()) + list(self.cameras.keys())
        self.values = list(self.robots.values()) + list(self.cameras.values())
        for key, value in zip(self.keys, self.values, strict=True):
            if not value.configure():
                raise RuntimeError(f"Failed to configure {key}.")

    def switch_mode(self, mode):
        """Switch the mode of the robot."""
        for robot in self.robots.values():
            robot.switch_mode(mode)

    def capture_observation(self) -> dict:
        """Capture the current observation from the robot."""
        obs = {}
        for name, ins in zip(self.keys, self.values, strict=True):
            for key, value in ins.capture_observation().items():
                obs[f"{name}/{key}"] = value
        return obs

    def send_action(self, action):
        """Send the action to the robot."""
        for index, (_group, robot) in enumerate(self.robots.items()):
            robot.send_action(list(action[index * 7 : (index + 1) * 7]))

    def get_qpos(self, obs: dict) -> list[float]:
        """Get the joint positions of the robot."""
        qpos = []
        for group in self.config.robot_groups:
            qpos.extend(obs[f"{group}/arm/joint_state"]["data"]["position"])
            qpos.extend(obs[f"{group}/eef/joint_state"]["data"]["position"])
        return qpos

    def shutdown(self) -> bool:
        """Shutdown the robot."""
        for robot in self.robots.values():
            robot.shutdown()
        for camera in self.cameras.values():
            camera.shutdown()
        return True
