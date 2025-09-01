from pydantic import BaseModel, PositiveInt
from airbot_data_collection.basis import System
from mmk2_types.types import (
    MMK2Components,
    ImageTypes,
    TopicNames,
    MMK2ComponentsGroup,
    ControllerTypes,
    JointNames,
)
from mmk2_types.grpc_msgs import JointState,Time, MoveServoParams,ForwardPositionParams,TrajectoryParams
from airbot_py.airbot_mmk2 import AirbotMMK2
from pydantic import BaseModel, PositiveInt
from typing import Optional, List, Union, Dict, Tuple
import numpy as np
import time
from turbojpeg import TurboJPEG
from time import time_ns

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIRBOTMMKConfig(BaseModel):
    ip: str = "192.168.11.200"
    port: PositiveInt = 50055
    name: Optional[str] = None
    domain_id: Optional[int] = None
    components: List[Union[str, MMK2Components]] = []
    default_action: Optional[List[float]] = None
    cameras: Dict[Union[str, MMK2Components], Dict[str, str]] = {}
    demonstrate: bool = True

    def model_post_init(self, context):
        for i, component in enumerate(self.components):
            if isinstance(component, str):
                self.components[i] = MMK2Components[component.upper()]
        for cam in list(self.cameras.keys()):
            if isinstance(cam, str):
                self.cameras[MMK2Components[cam.upper()]] = self.cameras.pop(cam)

class AIRBOTMMK(System):
    config: AIRBOTMMKConfig

    def on_configure(self) -> bool:
        self.interface = AirbotMMK2(ip=self.config.ip)
        self.traj_mode = False  # 添加traj_mode属性
        self._action_topics = {
            comp: TopicNames.tracking.format(component=comp.value)
            for comp in MMK2ComponentsGroup.ARMS
        }
        self._action_topics.update(
            {
                comp: TopicNames.controller_command.format(
                    component=comp.value,
                    controller=ControllerTypes.FORWARD_POSITION.value,
                )
                for comp in MMK2ComponentsGroup.HEAD_SPINE
            }
        )
        print(f"[DEBUG] Action topics ALL: {self._action_topics}")
        self.interface.listen_to(self._action_topics.values())
        self.interface.enable_resources(self.config.cameras)
        self._joint_names = JointNames().__dict__
        self.cameras = {cam: [ImageTypes.COLOR] for cam in self.config.cameras}

        print(f"[DEBUG] _joint_names: {self._joint_names}")
        self._check_joints(self.interface.get_robot_state().joint_state.name)
        self.reset()
        self.jpeg = TurboJPEG()
        return True

    def get_info(self):
        return {}

    def reset(self, sleep_time=0):
        if self.config.default_action is not None:
            goal = self._action_to_goal(self.config.default_action)
            self._move_by_traj(goal)
        else:
            logger.warning("No default action is set.")
        time.sleep(sleep_time)
        self.enter_servo_mode()
        print(f"[DEBUG] mmk reset done")

    def _move_by_traj(self, goal: dict):
        if self.config.demonstrate:
            # TODO: since the arms and eefs are controlled by the teleop bag
            for comp in MMK2ComponentsGroup.ARMS_EEFS:
                goal.pop(comp)
        if goal:
            self.interface.set_goal(goal, TrajectoryParams())
            self.interface.set_goal(goal, ForwardPositionParams())

    def send_action(self, action):
        goal = self._action_to_goal(action)
        if self.traj_mode:
            self.interface.set_goal(goal, TrajectoryParams())
        else:
            self.interface.set_goal(goal, MoveServoParams())

    def _observation_to_action(self, obs: dict) -> list[float]:
        """将 bson 观察数据转换为动作列表"""
        action = []

        # 按照组件顺序提取关节位置，兼容不同的数据格式
        for comp in self.config.components:
            comp_name = comp.value

            # 尝试多种可能的数据格式和命名空间
            possible_keys = [
                f"/mmk/mmk/{comp_name}/joint_state",  # bson_player 原始格式
                f"{comp_name}/joint_state",           # 简化格式
                f"action/{comp_name}/joint_state",    # action 命名空间
                f"observation/{comp_name}/joint_state"  # observation 命名空间
            ]

            found_data = False
            for joint_key in possible_keys:
                if joint_key in obs:
                    joint_data = obs[joint_key]

                    # 处理不同的数据结构
                    pos_data = None
                    if isinstance(joint_data, dict):
                        if "data" in joint_data and isinstance(joint_data["data"], dict):
                            pos_data = joint_data["data"].get("position", [])
                        elif "position" in joint_data:
                            pos_data = joint_data["position"]
                        elif "position" in joint_data:
                            pos_data = joint_data["position"]
                    elif isinstance(joint_data, list):
                        pos_data = joint_data

                    if pos_data:
                        action.extend(pos_data)
                        found_data = True
                        break

            if not found_data:
                self.get_logger().warning(f"未找到组件 {comp_name} 的关节数据，尝试的键: {possible_keys}")

        if not action:
            self.get_logger().error("无法从观察数据中提取任何关节位置信息")
            return []

        return action

    def _action_to_goal(self, action) -> Dict[MMK2Components, JointState]:
        self._action_check(action)
        goal = {}
        j_cnt = 0
        for comp in self.config.components:
            end = j_cnt + len(self._joint_names[comp.value])
            goal[comp] = JointState(position=action[j_cnt:end])
            j_cnt = end
        return goal

    def _action_check(self, action):
        """检查动作向量的维度是否正确"""
        expected_dim = sum(len(self._joint_names[comp.value]) for comp in self.config.components)
        if len(action) != expected_dim:
            raise ValueError(f"Action dimension mismatch: expected {expected_dim}, got {len(action)}")

    def enter_traj_mode(self):
        self.traj_mode = True

    def enter_servo_mode(self):
        self.traj_mode = False

    def on_switch_mode(self, mode):
        return True

    def _get_low_dim(self):
        data = {}
        robot_state = self.interface.get_robot_state()
        all_joints = robot_state.joint_state
        stamp = robot_state.joint_state.header.stamp
        # t = int((stamp.sec + stamp.nanosec * 1e-9)* 1000)
        t = int(stamp.sec * 1e9 + stamp.nanosec)
        # t = stamp.sec + stamp.nanosec * 1e-9
        for comp in self.config.components:
            comp_name = comp.value
            self._set_js_field(data, comp, t, all_joints)
            if comp == MMK2Components.BASE:
                base_pose = robot_state.base_state.pose
                base_vel = robot_state.base_state.velocity
                data_pose = [
                    base_pose.x,
                    base_pose.y,
                    base_pose.theta,
                ]
                # data[f"observation/{comp_name}/pose"] = data_pose
                data_vel = [
                    base_vel.x,
                    base_vel.y,
                    base_vel.omega,
                ]
                data[f"observation/{comp_name}/joint_state"] = {
                    "t": t,
                    "data": {
                        "position": data_pose,
                        "velocity": data_vel,
                        "effort": [0.0] * len(data_pose),
                    },
                }
        if self.config.demonstrate:
            for comp in [MMK2Components.LEFT_ARM, MMK2Components.RIGHT_ARM, MMK2Components.HEAD, MMK2Components.SPINE]:
                # print(f"[DEBUG] Processing component: {comp}, topic: {self._action_topics.get(comp)}")
                if comp in MMK2ComponentsGroup.ARMS:
                    arm_jn = self._joint_names[comp.value]
                    comp_eef = comp.value + "_eef"
                    eef_jn = self._joint_names[comp_eef]
                    js = self.interface.get_listened(self._action_topics[comp])
                    jq = self.interface.get_joint_values_by_names(js, arm_jn + eef_jn)
                    data[f"action/{comp.value}/joint_state"] = {
                        "t": t,
                        "data": {
                            "position": jq[:-1],
                            "velocity": [0.0] * len(arm_jn),
                            "effort": [0.0] * len(arm_jn),
                        },
                    }
                    data[f"action/{comp_eef}/joint_state"] = {
                        "t": t,
                        "data": {
                            "position": [jq[-1]],
                            "velocity": [0.0],
                            "effort": [0.0],
                        },
                    }

                if comp in MMK2ComponentsGroup.HEAD_SPINE:
                    # print(f"[DEBUG] HEAD_SPINE component: {comp}, topic: {self._action_topics.get(comp)}")
                    listened_data = self.interface.get_listened(self._action_topics[comp])
                    if listened_data and listened_data.data:  # 检查是否有数据
                        jq = list(listened_data.data)
                        data[f"action/{comp.value}/joint_state"] = {
                            "t": t,
                            "data": {
                                "position": jq,
                                "velocity": [0.0] * len(jq),
                                "effort": [0.0] * len(jq),
                            },
                        }
                    else:
                        print(f"[WARNING] No data received for component: {comp}")
        return data

    def _set_js_field(
        self, data: dict, comp: MMK2Components, t: float, js: JointState
    ):
        comp_data = {"t": t, "data": {}}
        for field in ["position", "velocity", "effort"]:
            value = self.interface.get_joint_values_by_names(
                js, self._joint_names[comp.value], field
            )
            comp_data["data"][field] = value
        data[f"observation/{comp.value}/joint_state"] = comp_data

    def _capture_images(self) -> Tuple[Dict[str, bytes], Dict[str, Time]]:
        images = {}
        img_stamps: Dict[MMK2Components, Time] = {}
        before_camread_t = time.perf_counter()
        comp_images = self.interface.get_image(self.cameras)
        for comp, image in comp_images.items():
            # TODO: now only support for color image
            images[comp.value] = image.data[ImageTypes.COLOR]
            img_stamps[comp.value] = image.stamp
        print(f"async_read_camera_{time.perf_counter() - before_camread_t}_dt_s")
        return images, img_stamps

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        # Capture images from cameras
        obs_act_dict = self._get_low_dim()
        images, img_stamps = self._capture_images()

        for name in images:
            stamp = img_stamps[name]
            # t = int((stamp.sec + stamp.nanosec * 1e-9) * 1e6)
            t = int(stamp.sec * 1e9 + stamp.nanosec)
            # print(f"[DEBUG] Image type for {name}: {type(images[name])}")  # 打印类型
            # print(f"[DEBUG] Image shape for {name}: {images[name].shape}")  # 打印形状
            # print(f"[DEBUG] Image dtype for {name}: {images[name].dtype}")  # 打印数据类型
            print(f"[DEBUG] Image stamp for {name}: {stamp}")  # 打印时间戳
            print(f"[DEBUG] Image time for {t}")  # 打印时间戳
            obs_act_dict[f"{name}/color/video"] = {
                # "t": time_ns(),
                "t": t,
                # "data": self.jpeg.encode(images[name]),
                "data": images[name],
            }
        return obs_act_dict

    def _check_joints(self, joint_names: List[str]):
        required_joints = []
        for component in MMK2ComponentsGroup.ARMS_EEFS + MMK2ComponentsGroup.HEAD_SPINE:
            required_joints.extend(self._joint_names[component.value])
        missing = [j for j in required_joints if j not in joint_names]
        if missing:
            raise KeyError(f"Missing required joints: {missing}")

    def shutdown(self) -> bool:
        self.interface.close()
        return True


if __name__ == "__main__":
    mmk = AIRBOTMMK(
        AIRBOTMMKConfig(
            ip="192.168.11.200",
            components=MMK2ComponentsGroup.ARMS_EEFS + MMK2ComponentsGroup.HEAD_SPINE,
            cameras={
                MMK2Components.HEAD_CAMERA: {
                    "camera_type": "REALSENSE",
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "false",
                },
                MMK2Components.LEFT_CAMERA: {
                    "camera_type": "USB",
                    "video_device": "/dev/left_camera",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25",
                },
                MMK2Components.RIGHT_CAMERA: {
                    "camera_type": "USB",
                    "video_device": "/dev/right_camera",
                    "image_width": "640",
                    "image_height": "480",
                    "framerate": "25",
                },
            },
        )
    )
    assert mmk.configure()
    for i in range(100000):
        mmk.capture_observation()
    mmk.shutdown()
