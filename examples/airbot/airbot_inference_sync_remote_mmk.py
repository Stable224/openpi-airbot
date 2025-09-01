import logging
import time
from typing import Dict, List

from airbot_data_collection.basis import System, SystemMode
from airbot_data_collection.utils import init_logging
import numpy as np
from pydantic import BaseModel
import tyro

from openpi_client import websocket_client_policy

init_logging(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RemotePolicyConfig(BaseModel):
    """Configuration for the remote policy client.
    Args:
        host (str): Hostname or IP address of the policy server.
        port (int): Port number of the policy server.
    """
    # host: str = "localhost"
    host: str = "172.25.11.97"
    port: int = 8000


class MMKInferConfig(BaseModel):
    """Configuration for MMK robot remote inference.
    Args:
        policy_host (str): Hostname or IP address of the policy server.
        policy_port (int): Port number of the policy server.
        prompt (str): Task prompt for the model.
        max_steps (int): Maximum number of action publishing steps.
        step_rate (int): The rate at which to publish the actions.
        step_length (list[float]): Maximum change allowed for each joint per timestep.
        reset_action (list[float]): Initial action to reset the robot arm (17 DOF for MMK).
        interpolate (bool): Whether to interpolate the actions if the difference is too large.
        chunk_size_execute (int): Size of the action chunk to be executed at once.
        debug (bool): Enable debug logging.
    """
    # policy_host: str = "localhost"
    policy_host: str = "172.25.11.97"
    policy_port: int = 8000
    prompt: str = "stack block"
    max_steps: int = 250
    step_rate: int = 20
    step_length: list[float] = [   # 各关节最大步长限制 (17 DOF)
        # left_arm (6 joints)
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        # left_arm_eef (1 joint) 
        0.05,
        # right_arm (6 joints)
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        # right_arm_eef (1 joint)
        0.05,
        # head (2 joints)
        0.02, 0.02,
        # spine (1 joint)
        0.01,
    ]
    
    # MMK reset action (17 DOF: left_arm:6 + left_eef:1 + right_arm:6 + right_eef:1 + head:2 + spine:1)
    reset_action: list[float] = [
        -0.233, -0.73, 1.088, 1.774, -1.1475, -0.1606,    # left_arm (6 joints)
        0.0,                             # left_arm_eef (1 joint)
        0.2258, -0.6518, 0.9543, -1.777, 1.0615, 0.3588,    # right_arm (6 joints)
        0.0,                             # right_arm_eef (1 joint)
        0.0, -0.5,                       # head (2 joints)
        0.15,                            # spine (1 joint)
    ]
    
    interpolate: bool = False
    chunk_size_execute: int = 25
    debug: bool = False


class AutoConfig(BaseModel):
    """Auto configuration for observation handling."""
    chunk_size_predict: int = 0
    state_dim: int = 17  # MMK robot has 17 DOF
    camera_names: list[str] = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    observation: dict = {"qpos": None, "images": {}}


auto_config = AutoConfig()


def interpolate_action(step_length: List[float], prev_action: np.ndarray, cur_action: np.ndarray) -> np.ndarray:
    """Interpolate the actions to make the robot move smoothly."""
    steps = np.array(step_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


CAMERA_TOPICS = {
    "base_0_rgb": "head_camera/color/video",
    "left_wrist_0_rgb": "left_camera/color/video",
    "right_wrist_0_rgb": "right_camera/color/video",
}

def update_observation(operator: System) -> None:
    """Update the observation in auto_config."""
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    
    # 使用 CAMERA_TOPICS 映射来获取图像数据
    for camera_key, topic_name in CAMERA_TOPICS.items():
        if topic_name in obs and obs[topic_name] is not None:
            if isinstance(obs[topic_name], dict) and "data" in obs[topic_name]:
                image_dict[f"observation/{camera_key}"] = obs[topic_name]["data"]
                logger.debug(f"找到图像数据: {camera_key} -> {topic_name}")
            elif isinstance(obs[topic_name], np.ndarray):
                image_dict[f"observation/{camera_key}"] = obs[topic_name]
                logger.debug(f"找到图像数据: {camera_key} -> {topic_name}")
        else:
            logger.debug(f"未找到相机 {camera_key} 的图像数据 (topic: {topic_name})")
    
    # 如果没有获取到任何图像，创建虚拟图像数据
    if not image_dict:
        logger.warning("没有获取到真实图像数据，创建虚拟图像数据")
        for camera_key in CAMERA_TOPICS.keys():
            image_dict[f"observation/{camera_key}"] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    auto_config.observation = {"qpos": np.array(qpos), "images": image_dict}
    logger.debug(f"更新观察数据: qpos长度={len(qpos)}, 图像数量={len(image_dict)}, 图像键={list(image_dict.keys())}")


def inference_once(policy, prompt: str) -> np.ndarray:
    """Perform a single inference step using the remote policy."""
    obs = {
        "observation/state": auto_config.observation["qpos"],
        "prompt": prompt
    }
    obs.update(auto_config.observation["images"])

    logger.debug(f"发送到服务器的观察数据键: {list(obs.keys())}")
    logger.debug(f"状态维度: {obs['observation/state'].shape if hasattr(obs['observation/state'], 'shape') else len(obs['observation/state'])}")
    
    try:
        result = policy.infer(obs)
        action_chunk = result["actions"]
        auto_config.chunk_size_predict = action_chunk.shape[0]
        auto_config.state_dim = action_chunk.shape[1]
        logger.debug(f"推理成功: 动作块形状 {action_chunk.shape}")
        return action_chunk
    except Exception as e:
        logger.error(f"远程推理失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        # Return zero actions as fallback
        fallback_actions = np.zeros([64, 17], dtype=np.float32)
        logger.warning(f"使用备用零动作: {fallback_actions.shape}")
        return fallback_actions


def model_inference(config: MMKInferConfig, operator: System) -> None:
    """Main inference loop for MMK robot using remote policy."""
    auto_config.camera_names = operator.config.camera_names
    
    # Initialize remote policy client
    logger.info(f"Connecting to remote policy server at {config.policy_host}:{config.policy_port}")
    policy = websocket_client_policy.WebsocketClientPolicy(
        host=config.policy_host, 
        port=config.policy_port
    )
    
    if not config.prompt:
        raise ValueError("Prompt must be provided for remote policy inference.")
    
    logger.info(f"Using prompt: '{config.prompt}'")
    
    while True:
        # Initialize position
        logger.info("Resetting robot to initial position...")
        print(f"[DEBUG] reset_action: {config.reset_action}")
        operator.send_action(config.reset_action)
        
        user_input = input("按回车继续或输入 'q' 退出...")
        if user_input.lower() in {"q", "quit", "exit", "z"}:
            logger.info("退出中...")
            break
            
        logger.info("开始推理循环...")
        
        # Initialize the previous action to be the initial robot state
        pre_action = np.array(config.reset_action)
        update_observation(operator)
        t = 0
        
        while t < config.max_steps:
            # When coming to the end of the action chunk
            action_index = t % config.chunk_size_execute
            if action_index == 0:
                # Perform inference
                update_observation(operator)
                start_time = time.monotonic()
                logger.info("开始远程推理...")
                action_chunk = inference_once(policy, config.prompt).copy()
                inference_time = time.monotonic() - start_time
                logger.info(f"推理时间: {inference_time:.3f} 秒")

            action: np.ndarray = action_chunk[action_index]
            print(action)

            # Interpolate the original action sequence
            if config.interpolate:
                interp_actions = interpolate_action(config.step_length, pre_action, action)
            else:
                interp_actions = action[np.newaxis, :]
                
            # Execute the interpolated actions one by one
            for act in interp_actions:
                operator.send_action(act)
                time.sleep(1.0 / config.step_rate)
                
            t += 1
            pre_action = action.copy()
            
            if config.debug and t % 10 == 0:
                logger.info(f"执行步骤: {t}/{config.max_steps}")
    
    operator.shutdown()


def main():
    """Main entry point."""
    config = tyro.cli(MMKInferConfig, config=[tyro.conf.ConsolidateSubcommandArgs])
    
    # 启用调试日志
    if config.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import MMK robot operator
    from mmk_operator import Robot
    from airbot_mmk import AIRBOTMMKConfig
    from mmk2_types.types import MMK2Components
    
    # Create a simple MMK robot config
    class SimpleMMKConfig:
        def __init__(self):
            self.robot_type = "mmk"
            self.camera_names = ["head_camera", "left_camera", "right_camera"]
    
    simple_config = SimpleMMKConfig()
    
    logger.info("初始化 MMK 机器人...")
    robot = Robot(simple_config)
    
    # 测试连接到远程策略服务器
    logger.info(f"测试连接到策略服务器 {config.policy_host}:{config.policy_port}")
    try:
        test_policy = websocket_client_policy.WebsocketClientPolicy(
            host=config.policy_host, 
            port=config.policy_port
        )
        logger.info("策略服务器连接测试成功")
    except Exception as e:
        logger.error(f"无法连接到策略服务器: {e}")
        logger.error("请确保策略服务器正在运行")
        return
    
    try:
        model_inference(config, robot)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    except Exception as e:
        logger.error(f"推理过程中出现错误: {e}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main() 