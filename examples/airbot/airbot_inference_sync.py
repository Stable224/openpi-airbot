import logging
import time

from airbot_data_collection.basis import System
from airbot_data_collection.basis import SystemMode
from airbot_data_collection.utils import init_logging
from airbot_data_config import get_config
from airbot_data_config import get_task_config
import numpy as np
from pydantic import BaseModel
from robot_config import RobotConfig
import torch
import tyro

from openpi.policies.policy import Policy
from openpi.policies.policy_config import create_trained_policy

init_logging(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LocalPolicyConfig(BaseModel):
    """Configuration for the local policy client.
    Args:
        config_path (str): Path to the configuration file for the robot and task.
        checkpoint_dir (str): The directory containing the model checkpoints.
        max_steps (int): Maximum number of action publishing steps.
        step_rate (int): The rate at which to publish the actions.
        seed (int, optional): Random seed for reproducibility.
    """

    config_path: str
    checkpoint_dir: str
    seed: int = -1


class RemotePolicyConfig(BaseModel):
    """Configuration for the remote policy client.
    Args:
        host (str): Hostname or IP address of the policy server.
        port (int): Port number of the policy server.
    """

    host: str = "localhost"
    port: int = 8000


class InferConfig(BaseModel):
    """Configuration for the inference script.
    Args:
        config_path (str): Path to the configuration file for the robot and task.
        checkpoint_dir (str): The directory containing the model checkpoints.
        seed (int, optional): Random seed for reproducibility.
        max_steps (int): Maximum number of action publishing steps.
        step_rate (int): The rate at which to publish the actions.
        chunk_size (int): Size of the action chunk to be processed at once.
        step_length (list[float]): Maximum change allowed for each joint per timestep.
        reset_action (list[float]): Initial action to reset the robot arm.
        interpolate (bool): Whether to interpolate the actions if the difference is too large.
        camera_names (list[str]): Names of the cameras to capture images from.
        state_dim (int): The dimension of the state space, which is the number of joints in the robot arm.
        robot_type (str): Type of the robot, e.g., "play" or "mmk".
    """

    policy_config: LocalPolicyConfig | RemotePolicyConfig
    max_steps: int = 250
    step_rate: int = 20
    step_length: list[float] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05]

    reset_action: list[float] = [
        -0.001618136651813984,
        -1.0361113548278809,
        0.8421794176101685,
        1.6158959865570068,
        -0.6345375776290894,
        -1.6957406997680664,
        0.0,
        0.1323927342891693,
        -1.2208569049835205,
        1.0429750680923462,
        -2.0076663494110107,
        0.840582549571991,
        2.0390350818634033,
        0.0,
    ]
    interpolate: bool = False
    chunk_size_execute: int = 25
    debug: bool = False
    prompt: str = ""
    robot_config: RobotConfig


class AutoConfig(BaseModel):
    chunk_size_predict: int = 0
    state_dim: int = -1
    camera_names: list[str] = []
    observation: dict = {"qpos": None, "images": {}}


auto_config = AutoConfig()

config = tyro.cli(InferConfig, config=[tyro.conf.ConsolidateSubcommandArgs])
robot_config = config.robot_config
if robot_config.robot_type == "mmk":
    from mmk_operator import Robot
    from mmk_operator import RobotConfig
elif robot_config.robot_type == "play":
    from play_operator import Robot
    from play_operator import RobotConfig
else:
    raise ValueError("Unsupported robot type. Please use a valid config path for the robot.")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(step_length, prev_action, cur_action):
    steps = np.array(step_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


# Update the observation in auto_config
def update_observation(camera_names: list[str], operator: System):
    obs = operator.capture_observation()
    qpos = operator.get_qpos(obs)
    image_dict = {}
    for camera_name in camera_names:
        image_dict[f"observation/{camera_name}"] = obs[f"{camera_name}/color/image_raw"]["data"]
    auto_config.observation = {"qpos": np.array(qpos), "images": image_dict}


def inference_once(policy: Policy, prompt: str) -> np.ndarray:
    """Perform a single inference step using the trained policy."""
    obs = {"observation/state": auto_config.observation["qpos"], "prompt": prompt} | auto_config.observation["images"]

    action_chunk = policy.infer(obs)["actions"] if policy is not None else np.zeros([64, 7], dtype=np.float32)
    auto_config.chunk_size_predict = action_chunk.shape[0]
    auto_config.state_dim = action_chunk.shape[1]
    return action_chunk


def model_inference(config: InferConfig, operator: System):
    auto_config.camera_names = operator.config.camera_names
    policy_config = config.policy_config
    if isinstance(policy_config, LocalPolicyConfig):
        if policy_config.seed >= 0:
            set_seed(policy_config.seed)
        task_config = get_task_config(policy_config.config_path)
        policy = create_trained_policy(get_config(task_config), policy_config.checkpoint_dir)
        if not config.prompt:
            config.prompt = task_config.task_name
    else:
        from openpi_client import websocket_client_policy

        # Outside of episode loop, initialize the policy client.
        # Point to the host and port of the policy server (localhost and 8000 are the defaults).
        policy = websocket_client_policy.WebsocketClientPolicy(host=policy_config.host, port=policy_config.port)
        assert config.prompt, "Prompt must be provided for remote policy inference."

    while True:
        # Initialize position
        operator.switch_mode(SystemMode.RESETTING)
        operator.send_action(config.reset_action)
        if input("Press 'Enter' to continue or 'q' and 'Enter' to quit...") in {"q", "Q", "z", "Z"}:
            logger.info("Quitting...")
            break
        operator.switch_mode(SystemMode.SAMPLING)
        # Initialize the previous action to be the initial robot state
        with torch.inference_mode():
            pre_action = np.array(config.reset_action)
            update_observation(auto_config.camera_names, operator)
            t = 0
            while t < config.max_steps:
                # When coming to the end of the action chunk
                action_index = t % config.chunk_size_execute
                if action_index == 0:
                    # infer once
                    update_observation(auto_config.camera_names, operator)
                    start_time = time.monotonic()
                    logger.info("Start inference...")
                    action_chunk = inference_once(policy, config.prompt).copy()
                    logger.info(f"Inference time: {time.monotonic() - start_time} s")

                action: np.ndarray = action_chunk[action_index]

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
    operator.shutdown()


def main():
    model_inference(config, Robot(config.robot_config))


if __name__ == "__main__":
    main()
