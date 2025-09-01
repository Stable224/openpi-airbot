# NOTE: This is an example configuration file for Airbot. Move to data folder and rename it to config.py to use it.
TASK_NAME = "put_cup"
ROBOT_TYPE = "1-1"
FOLDERS = ["station1", "station2"]  # The folders contain MCAP files
# The following topics should be consistent with that in MCAP
STATE_TOPICS = ["/follow/arm/joint_state/position", "/follow/eef/joint_state/position"]
ACTION_TOPICS = ["/lead/arm/joint_state/position", "/lead/eef/joint_state/position"]
CAMERA_TOPICS = {
    # The key name can be only in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"] and the number should not larger than 3.
    "base_0_rgb": "/env_camera/color/image_raw",
    "left_wrist_0_rgb": "/follow_camera/color/image_raw",
}
FPS = 20
# Openpi requires all actions convert into delta actions except for the end effector.
# The value of DELTA_ACTION_MASK is a tuple, the example (6, -1) means the first 6 joints should be converted to
# delta actions, and the last joint (end effector) should not be converted.
DELTA_ACTION_MASK = (6, -1)

# Optional Configurations

# Model Config
BASE_MODEL = "pi0"  # only choose from pi0, pi0-FAST
USING_LORA = True  # whether to use LoRa
OVER_WRITE = True  # whether to overwrite the existing data
EXP_NAME = "test"  # experiment name
BATCH_SIZE = 32  # batch size for training
NUM_WORKERS = 2  # number of workers for data loading
NUM_TRAIN_STEPS = 30_000  # number of training steps
LOG_INTERVAL = 100  # interval for logging
SAVE_INTERVAL = 1000  # interval for saving checkpoints
KEEP_PERIOD = 5000  # period for keeping checkpoints
RESUME = False  # whether to resume training from the last checkpoint
LOG_METHOD = "wandb"  # logging method, can be "wandb" or "mlflow" or "none"
FSDP_DEVICES = 1  # number of devices for Fully Sharded Data Parallel (FSDP)
