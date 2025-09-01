import argparse
from pathlib import Path


def parse_camera_topics(pairs):
    """Parse list of key=value strings into a dict."""
    topics = {}
    # for pair in pairs:
    if "=" not in pairs:
        raise argparse.ArgumentTypeError(f"Invalid camera topic format: '{pairs}'. Use key=topic_path.")
    key, value = pairs.split("=", 1)
    if key not in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        raise argparse.ArgumentTypeError(
            f"Invalid camera topic key: '{key}'. Must be one of ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']."
        )
    topics[key] = value
    return topics


def camera_topics_to_dict(camera_topics):
    """Convert a list of camera topics to a dictionary."""
    topics_dict = {}
    for topic in camera_topics:
        # topics_dict[topic.keys] = topic.values
        for key, value in topic.items():
            if key in topics_dict:
                raise ValueError(f"Duplicate camera topic key: '{key}'. Each key must be unique.")
            topics_dict[key] = value
    return topics_dict


def main():
    parser = argparse.ArgumentParser(description="Generate a configuration file for the robot experiment.")
    # Required arguments
    parser.add_argument("--output", type=str, default="config.py", help="Path to the generated config file")
    parser.add_argument("--task-name", type=str, required=True, help="Name of the task (e.g., put_cup)")
    parser.add_argument("--robot-type", type=str, required=True, help="Type of the robot (e.g., 1-1)")
    parser.add_argument("--folders", type=str, nargs="+", required=True, help="List of folder names (space-separated)")
    parser.add_argument(
        "--state-topics", type=str, nargs="+", required=True, help="List of state topic names (space-separated)"
    )
    parser.add_argument(
        "--action-topics", type=str, nargs="+", required=True, help="List of action topic names (space-separated)"
    )
    parser.add_argument(
        "--camera-topics",
        type=parse_camera_topics,
        nargs="+",
        required=True,
        help="Camera topics in format key=topic_path (e.g., base_0_rgb=/env_camera/color/image_raw)",
    )
    parser.add_argument("--fps", type=int, required=True, help="Frames per second (e.g., 20)")
    parser.add_argument(
        "--delta-action-mask",
        type=int,
        nargs=2,
        required=True,
        metavar=("START", "END"),
        help="Delta action mask as two integers (e.g., 6 -1)",
    )

    # Optional arguments with defaults matching example
    parser.add_argument(
        "--base-model",
        type=str,
        default="pi0",
        choices=["pi0", "pi0-FAST"],
        help="Base model, choose from pi0 or pi0-FAST",
    )
    parser.add_argument("--using-lora", action="store_true", help="Flag to use LoRa (sets USING_LORA=True)")
    parser.add_argument(
        "--overwrite", action="store_true", help="Flag to overwrite existing data (sets OVER_WRITE=True)"
    )
    parser.add_argument("--exp-name", type=str, default="test", help="Experiment name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--num-train-steps", type=int, default=30000, help="Number of training steps")
    parser.add_argument("--log-interval", type=int, default=100, help="Interval for logging")
    parser.add_argument("--save-interval", type=int, default=1000, help="Interval for saving checkpoints")
    parser.add_argument("--keep-period", type=int, default=5000, help="Period for keeping checkpoints")
    parser.add_argument("--resume", action="store_true", help="Flag to resume training from the last checkpoint")
    parser.add_argument(
        "--log-method",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow"],
        help="Logging method, can be wandb or mlflow",
    )
    parser.add_argument("--fsdp-devices", type=int, default=1, help="Number of devices for FSDP")

    args = parser.parse_args()

    # Build config dictionary
    config_vars = {
        "TASK_NAME": args.task_name,
        "ROBOT_TYPE": args.robot_type,
        "FOLDERS": args.folders,
        "STATE_TOPICS": args.state_topics,
        "ACTION_TOPICS": args.action_topics,
        # 'CAMERA_TOPICS': args.camera_topics,
        "CAMERA_TOPICS": camera_topics_to_dict(args.camera_topics),
        "FPS": args.fps,
        "DELTA_ACTION_MASK": tuple(args.delta_action_mask),
        "BASE_MODEL": args.base_model,
        "USING_LORA": args.using_lora,
        "OVER_WRITE": args.overwrite,
        "EXP_NAME": args.exp_name,
        "BATCH_SIZE": args.batch_size,
        "NUM_WORKERS": args.num_workers,
        "NUM_TRAIN_STEPS": args.num_train_steps,
        "LOG_INTERVAL": args.log_interval,
        "SAVE_INTERVAL": args.save_interval,
        "KEEP_PERIOD": args.keep_period,
        "RESUME": args.resume,
        "LOG_METHOD": args.log_method,
        "FSDP_DEVICES": args.fsdp_devices,
    }

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for key, value in config_vars.items():
            # For strings, wrap in quotes; else use repr
            line = f"{key} = {value!r}\n"
            f.write(line)

    print(f"Configuration file generated at: {output_path}")


if __name__ == "__main__":
    main()
