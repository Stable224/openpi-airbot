import dataclasses
import importlib.util
from pathlib import Path
import sys

from airbot_policy import AirbotInputs
from airbot_policy import AirbotOutputs
import flax.nnx as nnx
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0 as pi0
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class AirbotTaskConfig:
    task_name: str
    robot_type: str
    folders: list[str]
    state_topics: list[str]
    action_topics: list[str]
    camera_topics: dict[str, str]
    fps: int
    delta_action_mask: tuple[int, ...]
    base_model: str = "pi0"
    using_lora: bool = False
    exp_name: str = "default_exp"
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int = 5000
    over_write: bool = True
    resume: bool = False
    # wandb_enabled: bool = True
    log_method: str = "wandb"
    fsdp_devices: int = 1


@dataclasses.dataclass(frozen=True)
class LeRobotAirbotDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    """

    task_config: AirbotTaskConfig = dataclasses.field(
        default_factory=AirbotTaskConfig,
        metadata={"description": "The task configuration containing all the necessary parameters for the Airbot task."},
    )

    @override
    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).

        config = self.task_config

        repack_transform_dict = {}
        for camera in config.camera_topics:
            repack_transform_dict[f"observation/{camera}"] = f"{camera}"

        repack_transform_dict["observation/state"] = "state"
        repack_transform_dict["actions"] = "actions"
        repack_transform_dict["prompt"] = "prompt"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_transform_dict)])

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `airbot_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset.
        data_transforms = _transforms.Group(
            inputs=[AirbotInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[AirbotOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(*config.delta_action_mask)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def get_task_config(config_path: str) -> AirbotTaskConfig:
    """
    This function loads the task config from the given path. The config should be a Python file
    that defines the required attributes.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = config_path / "config.py"
        if not config_path.is_file():
            raise ValueError(f"Config path {config_path} does not exist or is not a file.")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return AirbotTaskConfig(
        task_name=config.TASK_NAME,
        robot_type=config.ROBOT_TYPE,
        folders=config.FOLDERS,
        state_topics=config.STATE_TOPICS,
        action_topics=config.ACTION_TOPICS,
        camera_topics=config.CAMERA_TOPICS,
        fps=config.FPS,
        delta_action_mask=tuple(config.DELTA_ACTION_MASK),
        base_model=getattr(config, "BASE_MODEL", "pi0"),
        using_lora=getattr(config, "USING_LORA", False),
        exp_name=getattr(config, "EXP_NAME", "default_exp"),
        batch_size=getattr(config, "BATCH_SIZE", 32),
        num_workers=getattr(config, "NUM_WORKERS", 2),
        num_train_steps=getattr(config, "NUM_TRAIN_STEPS", 30_000),
        log_interval=getattr(config, "LOG_INTERVAL", 100),
        save_interval=getattr(config, "SAVE_INTERVAL", 1000),
        keep_period=getattr(config, "KEEP_PERIOD", 5000),
        over_write=getattr(config, "OVER_WRITE", True),
        resume=getattr(config, "RESUME", False),
        # wandb_enabled=getattr(config, "WANDB_ENABLED", True),
        log_method=getattr(config, "LOG_METHOD", "wandb"),
        fsdp_devices=getattr(config, "FSDP_DEVICES", 1),
    )


def get_config(task_config) -> TrainConfig:
    model: _model.BaseModelConfig
    if task_config.base_model == "pi0":
        if task_config.using_lora:
            model = pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
        else:
            model = pi0.Pi0Config()
    elif task_config.base_model == "pi0-FAST":
        raise ValueError("pi0-FAST has not been implemented yet. Please use pi0 instead.")
    else:
        raise ValueError(f"Unknown BASE_MODEL: {task_config.base_model}. Please use 'pi0' or 'pi0-FAST'.")

    return TrainConfig(
        name=f"{task_config.robot_type}_{task_config.task_name}",
        exp_name=task_config.exp_name if hasattr(task_config, "exp_name") else "default_exp",
        overwrite=task_config.over_write if hasattr(task_config, "over_write") else True,
        model=model,
        data=LeRobotAirbotDataConfig(
            repo_id=task_config.task_name,
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            task_config=task_config,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        # TODO: hardcode to pi0 now
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),  # TODO
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        batch_size=task_config.batch_size,
        num_workers=task_config.num_workers,
        num_train_steps=task_config.num_train_steps,
        log_interval=task_config.log_interval,
        save_interval=task_config.save_interval,
        keep_period=task_config.keep_period,
        resume=task_config.resume,
        # wandb_enabled=task_config.log_method == "wandb",
        fsdp_devices=task_config.fsdp_devices,
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter()
        if task_config.using_lora
        else nnx.Nothing(),
        ema_decay=None if task_config.using_lora else 0.99,  # EMA is not used for LoRa models
    )
