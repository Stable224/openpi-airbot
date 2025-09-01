import dataclasses
import logging
import socket

from airbot_data_config import get_config
from airbot_data_config import get_task_config
import tyro

from openpi.policies.policy import Policy
from openpi.policies.policy import PolicyRecorder
from openpi.policies.policy_config import create_trained_policy
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config_path: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


def create_policy(args: Args) -> Policy:
    """Create a policy from the given arguments."""
    return create_trained_policy(
        get_config(get_task_config(args.policy.config_path)), args.policy.dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
