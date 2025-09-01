"""
Our minimal script for converting AIRBOT dataset from MCAP to LeRobot format.

The script is modified from examples/libero/convert_libero_data_to_lerobot.py

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:

Running this conversion script will take approximately 30 minutes.
"""

import os
from pathlib import Path
import shutil
import tempfile

from airbot_data_config import get_task_config
import cv2
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
import numpy as np
import tyro

from openpi.airbot_type.FloatArray import FloatArray


class McapConverter:
    """
    A class to convert MCAP dataset to LeRobot format.
    """

    def __init__(self, data_dir: str):
        # Validate data_dir
        if not Path(data_dir).is_dir():
            raise ValueError(f"data_dir '{data_dir}' is not a valid directory")
        self.data_dir = data_dir

        # 2. Locate the Python-format config file
        config_path = Path(self.data_dir) / "config.py"
        if not config_path.is_file():
            raise ValueError(f"Config file 'config.py' not found in '{data_dir}'.")

        config = get_task_config(config_path)

        self.task_name = config.task_name
        self.robot_type = config.robot_type
        self.folders = config.folders
        self.state_topics = config.state_topics
        self.action_topics = config.action_topics
        self.camera_topics: dict[str:str] = config.camera_topics
        self.fps = config.fps

        # 5. Scan each folder, verify existence, and count .mcap files
        self.folder_file_counts = {}
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            if not folder_path.is_dir():
                raise ValueError(f"Configured folder '{folder}' not found in '{self.data_dir}'.")

            # Count MCAP files in this folder
            mcap_files = [
                f for f in os.listdir(folder_path) if Path(folder_path / f).is_file() and f.lower().endswith(".mcap")
            ]
            self.folder_file_counts[folder] = len(mcap_files)

        print(f"Found {sum(self.folder_file_counts.values())} MCAP files across {len(self.folders)} folders.")

        # read first mcap file for many infos
        self.schemas = {}
        self.schemas["airbot_fbs.FloatArray"] = FloatArray.GetRootAsFloatArray
        self.state_length = 0
        self.action_length = 0
        self.camera_image_shape = {}
        
        # Find the first mcap file to read schema and camera information
        mcap_file_path = None
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(".mcap"):
                        mcap_file_path = Path(root) / filename
                        break
                if mcap_file_path:
                    break
        
        if mcap_file_path is None:
            raise ValueError("No MCAP files found in any of the configured folders.")
        
        print(f"Using {mcap_file_path} to read schema and camera information.")
        
        with mcap_file_path.open("rb") as f:
            reader = make_reader(f)

            # Read camera image shape from the valid mcap file
            cam_attachment_path = {}
            for attach in reader.iter_attachments():
                media_type = attach.media_type
                if media_type == "video/mp4" and attach.name in self.camera_topics.values():
                    cam_attachment_path[attach.name] = self._save_temporary_video(attach.data)
            
            for camera_name, topic in self.camera_topics.items():
                if topic in cam_attachment_path:
                    frame = self._get_frame_image(cam_attachment_path[topic], 0)
                    self.camera_image_shape[camera_name] = frame.shape
                else:
                    raise ValueError(f"Camera attachment for {camera_name} not found in {mcap_file_path}.")
            
            print(f"Camera image shapes: {self.camera_image_shape}")
            
            is_read_topics = {topic: False for topic in self.state_topics + self.action_topics}
            for schema_obj, channel_obj, message_obj in reader.iter_messages(
                topics=self.state_topics + self.action_topics
            ):
                if is_read_topics[channel_obj.topic]:
                    break
                is_read_topics[channel_obj.topic] = True
                if schema_obj.name not in self.schemas:
                    raise ValueError(f"Schema '{schema_obj.name}' not found in schemas.")
                if channel_obj.topic in self.state_topics:
                    self.state_length += len(self.schemas[schema_obj.name](message_obj.data).ValuesAsNumpy())
                elif channel_obj.topic in self.action_topics:
                    self.action_length += len(self.schemas[schema_obj.name](message_obj.data).ValuesAsNumpy())
        
        print(f"State length: {self.state_length}, Action length: {self.action_length}")

    def create_dataset(self) -> LeRobotDataset:
        """
        Placeholder for dataset creation logic.
        This method should implement the logic to convert MCAP files to LeRobot format.
        """
        features = {}
        for camera_name in self.camera_topics:
            features[camera_name] = {
                "dtype": "image",
                "shape": self.camera_image_shape[camera_name],
                "names": ["height", "width", "channel"],
            }
        features["state"] = {
            "dtype": "float32",
            "shape": (self.state_length,),
            "names": ["state"],
        }
        features["actions"] = {
            "dtype": "float32",
            "shape": (self.action_length,),
            "names": ["actions"],
        }
        print(f"Creating LeRobot dataset with features: {features}")

        return LeRobotDataset.create(
            repo_id=self.task_name,
            robot_type=self.robot_type,
            fps=self.fps,
            features=features,
            image_writer_threads=10,  # TODO: add config for image writer threads
            image_writer_processes=5,  # TODO: add config for image writer processes
        )

    def validate_metadata(self, mcap_file_path: Path) -> bool:
        """
        Validate the metadata of a specific MCAP file.
        This method should implement the logic to check if the metadata is consistent with the expected format.
        """
        try:
            with mcap_file_path.open("rb") as f:
                reader = make_reader(f)
                for md in reader.iter_metadata():
                    if md.name == "task_info":
                        task_name = md.metadata.get("task_name", "N/A")
                        if task_name != self.task_name:
                            print(f"Task name mismatch: expected {self.task_name}, got {task_name}")
                            return False
            return True
        except Exception as e:
            print(f"Error reading MCAP file {mcap_file_path}: {e}")
            return False

    def _save_temporary_video(self, bytes_data: bytes) -> str:
        """
        Save a temporary video file from the MCAP attachment.
        This method is used to handle camera attachments in the MCAP files.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_file.flush()
        return tmp_file.name

    def _get_frame_image(self, path: str, frame_index: int) -> np.array:
        """
        Get a specific frame image from a video file.
        This method is used to extract images from camera attachments in the MCAP files.
        """
        cap = cv2.VideoCapture(path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If requested frame index is beyond video length, use the last frame
        if frame_index >= total_frames:
            original_frame_index = frame_index
            frame_index = total_frames - 1
            print(f"Warning: Requested frame {original_frame_index} beyond video length ({total_frames}), using last frame ({frame_index})")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # If still can't read, try the first frame
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError(f"Could not read any frame from video {path}")
            else:
                print(f"Warning: Could not read frame {frame_index}, using frame 0 instead")
        
        return frame

    def load_to_dataset(self, dataset: LeRobotDataset):
        """
        Load the MCAP files into the LeRobot dataset.
        This method should implement the logic to read the MCAP files and convert them to the LeRobot format.
        """
        # Loop over mcap datasets and write episodes to the LeRobot dataset
        for folder_name, file_count in self.folder_file_counts.items():
            print(f"Processing folder: {folder_name} with {file_count} files")

            # for file_index in range(1, file_count + 1):
            #     mcap_file_path = Path(self.data_dir) / folder_name / f"{file_index}.mcap"
            for root, _, files in os.walk(Path(self.data_dir) / folder_name):
                for filename in files:
                    if not filename.lower().endswith(".mcap"):
                        continue
                    mcap_file_path = Path(root) / filename
                    # Validate metadata for the current file
                    if not self.validate_metadata(mcap_file_path):
                        print(f"Metadata validation failed for {folder_name} file {filename}. Skipping this file.")
                        continue

                    # Here you would load the MCAP file and convert it to LeRobot format
                    # This is a placeholder for the actual conversion logic
                    print(f"Loading and converting {folder_name} file {filename} to LeRobot format...")
                    with mcap_file_path.open("rb") as f:
                        reader = make_reader(f)
                        cnt_topics = {topic: 0 for topic in self.state_topics + self.action_topics}
                        state_msg = {}
                        action_msg = {}
                        cnt = 0
                        cam_attachment_path = {}
                        for attach in reader.iter_attachments():
                            media_type = attach.media_type
                            if media_type == "video/mp4" and attach.name in self.camera_topics.values():
                                cam_attachment_path[attach.name] = self._save_temporary_video(attach.data)
                        # print(f"Camera attachments saved: {cam_attachment_path}")
                        if not cam_attachment_path or len(cam_attachment_path) != len(self.camera_topics):
                            print(
                                f"Warning: Not all camera attachments were found in {mcap_file_path}. Expected {len(self.camera_topics)}, found {len(cam_attachment_path)}."
                            )
                            continue
                        for schema_obj, channel_obj, message_obj in reader.iter_messages(
                            topics=self.state_topics + self.action_topics
                        ):
                            cnt_topics[channel_obj.topic] += 1
                            if cnt_topics[channel_obj.topic] - cnt == 2:
                                self._save_frame(dataset, state_msg, action_msg, cam_attachment_path, cnt)
                                cnt += 1
                            if channel_obj.topic in self.state_topics:
                                state_msg[channel_obj.topic] = self.schemas[schema_obj.name](
                                    message_obj.data
                                ).ValuesAsNumpy()
                            if channel_obj.topic in self.action_topics:
                                action_msg[channel_obj.topic] = self.schemas[schema_obj.name](
                                    message_obj.data
                                ).ValuesAsNumpy()
                        # save last frame
                        self._save_frame(dataset, state_msg, action_msg, cam_attachment_path, cnt)
                        dataset.save_episode()
                        
                        # Clean up temporary video files
                        for temp_path in cam_attachment_path.values():
                            try:
                                os.unlink(temp_path)
                            except Exception as e:
                                print(f"Warning: Could not delete temporary file {temp_path}: {e}")

    def _save_frame(
        self, dataset: LeRobotDataset, state_msg: dict, action_msg: dict, cam_attachment_path: dict, cnt: int
    ):
        """
        Save a frame to the dataset.
        This method is used to handle the saving of frames after processing all messages in an episode.
        """
        # Check if all required state and action topics have messages
        for topic in self.state_topics:
            if topic not in state_msg:
                print(f"Warning: State topic {topic} not found in messages. Skipping frame {cnt}.")
                return
        for topic in self.action_topics:
            if topic not in action_msg:
                print(f"Warning: Action topic {topic} not found in messages. Skipping frame {cnt}.")
                return
        
        state_vec = np.array([])
        action_vec = np.array([])
        for topic in self.state_topics:
            state_vec = np.concatenate((state_vec, state_msg[topic]))
        for topic in self.action_topics:
            action_vec = np.concatenate((action_vec, action_msg[topic]))
        frame = {}
        for camera_name, topic in self.camera_topics.items():
            if topic in cam_attachment_path:
                frame[camera_name] = self._get_frame_image(cam_attachment_path[topic], cnt)
            else:
                print(f"Warning: Camera attachment for {camera_name} not found.")
        frame["state"] = state_vec.astype(np.float32)
        frame["actions"] = action_vec.astype(np.float32)
        frame["task"] = self.task_name
        dataset.add_frame(frame)


def main(data_dir: str):
    mcap_converter = McapConverter(data_dir)

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / mcap_converter.task_name
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = mcap_converter.create_dataset()
    mcap_converter.load_to_dataset(dataset)


if __name__ == "__main__":
    tyro.cli(main)
