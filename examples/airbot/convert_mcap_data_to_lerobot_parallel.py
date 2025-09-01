"""
并行版本的MCAP到LeRobot格式转换脚本

This is a parallel version of the MCAP conversion script that uses multiprocessing 
to significantly speed up the conversion process.

Usage:
uv run examples/airbot/convert_mcap_data_to_lerobot_parallel.py --data_dir /path/to/your/data --num_workers 8

Running this conversion script will be much faster than the sequential version.
"""

import os
from pathlib import Path
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import Manager
import time
from typing import Dict, List, Tuple, Any
import signal
import subprocess

from airbot_data_config import get_task_config
import cv2
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import make_reader
import numpy as np
import tyro

from openpi.airbot_type.FloatArray import FloatArray


def process_single_mcap(args: Tuple[str, str, Dict[str, Any], int]) -> Dict[str, Any]:
    """
    处理单个MCAP文件的函数，用于多进程处理
    
    Args:
        args: (mcap_file_path, data_dir, config_dict, file_index)
        
    Returns:
        Dict containing processed episode data or error info
    """
    mcap_file_path, data_dir, config_dict, file_index = args
    
    try:
        # 重建配置对象
        class Config:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        config = Config(config_dict)
        schemas = {"airbot_fbs.FloatArray": FloatArray.GetRootAsFloatArray}
        
        mcap_path = Path(mcap_file_path)
        
        print(f"[Worker {os.getpid()}] Processing {mcap_path.name}...")
        
        # 增加文件存在性检查
        if not mcap_path.exists():
            return {"success": False, "file": mcap_path.name, "error": "File does not exist"}
        
        try:
            with mcap_path.open("rb") as f:
                reader = make_reader(f)
                
                # 验证元数据
                metadata_valid = True
                try:
                    for md in reader.iter_metadata():
                        if md.name == "task_info":
                            task_name = md.metadata.get("task_name", "N/A")
                            if task_name != config.task_name:
                                print(f"Task name mismatch in {mcap_path.name}: expected {config.task_name}, got {task_name}")
                                metadata_valid = False
                                break
                except Exception as e:
                    print(f"Error validating metadata for {mcap_path.name}: {e}")
                    metadata_valid = False
                
                if not metadata_valid:
                    return {"success": False, "file": mcap_path.name, "error": "Metadata validation failed"}
                
                # 处理相机附件
                cam_attachment_path = {}
                try:
                    for attach in reader.iter_attachments():
                        if attach.media_type == "video/mp4" and attach.name in config.camera_topics.values():
                            try:
                                temp_path = _save_temporary_video(attach.data)
                                cam_attachment_path[attach.name] = temp_path
                            except Exception as e:
                                print(f"Error processing video attachment {attach.name} in {mcap_path.name}: {e}")
                                # 清理已创建的临时文件
                                for temp_path in cam_attachment_path.values():
                                    try:
                                        os.unlink(temp_path)
                                    except:
                                        pass
                                return {"success": False, "file": mcap_path.name, "error": f"Invalid video attachment: {attach.name}"}
                except Exception as e:
                    # 清理已创建的临时文件
                    for temp_path in cam_attachment_path.values():
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    return {"success": False, "file": mcap_path.name, "error": f"Error processing attachments: {e}"}
                
                if len(cam_attachment_path) != len(config.camera_topics):
                    # 清理临时文件
                    for temp_path in cam_attachment_path.values():
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    return {"success": False, "file": mcap_path.name, "error": f"Camera attachments mismatch: expected {len(config.camera_topics)}, found {len(cam_attachment_path)}"}
                
                # 处理消息数据
                cnt_topics = {topic: 0 for topic in config.state_topics + config.action_topics}
                episode_data = []
                state_msg = {}
                action_msg = {}
                cnt = 0
                max_frames_to_process = 1000  # 限制最大处理帧数，避免处理超大文件
                
                try:
                    for schema_obj, channel_obj, message_obj in reader.iter_messages(
                        topics=config.state_topics + config.action_topics
                    ):
                        try:
                            cnt_topics[channel_obj.topic] += 1
                            if cnt_topics[channel_obj.topic] - cnt == 2:
                                # 检查是否超过最大帧数限制
                                if cnt >= max_frames_to_process:
                                    print(f"Warning: {mcap_path.name} has too many frames (>{max_frames_to_process}), truncating")
                                    break
                                    
                                # 保存帧数据
                                frame_data = _create_frame_data(config, schemas, state_msg, action_msg, cam_attachment_path, cnt)
                                if frame_data:
                                    episode_data.append(frame_data)
                                    # 重置失败计数
                                    _create_frame_data._consecutive_failures = 0
                                else:
                                    # 如果连续多次失败，直接放弃这个文件
                                    consecutive_failures = getattr(_create_frame_data, '_consecutive_failures', 0) + 1
                                    if consecutive_failures >= 5:
                                        print(f"Too many consecutive frame failures in {mcap_path.name}, skipping file")
                                        raise ValueError("Too many frame creation failures")
                                    _create_frame_data._consecutive_failures = consecutive_failures
                                cnt += 1
                            
                            if channel_obj.topic in config.state_topics:
                                state_msg[channel_obj.topic] = schemas[schema_obj.name](message_obj.data).ValuesAsNumpy()
                            elif channel_obj.topic in config.action_topics:
                                action_msg[channel_obj.topic] = schemas[schema_obj.name](message_obj.data).ValuesAsNumpy()
                        except Exception as e:
                            print(f"Error processing message in {mcap_path.name} at frame {cnt}: {e}")
                            continue  # 跳过这个消息，继续处理下一个
                
                    # 保存最后一帧（如果没有超过限制）
                    if cnt < max_frames_to_process:
                        try:
                            frame_data = _create_frame_data(config, schemas, state_msg, action_msg, cam_attachment_path, cnt)
                            if frame_data:
                                episode_data.append(frame_data)
                        except Exception as e:
                            print(f"Error creating final frame for {mcap_path.name}: {e}")
                
                except Exception as e:
                    # 清理临时文件
                    for temp_path in cam_attachment_path.values():
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    return {"success": False, "file": mcap_path.name, "error": f"Error processing messages: {e}"}
                
                # 清理临时视频文件
                for temp_path in cam_attachment_path.values():
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                if len(episode_data) == 0:
                    return {"success": False, "file": mcap_path.name, "error": "No valid frames generated"}
                
                return {
                    "success": True,
                    "file": mcap_path.name,
                    "episode_data": episode_data,
                    "frame_count": len(episode_data)
                }
        
        except Exception as e:
            return {"success": False, "file": mcap_path.name, "error": f"Error opening/reading MCAP file: {e}"}
            
    except Exception as e:
        return {"success": False, "file": mcap_file_path, "error": f"Unexpected error: {e}"}


def _check_video_validity(video_path: str, max_check_frames: int = 3) -> bool:
    """
    快速检查视频文件是否有效，使用更严格的检查
    
    Args:
        video_path: 视频文件路径
        max_check_frames: 最多检查的帧数
        
    Returns:
        bool: 视频是否有效
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # 检查基本属性
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            return False
            
        # 检查文件格式 - 如果是明显损坏的mp4文件，直接返回False
        try:
            # 使用ffprobe快速检查视频文件完整性
            result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', video_path], 
                                  capture_output=True, timeout=10)
            if result.returncode != 0:
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # 如果ffprobe不可用或超时，继续使用OpenCV检查，但更加保守
            pass
        
        # 只检查前几帧，避免读取可能损坏的后续帧
        valid_frames = 0
        for i in range(min(max_check_frames, min(total_frames, 5))):  # 最多检查5帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                valid_frames += 1
            else:
                break
                
        # 要求至少读取到2帧才认为视频有效
        return valid_frames >= min(2, max_check_frames)
        
    except Exception:
        return False
    finally:
        if cap is not None:
            cap.release()


def _save_temporary_video(bytes_data: bytes) -> str:
    """保存临时视频文件并验证有效性"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_file.flush()
            temp_path = tmp_file.name
        
        # 严格验证视频文件有效性
        if not _check_video_validity(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
            raise ValueError("Video file failed validation check")
            
        return temp_path
    except Exception as e:
        raise ValueError(f"Failed to save or validate temporary video: {e}")


def _get_frame_image_safe(path: str, frame_index: int, timeout_seconds: int = 5) -> np.array:
    """
    安全地从视频文件中获取指定帧的图像，使用更短的超时时间
    """
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")
        
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Invalid video file {path}: no frames found")
        
        # 如果请求的帧超出视频长度，使用最后一帧
        effective_frame_index = min(frame_index, total_frames - 1)
        
        # 尝试读取指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, effective_frame_index)
        ret, frame = cap.read()
        
        if not ret or frame is None or frame.size == 0:
            raise ValueError(f"Could not read frame {effective_frame_index} from video {path}")
            
        return frame
    
    except Exception as e:
        raise ValueError(f"Error reading frame {frame_index} from {path}: {e}")
    
    finally:
        if cap is not None:
            cap.release()


def _create_frame_data(config, schemas, state_msg: dict, action_msg: dict, cam_attachment_path: dict, cnt: int) -> dict:
    """创建帧数据，增强错误处理和快速失败机制"""
    try:
        # 构建状态向量
        state_vec = np.array([])
        for topic in config.state_topics:
            if topic in state_msg:
                try:
                    data = state_msg[topic]
                    if data is None or len(data) == 0:
                        print(f"Warning: Empty state data for topic {topic} at frame {cnt}")
                        return None
                    state_vec = np.concatenate((state_vec, data))
                except Exception as e:
                    print(f"Error concatenating state for topic {topic} at frame {cnt}: {e}")
                    return None
        
        # 构建动作向量
        action_vec = np.array([])
        for topic in config.action_topics:
            if topic in action_msg:
                try:
                    data = action_msg[topic]
                    if data is None or len(data) == 0:
                        print(f"Warning: Empty action data for topic {topic} at frame {cnt}")
                        return None
                    action_vec = np.concatenate((action_vec, data))
                except Exception as e:
                    print(f"Error concatenating action for topic {topic} at frame {cnt}: {e}")
                    return None
        
        # 验证状态和动作向量
        if len(state_vec) == 0 or len(action_vec) == 0:
            print(f"Warning: Empty state or action vector at frame {cnt}")
            return None
        
        # 获取相机图像 - 使用快速失败策略
        frame = {}
        for camera_name, topic in config.camera_topics.items():
            if topic in cam_attachment_path:
                try:
                    # 使用更短的超时时间，快速失败
                    frame[camera_name] = _get_frame_image_safe(cam_attachment_path[topic], cnt, timeout_seconds=3)
                except Exception as e:
                    print(f"Failed to read frame {cnt} from camera {camera_name}: {e}")
                    # 不再尝试fallback，直接失败
                    return None
            else:
                print(f"Camera topic {topic} not found in attachments")
                return None
        
        # 验证所有相机图像都已成功获取
        if len(frame) != len(config.camera_topics):
            print(f"Missing camera images at frame {cnt}: expected {len(config.camera_topics)}, got {len(frame)}")
            return None
        
        frame["state"] = state_vec.astype(np.float32)
        frame["actions"] = action_vec.astype(np.float32)
        frame["task"] = config.task_name
        
        return frame
    except Exception as e:
        print(f"Unexpected error creating frame data at frame {cnt}: {e}")
        return None


class ParallelMcapConverter:
    """
    并行MCAP转换器类
    """

    def __init__(self, data_dir: str, num_workers: int = 4):
        # 验证数据目录
        if not Path(data_dir).is_dir():
            raise ValueError(f"data_dir '{data_dir}' is not a valid directory")
        self.data_dir = data_dir
        self.num_workers = num_workers

        # 加载配置文件
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

        # 将配置转换为字典，用于多进程传递
        self.config_dict = {
            'task_name': self.task_name,
            'robot_type': self.robot_type,
            'folders': self.folders,
            'state_topics': self.state_topics,
            'action_topics': self.action_topics,
            'camera_topics': self.camera_topics,
            'fps': self.fps
        }

        # 扫描MCAP文件
        self.mcap_files = []
        for folder in self.folders:
            folder_path = Path(self.data_dir) / folder
            if not folder_path.is_dir():
                raise ValueError(f"Configured folder '{folder}' not found in '{self.data_dir}'.")
            
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith('.mcap'):
                        mcap_file_path = Path(root) / filename
                        self.mcap_files.append(str(mcap_file_path))

        print(f"Found {len(self.mcap_files)} MCAP files to process with {self.num_workers} workers.")

        # 读取第一个有效文件获取图像形状等信息
        self._initialize_from_first_file()

    def _initialize_from_first_file(self):
        """从第一个有效文件初始化图像形状等信息"""
        schemas = {"airbot_fbs.FloatArray": FloatArray.GetRootAsFloatArray}
        
        for mcap_file_path in self.mcap_files:
            try:
                with Path(mcap_file_path).open("rb") as f:
                    reader = make_reader(f)
                    
                    # 读取相机图像形状
                    cam_attachment_path = {}
                    for attach in reader.iter_attachments():
                        if attach.media_type == "video/mp4" and attach.name in self.camera_topics.values():
                            cam_attachment_path[attach.name] = _save_temporary_video(attach.data)
                    
                    if len(cam_attachment_path) != len(self.camera_topics):
                        # 清理临时文件
                        for temp_path in cam_attachment_path.values():
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                        continue
                    
                    # 获取图像形状
                    self.camera_image_shape = {}
                    for camera_name, topic in self.camera_topics.items():
                        if topic in cam_attachment_path:
                            try:
                                frame = _get_frame_image_safe(cam_attachment_path[topic], 0)
                                self.camera_image_shape[camera_name] = frame.shape
                            except Exception as e:
                                print(f"Error reading sample frame from {camera_name}: {e}")
                                # 清理临时文件
                                for temp_path in cam_attachment_path.values():
                                    try:
                                        os.unlink(temp_path)
                                    except:
                                        pass
                                continue
                    
                    # 计算状态和动作长度
                    self.state_length = 0
                    self.action_length = 0
                    is_read_topics = {topic: False for topic in self.state_topics + self.action_topics}
                    
                    for schema_obj, channel_obj, message_obj in reader.iter_messages(
                        topics=self.state_topics + self.action_topics
                    ):
                        if is_read_topics[channel_obj.topic]:
                            continue
                        is_read_topics[channel_obj.topic] = True
                        
                        if channel_obj.topic in self.state_topics:
                            self.state_length += len(schemas[schema_obj.name](message_obj.data).ValuesAsNumpy())
                        elif channel_obj.topic in self.action_topics:
                            self.action_length += len(schemas[schema_obj.name](message_obj.data).ValuesAsNumpy())
                        
                        if all(is_read_topics.values()):
                            break
                    
                    # 清理临时文件
                    for temp_path in cam_attachment_path.values():
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    
                    print(f"Initialized from {Path(mcap_file_path).name}")
                    print(f"Camera image shapes: {self.camera_image_shape}")
                    print(f"State length: {self.state_length}, Action length: {self.action_length}")
                    return
                    
            except Exception as e:
                print(f"Failed to initialize from {mcap_file_path}: {e}")
                continue
        
        raise ValueError("No valid MCAP file found for initialization")

    def create_dataset(self) -> LeRobotDataset:
        """创建LeRobot数据集"""
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
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def load_to_dataset_parallel(self, dataset: LeRobotDataset):
        """并行加载MCAP文件到数据集"""
        print(f"\n开始并行处理 {len(self.mcap_files)} 个MCAP文件...")
        
        # 准备任务参数
        task_args = [(mcap_file, self.data_dir, self.config_dict, idx) 
                     for idx, mcap_file in enumerate(self.mcap_files)]
        
        successful_files = 0
        failed_files = 0
        timeout_files = 0
        video_error_files = 0
        total_frames = 0
        processed_files = 0
        
        start_time = time.time()
        task_timeout = 300  # 单个任务最大超时时间5分钟
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(process_single_mcap, args): args[0] 
                             for args in task_args}
            
            # 处理完成的任务
            for future in as_completed(future_to_file, timeout=task_timeout * len(self.mcap_files)):
                file_path = future_to_file[future]
                processed_files += 1
                
                try:
                    # 给单个任务设置超时
                    try:
                        result = future.result(timeout=task_timeout)
                    except TimeoutError:
                        timeout_files += 1
                        failed_files += 1
                        print(f"⏰ 超时 {Path(file_path).name}: 处理时间超过{task_timeout}秒 [{processed_files}/{len(self.mcap_files)}]")
                        continue
                    
                    if result["success"]:
                        # 尝试将episode数据添加到数据集
                        try:
                            frame_added = 0
                            for frame_data in result["episode_data"]:
                                try:
                                    dataset.add_frame(frame_data)
                                    frame_added += 1
                                except Exception as e:
                                    print(f"Error adding frame {frame_added} to dataset for {result['file']}: {e}")
                                    # 如果添加帧失败，跳过这个episode
                                    break
                            else:
                                # 只有所有帧都成功添加时才保存episode
                                try:
                                    dataset.save_episode()
                                    successful_files += 1
                                    total_frames += result["frame_count"]
                                    print(f"✓ 完成 {result['file']} ({result['frame_count']} 帧) [{processed_files}/{len(self.mcap_files)}]")
                                except Exception as e:
                                    print(f"✗ 保存episode失败 {result['file']}: {e}")
                                    failed_files += 1
                        except Exception as e:
                            print(f"✗ 处理episode数据失败 {result['file']}: {e}")
                            failed_files += 1
                    else:
                        failed_files += 1
                        error_msg = result.get("error", "Unknown error")
                        
                        # 统计错误类型
                        if "video" in error_msg.lower() or "frame" in error_msg.lower() or "attachment" in error_msg.lower():
                            video_error_files += 1
                            
                        print(f"✗ 失败 {result['file']}: {error_msg} [{processed_files}/{len(self.mcap_files)}]")
                        
                except Exception as e:
                    failed_files += 1
                    print(f"✗ 处理 {Path(file_path).name} 时发生异常: {e} [{processed_files}/{len(self.mcap_files)}]")
                
                # 定期报告进度
                if processed_files % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_speed = processed_files / elapsed_time if elapsed_time > 0 else 0
                    remaining_files = len(self.mcap_files) - processed_files
                    estimated_remaining_time = remaining_files / avg_speed if avg_speed > 0 else 0
                    print(f"🔄 进度更新: {processed_files}/{len(self.mcap_files)} 已处理, "
                          f"成功: {successful_files}, 失败: {failed_files} (超时: {timeout_files}, 视频错误: {video_error_files}), "
                          f"平均速度: {avg_speed:.2f} 文件/秒, "
                          f"预计剩余时间: {estimated_remaining_time:.1f} 秒")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n=== 处理完成 ===")
        print(f"总文件数: {len(self.mcap_files)}")
        print(f"成功处理: {successful_files}")
        print(f"失败文件: {failed_files}")
        print(f"  - 超时文件: {timeout_files}")
        print(f"  - 视频错误: {video_error_files}")
        print(f"  - 其他错误: {failed_files - timeout_files - video_error_files}")
        print(f"总帧数: {total_frames}")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"平均速度: {len(self.mcap_files)/processing_time:.2f} 文件/秒")
        if total_frames > 0:
            print(f"帧处理速度: {total_frames/processing_time:.2f} 帧/秒")
        
        # 根据失败类型给出建议
        if video_error_files > 0:
            print(f"\n💡 建议: 发现 {video_error_files} 个视频相关错误，可能的原因：")
            print("  - MCAP文件中的视频数据损坏")
            print("  - 视频编码格式不兼容")
            print("  - 视频文件不完整")
        
        if timeout_files > 0:
            print(f"\n⏰ 建议: 发现 {timeout_files} 个超时文件，可能的原因：")
            print("  - 文件过大，处理时间超过限制")
            print("  - 系统资源不足")
            print("  - 可以尝试减少并行工作进程数")
        
        # 如果失败文件过多，给出警告
        if failed_files > len(self.mcap_files) * 0.1:  # 超过10%失败
            print(f"\n⚠️  警告: 失败文件比例较高 ({failed_files}/{len(self.mcap_files)} = {failed_files/len(self.mcap_files)*100:.1f}%)")
            print("建议检查数据文件的完整性和格式是否正确")
        
        if successful_files == 0:
            print("\n❌ 没有任何文件成功转换！请检查：")
            print("1. 数据文件路径是否正确")
            print("2. MCAP文件是否完整")
            print("3. 配置文件是否正确")
            print("4. 是否有足够的磁盘空间")
            raise ValueError("转换失败，没有生成任何有效数据")


def main(data_dir: str, num_workers: int = 4):
    """
    主函数
    
    Args:
        data_dir: 数据目录路径
        num_workers: 并行工作进程数，默认为4
    """
    print(f"开始并行MCAP转换，使用 {num_workers} 个工作进程")
    
    try:
        converter = ParallelMcapConverter(data_dir, num_workers)
    except Exception as e:
        print(f"❌ 初始化转换器失败: {e}")
        print("请检查：")
        print("1. 数据目录是否存在且可访问")
        print("2. config.py文件是否存在且格式正确")
        print("3. MCAP文件是否存在")
        return

    # 清理现有数据集
    output_path = HF_LEROBOT_HOME / converter.task_name
    print(f"输出路径: {output_path}")
    if output_path.exists():
        try:
            shutil.rmtree(output_path)
            print(f"已清理现有数据集: {output_path}")
        except Exception as e:
            print(f"⚠️  清理现有数据集失败: {e}")
            print("继续进行转换...")

    # 创建LeRobot数据集
    try:
        dataset = converter.create_dataset()
        print("✓ 数据集创建成功")
    except Exception as e:
        print(f"❌ 创建数据集失败: {e}")
        return
    
    # 并行加载数据
    try:
        converter.load_to_dataset_parallel(dataset)
        print(f"✅ 转换完成！数据集已保存到: {output_path}")
        
        # 验证最终结果
        try:
            if output_path.exists():
                data_files = list((output_path / "data").glob("**/*.parquet"))
                print(f"📊 生成的数据文件: {len(data_files)} 个parquet文件")
            else:
                print("⚠️  输出目录不存在，可能转换未成功")
        except Exception as e:
            print(f"验证输出时出错: {e}")
            
    except Exception as e:
        print(f"❌ 数据加载过程中发生错误: {e}")
        print("部分数据可能已经成功转换，请检查输出目录")
        return


if __name__ == "__main__":
    tyro.cli(main) 