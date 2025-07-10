"""The dataloader for V4V dataset.

Details for the UBFCrPPG Dataset see https://vision4vitals.github.io/v4v_dataset.html.
If you use this dataset, please cite this paper:
Revanur, A., Li, Z., Ciftci, U. A., Yin, L., & Jeni, L. A. (2021). The first vision for vitals (v4v) challenge for non-contact video-based physiological estimation. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 2760-2767).
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader

class V4VLoader(BaseLoader):
    """The data loader for the V4V dataset."""

    def __init__(self, name, data_path, config_data, split='test'):
        """Initializes an V4V dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                RawData/
                    |--train/
                        |--signal
                            |--.txt
                        |--video
                            |--./mkv
                    |--test/
                        |--signal
                            |--.txt
                        |--video
                            |--./mkv
                    
                    |--val/
                        |--signal
                            |--.txt
                        |--video
                            |--./mkv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.split = split
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        split_path = os.path.join(data_path, self.split)
        video_path = os.path.join(split_path, 'video')
        signal_path = os.path.join(split_path, 'signal')

        data_dirs = []
        video_files = glob.glob(os.path.join(video_path, '*.mkv'))

        if not video_files:
            raise ValueError(f"No video files found in {video_path}!")

        if self.split == "train":
            for i, video_file in enumerate(sorted(video_files)):
                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(video_file))[0]  # e.g., F001_T1
                # Modify base_name to match the signal file naming convention
                signal_base_name = base_name.replace('_', '-') + '-BP'
                signal_file = os.path.join(signal_path, signal_base_name + '.txt')
                
                if not os.path.exists(signal_file):
                    raise ValueError(f"Signal file {signal_file} not found for video {video_file}")

                data_dirs.append({
                    'index': i,
                    'split': self.split,
                    'video_file': video_file,
                    'signal_file': signal_file,
                    'base_name': base_name
                })
        else: 
            for i, video_file in enumerate(sorted(video_files)):
                base_name = os.path.splitext(os.path.basename(video_file))[0]  # e.g., F001_T1
                signal_file = os.path.join(signal_path, base_name + '.txt')

                if not os.path.exists(signal_file):
                    raise ValueError(f"Signal file {signal_file} not found for video {video_file}")

                data_dirs.append({
                    'index': i,
                    'split': self.split,
                    'video_file': video_file,
                    'signal_file': signal_file,
                    'base_name': base_name
                })

        if not data_dirs:
            raise ValueError(f"{self.dataset_name} data paths empty!")

        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Splits the data according to begin and end ratios.

        Args:
            data_dirs (list): List of data directories.
            begin (float): Starting ratio (e.g., 0.0).
            end (float): Ending ratio (e.g., 1.0).

        Returns:
            list: A subset of data_dirs.
        """
        total_num = len(data_dirs)
        start_idx = int(total_num * begin)
        end_idx = int(total_num * end)
        data_dirs_split = data_dirs[start_idx:end_idx]
        return data_dirs_split

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Preprocesses the raw data.

        Args:
            data_dirs (list): List of data directories.
            config_preprocess (CfgNode): Preprocessing configuration.
            begin (float): Starting ratio for data split.
            end (float): Ending ratio for data split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess)
        self.build_file_list(file_list_dict)
        self.load_preprocessed_data()

        print("Total Number of raw files preprocessed:", len(data_dirs_split), end='\n\n')

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocesses a single data entry (used in multiprocessing).

        Args:
            data_dirs (list): List of data directories.
            config_preprocess (CfgNode): Preprocessing configuration.
            i (int): Index of the data to process.
            file_list_dict (dict): Dictionary to store file paths.
        """
        data_dir = data_dirs[i]
        video_file = data_dir['video_file']
        signal_file = data_dir['signal_file']
        index = data_dir['index']
        base_name = data_dir['base_name']

        try:
            frames = self.read_video(video_file)

            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(signal_file)

            target_length = frames.shape[0]
            bvps = BaseLoader.resample_ppg(bvps, target_length)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, base_name)
            file_list_dict[i] = input_name_list
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            file_list_dict[i] = []

    @staticmethod
    def read_video(video_file):
        """Reads a video file and returns frames in (T, H, W, 3) format.

        Args:
            video_file (str): Path to the video file.

        Returns:
            np.ndarray: Array of video frames.
        """
        VidObj = cv2.VideoCapture(video_file)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_wave(signal_file):
        """Reads a signal file.

        Args:
            signal_file (str): Path to the signal (.txt) file.

        Returns:
            np.ndarray: Array of signal values.
        """
        bvps = np.loadtxt(signal_file)
        return bvps