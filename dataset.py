# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello
# --------------------------------------------------------
# Code modified by Yufeng Zheng
# --------------------------------------------------------
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset

import cv2
import h5py
from core import DefaultConfig
import random
import logging
import threading

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
config = DefaultConfig()

_fa_local = threading.local()

class HDFDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 is_bgr=False,
                 get_2nd_sample=False,
                 pick_at_least_per_person=None,
                 num_labeled_samples=None,
                 sample_target_label=False,
                 ):

        assert os.path.isfile(hdf_file_path)
        self.get_2nd_sample = get_2nd_sample
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.is_bgr = is_bgr
        self.sample_target_label = sample_target_label
        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            if prefixes is None:
                self.prefixes = hdf_keys
            else:
                self.prefixes = [k for k in prefixes if k in h5f]
            if pick_at_least_per_person is not None:
                self.prefixes = [k for k in self.prefixes if k in h5f and len(next(iter(h5f[k].values()))) >=
                            pick_at_least_per_person]
            self.index_to_query = sum([[(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                                       for prefix in self.prefixes], [])
            if num_labeled_samples is not None:
                # randomly pick labeled samples for semi-supervised training
                ra = list(range(len(self.index_to_query)))
                random.seed(0)
                random.shuffle(ra)
                # Make sure that the ordering is the same
                # assert ra[:3] == [744240, 1006758, 1308368]
                ra = ra[:num_labeled_samples]
                list.sort(ra)
                self.index_to_query = [self.index_to_query[i] for i in ra]

            # calculate kernel density of gaze and head pose, for generating new redirected samples
            if sample_target_label:
                if num_labeled_samples is not None:
                    sample = []
                    old_key = -1
                    for key, idx in self.index_to_query:
                        if old_key != key:
                            group = h5f[key]
                        sample.append(group['labels'][idx, :4])
                    sample = np.asarray(sample, dtype=np.float32)
                else:
                    # can calculate faster if load by group
                    sample = None
                    for key in self.prefixes:
                        group = h5f[key]
                        if sample is None:
                            sample = group['labels'][:, :4]
                        else:
                            sample = np.concatenate([sample, group['labels'][:, :4]], axis=0)
                sample = sample.transpose()
                from scipy import stats
                self.kernel = stats.gaussian_kde(sample)
                logging.info("Finished calculating kernel density for gaze and head angles")
                # Sample new gaze and head pose angles
                new_samples = self.kernel.resample(len(self.index_to_query))
                self.gaze = new_samples[:2, :].transpose()
                self.head = new_samples[2:4, :].transpose()
                self.index_of_sample = 0

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def get_eye_mask(self, image):
        """
        Generate an eye region mask using facial landmarks.
        Args:
            image: numpy array, shape [H, W, 3], RGB, uint8
        Returns:
            mask: numpy array, shape [H, W], float32, values in [0,1]
        """
        # Lazy-load face_alignment in each process/thread
        if not hasattr(_fa_local, 'fa'):
            from face_alignment import FaceAlignment, LandmarksType
            _fa_local.fa = FaceAlignment(LandmarksType.TWO_D, device='cuda')
        fa = _fa_local.fa

        landmarks = fa.get_landmarks(image)
        if landmarks is None or len(landmarks) == 0:
            # fallback: return zeros
            return np.zeros(image.shape[:2], dtype=np.float32)
        lm = landmarks[0]  # [68, 2]
        # dlib 68-point: left eye 36-41, right eye 42-47
        left_eye = lm[36:42]
        right_eye = lm[42:48]
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.fillPoly(mask, [np.int32(left_eye)], 1)
        cv2.fillPoly(mask, [np.int32(right_eye)], 1)
        # Optional: Gaussian blur for smooth edge
        # mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    def get_eye_mask_boundary(self, image):
        mask = self.get_eye_mask(image)
        kernel = np.ones((7, 7), np.uint8)  # kernel size can be adjusted
        dilated_mask = cv2.dilate((mask > 0.2).astype(np.uint8), kernel, iterations=1)
        boundary_mask = np.logical_xor(dilated_mask, (mask > 0.2)).astype(np.float32)
        return boundary_mask

    def preprocess_image(self, image):
        if self.is_bgr:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.long, requires_grad=False)
        return entry

    def retrieve(self, group, index):
        eyes = self.preprocess_image(group['pixels'][index, :])
        g = group['labels'][index, :2]
        h = group['labels'][index, 2:4]
        # Generate eye mask using landmarks
        # Convert normalized eyes back to uint8 RGB for landmark detection
        img = ((eyes.transpose(1,2,0) + 1) * 127.5).astype(np.uint8)
        mask = self.get_eye_mask(img)
        # Create boundary mask: dilate mask then xor with original mask
        boundary_mask = self.get_eye_mask_boundary(img)
        # mask = np.ones(eyes.shape[1:])
        # boundary_mask = np.zeros(eyes.shape[1:])
        return eyes, g, h, mask, boundary_mask

    def __getitem__(self, idx):
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick entry a and b from same person
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]
        group_b = group_a

        # Grab 1st (input) entry
        eyes_a, g_a, h_a, mask_a, boundary_mask_a = self.retrieve(group_a, idx_a)
        entry = {
            'key': key_a,
            'image_a': eyes_a,
            'gaze_a': g_a,
            'head_a': h_a,
            'mask_a': mask_a,  # Add mask to entry
            'boundary_mask_a': boundary_mask_a,  # Add boundary mask
        }
        if self.sample_target_label:
            entry['gaze_b_r'] = self.gaze[self.index_of_sample]
            entry['head_b_r'] = self.head[self.index_of_sample]
            self.index_of_sample += 1
        if self.get_2nd_sample:
            all_indices = list(range(len(next(iter(group_a.values())))))
            if len(all_indices) == 1:
                idx_b = idx_a
            else:
                all_indices_but_a = np.delete(all_indices, idx_a)
                idx_b = np.random.choice(all_indices_but_a)
            eyes_b, g_b, h_b, mask_b, boundary_mask_b = self.retrieve(group_b, idx_b)
            entry['image_b'] = eyes_b
            entry['gaze_b'] = g_b
            entry['head_b'] = h_b
            entry['mask_b'] = mask_b  # Add mask for second sample
            entry['boundary_mask_b'] = boundary_mask_b  # Add boundary mask for second sample
        return self.preprocess_entry(entry)

