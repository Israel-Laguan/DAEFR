import os
import cv2
import numpy as np
import os.path as osp
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FFHQPreDegradedDataset(data.Dataset):
    """
    Fast dataset that loads pre-degraded small images (32x32 or 64x64) and
    corresponding GT images (512x512).

    The LQ images already have blur, noise, and JPEG compression baked in from
    the HuggingFace source. During loading, they are upsampled to 512x512 with
    bicubic interpolation and light noise is added for variation.

    This skips the heavy on-the-fly degradation pipeline (blur kernel generation,
    downsampling, JPEG simulation), providing significant training speedup.

    Use scripts/download_predegraded.py to download and extract the LQ images.

    Returns proper LQ-GT pairs for DAEFR restoration training.
    """

    def __init__(self, opt):
        super(FFHQPreDegradedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']  # Pre-degraded 32x32/64x64 images
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        # Light noise for on-the-fly variation (much lower than full pipeline's [0, 20])
        self.noise_range = opt.get('noise_range', [0, 5])

        self.crop_components = opt.get('crop_components', False)  # facial components
        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)

        if self.crop_components:
            self.components_list = torch.load(opt.get('component_path'), weights_only=False)

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError("LMDB not supported for pre-degraded dataset")
        else:
            self.gt_paths = sorted(paths_from_folder(self.gt_folder))
            self.lq_paths = sorted(paths_from_folder(self.lq_folder))

        assert len(self.gt_paths) == len(self.lq_paths), \
            f'GT and LQ must have same number of images, got {len(self.gt_paths)} GT and {len(self.lq_paths)} LQ'

        logger = get_root_logger()
        logger.info(f'FFHQPreDegradedDataset: {len(self.gt_paths)} samples')
        logger.info(f'  GT folder: {self.gt_folder}')
        logger.info(f'  LQ folder: {self.lq_folder}')
        logger.info(f'  Noise range: {self.noise_range}')
        logger.info(f'  Skipping on-the-fly degradation pipeline (blur, downsample, JPEG)')

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load GT image (512x512)
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # Random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        if self.crop_components:
            locations = self.get_component_coordinates(index, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # Load pre-degraded LQ image (32x32 or 64x64)
        lq_path = self.lq_paths[index]
        lq_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(lq_bytes, float32=True)

        # Apply same flip to LQ
        if status[0]:  # hflip was applied
            img_lq = cv2.flip(img_lq, 1)

        # Add light noise for on-the-fly variation
        if self.noise_range is not None:
            noise_level = np.random.uniform(self.noise_range[0], self.noise_range[1])
            if noise_level > 0:
                noise = np.random.randn(*img_lq.shape) * (noise_level / 255.0)
                img_lq = img_lq + noise

        # Upsample LQ to target size using bicubic interpolation
        img_lq = cv2.resize(img_lq, (self.out_size, self.out_size), interpolation=cv2.INTER_CUBIC)

        # Clamp after noise + upsample
        img_lq = np.clip(img_lq, 0, 1)

        # Resize GT if needed
        if (h != self.out_size) and (w != self.out_size):
            img_gt = cv2.resize(img_gt, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # Round and clip LQ (same as original pipeline)
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # Normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)

        return_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path
        }
        if self.crop_components:
            return_dict['loc_left_eye'] = loc_left_eye
            return_dict['loc_right_eye'] = loc_right_eye
            return_dict['loc_mouth'] = loc_mouth

        return return_dict

    def __len__(self):
        return len(self.gt_paths)
