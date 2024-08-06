# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils.gaussian_heatmap import (generate_gaussian_heatmaps,
                                     generate_unbiased_gaussian_heatmaps)
from .utils.post_processing import get_heatmap_maximum
from .utils.refinement import refine_keypoints, refine_keypoints_dark


@KEYPOINT_CODECS.register_module()
class MSRAHeatmap(BaseKeypointCodec):
    """Represent keypoints as heatmaps via "MSRA" approach. See the paper:
    `Simple Baselines for Human Pose Estimation and Tracking`_ by Xiao et al
    (2018) for details.

    MSRA的方法 2D关键点的2d-heatmap表征

    根据heatmap的分辨率(配置文件写了sigma) 是一共不同大小的sigma, 高斯图大小(相当于kernelsize)使用3sigma或者整个heatmap大小(无偏)

    config比较多 backbone可以是 hrnet hrformer resnet resnext hourglass shufflenet swin等等

    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
        codec = dict( type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-aic-256x192-combine.py
        codec = dict(ype='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The generated heatmap in shape (K, H, W)
            where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        unbiased (bool): Whether use unbiased method (DarkPose) in ``'msra'``
            encoding. See `Dark Pose`_ for details. Defaults to ``False``
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. The kernel size and sigma should follow
            the expirical formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`.
            Defaults to 11

    .. _`Simple Baselines for Human Pose Estimation and Tracking`:
        https://arxiv.org/abs/1804.06208
    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.unbiased = unbiased

        # encode 都不用kernel size ?  generate_gaussian_heatmaps 中 kernel size使用 3sigma * 2 + 1 (覆盖高斯分布99%)
        # decode 只有DarkPose才会使用这个kernel size
        # megvii方法 反而用外面提供的kernel size 但是sigma用cv2.GaussianBlur自动计算的

        # The Gaussian blur kernel size of the heatmap modulation
        # in DarkPose and the sigma value follows the expirical
        # formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`
        # which gives:
        #   sigma~=3 if ks=17
        #   sigma=2 if ks=11;
        #   sigma~=1.5 if ks=7;
        #   sigma~=1 if ks=3;
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # encode    不用外面传入的 kernel size ; 只用外面传入的sigma 
        #           关键点, 先从‘input空间’转换到‘output空间‘(heatmap分辨率)
        #           两个方式(unbiased)都一样的参数
        #           mmpose/codecs/utils/gaussian_heatmap.py
        if self.unbiased:
            # 无偏的高斯分布 
            #   1. 还是确保中心的概率是1.0 (没有归一化) 
            #   2. 生成整个W*H heatmap大小的高斯图(不只是kernel size大小的)
            heatmaps, keypoint_weights = generate_unbiased_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            # generate_gaussian_heatmaps
            #   1. 支持多个人(但是msra_heatmap只支持一个实例) 一个关键点用一个2d-heatmap表示  一个heatmap上的多个人的(同个)关键点的 高斯分布 用max方式 混合 
            #   2. 高斯分布使用非归一化的, 这样, 中心点的概率值是1
            #   3. 不同的’实例‘ 可以使用不同的sigma, kernel size(或者说高斯图的大小)都是3*sigma * 2 + 1 (高斯图, 覆盖99%高斯分布)
            heatmaps, keypoint_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=keypoints / self.scale_factor,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

        encoded = dict(heatmaps=heatmaps, keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = encoded.copy()
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        if self.unbiased:
            # Alleviate biased coordinate
            keypoints = refine_keypoints_dark(
                keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size)

        else:
            keypoints = refine_keypoints(keypoints, heatmaps)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores
