# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import gaussian_blur, get_heatmap_maximum


@KEYPOINT_CODECS.register_module()
class MegviiHeatmap(BaseKeypointCodec):
    """Represent keypoints as heatmaps via "Megvii" approach. See `MSPN`_
    (2019) and `CPN`_ (2018) for details.

    Megvii方法 2d关键点坐标的2d-heatmap表征 

    热图的生成方式：
        Megvii(旷视科技)方法: 
            使用基于高斯核的方法生成热图。每个关键点在热图中由一个高斯分布表示，
            高斯分布的标准差(σ)不作为参数, 根据kernelSize自动计算(cv2.GaussianBlur)
            使用多种kernelSize(config文件配置) 对应不同stage的heatmap kernelsize越来越小 越来越精确
            MSPN CPN 两个相关的论文 

        MSRA方法:(Microsoft Research Asia)
            根据heatmap尺寸,在配置文件中写好了sigma 
            高斯图 是3sigma或者是heatmap大小 

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
        kernel_size (tuple): The kernel size of the heatmap gaussian in
            [ks_x, ks_y]

    .. _`MSPN`: https://arxiv.org/abs/1901.00148
    .. _`CPN`: https://arxiv.org/abs/1711.07319
    """

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        kernel_size: int,
    ) -> None:

        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.kernel_size = kernel_size
        self.scale_factor = (np.array(input_size) /
                             heatmap_size).astype(np.float32)
        # 输入分辨率 和 输出的heatmap分辨率的缩放比例 
        # sigma有cv2.GaussBlur自动计算
        # 
        # 使用到的模型(config文件)是:
        #   configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192.py
        #   top-down backbone是rsn
        #   configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mspn50_8xb32-210e_coco-256x192.py
        #   top-down backbone是mspn 
        # ?? 输入分辨率都只有 256x192 一种 ??

        '''
            mmpose中megvii_heatmap的模型配置 都是这样 :(backbone不管是rsn还是mspn)
            ??? 使用 多个 高斯核 ??
            ??? 使用多个 4阶Hourglass 串起来 ?? 但是每个 4阶Hourglass 输出都是一样分辨率呀?? 

            configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xmspn50_8xb32-210e_coco-256x192.py
            # codec settings
            # multiple kernel_sizes of heatmap gaussian for 'Megvii' approach.
            kernel_sizes = [15, 11, 9, 7, 5]
            codec = [
                dict(
                    type='MegviiHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    kernel_size=kernel_size) for kernel_size in kernel_sizes
            ]

            这样会有 5个codec !!!

            head 都是使用 MSPNHead MSPNHead
            mmpose/models/heads/heatmap_heads/mspn_head.py

            backbone是MSPN (Multi-Stage Pose estimation Network)
           

            # MSPN (vs Hourglass)
            # https://github.com/megvii-research/MSPN
            # https://www.cnblogs.com/easy-hard/p/12153436.html
            1. hourglass的网络, 第一个stage的输出作为第二个stage的输入, 这是一种stage级别的融和;
               MSPN, 则是同一分辨率的feature级别的特征融和, 是深入到stage内部的, 所以融和的效果更好
            2. 一个是在stage 1 中, 目标heatmap的生成采用的是大的高斯核, 也就是说生成的heatmap“比较模糊”
                而在stage 2中则采用了比较精确的heatmap.   
                这个设计原则其实也体现了由粗到精的优化原则.  --------- 所以配置文件中 heatmap指定了5个kernel size  
            3. 在每一个stage的最高的resolution中产生的prediction只取损失值最大的k个, 放到损失函数中 
                即所谓的OHKM(online hard keypoint mining)
                这个OHKM是每一轮迭代的时候就进行的。
                
        '''



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

        N, K, _ = keypoints.shape
        W, H = self.heatmap_size

        assert N == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')

        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()

        for n, k in product(range(N), range(K)):
            # 跳过没有标注的点 
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # 把输入空间的坐标 转换成 输出空间(heatmap)的坐标
            # get center coordinates
            kx, ky = (keypoints[n, k] / self.scale_factor).astype(np.int64)
            
            # 如果关键点 超出了 heatmap的大小 直接设置权重伪0 
            if kx < 0 or kx >= W or ky < 0 or ky >= H:
                keypoint_weights[n, k] = 0
                continue

            # cv2.GaussianBlur 自动计算sigma
            # 
            #   sigmaX 设置为 0 时，OpenCV 会根据高斯核的大小 (ksize) 来自动计算标准差 (sigmaX)
            #   
            #   ksize: 3, sigma: 0.80, ratio: 0.27  ksize越大 sigma就越大 
            #   ksize: 5, sigma: 1.10, ratio: 0.22
            #   ksize: 7, sigma: 1.40, ratio: 0.20
            #   ksize: 9, sigma: 1.70, ratio: 0.19
        
            # 高斯核大小和sigma的关系
            #   计算公式 是 高斯分布公式 ,  计算核大小范围内(以核中心为均值点)上的概率密度 
            #  
            #   高斯核的大小, 只是直接影响 计算出来的矩阵的大小, 高斯模糊的范围 比如5x5 就是计算5x5个像素上的2d高斯概率值  
            #   sigma值 才是影响平滑的效果  
            #   kernel_size = 6 * sigma  (3-sigma 包含99%) 可以捕获到大部分的高斯分布
            
            # 比较自动计算的 sigma 与理想情况下覆盖 99% 高斯分布的 sigma
            # ksize: 3, auto_sigma: 0.80, ideal_sigma: 0.33
            # ksize: 5, auto_sigma: 1.10, ideal_sigma: 0.67
            # ksize: 7, auto_sigma: 1.40, ideal_sigma: 1.00
            # ksize: 9, auto_sigma: 1.70, ideal_sigma: 1.33
            # ksize: 11, auto_sigma: 2.00, ideal_sigma: 1.67
            # ksize: 13, auto_sigma: 2.30, ideal_sigma: 2.00
            # ksize: 15, auto_sigma: 2.60, ideal_sigma: 2.33
            # ksize: 17, auto_sigma: 2.90, ideal_sigma: 2.67
            # ksize: 19, auto_sigma: 3.20, ideal_sigma: 3.00
            # 1. 自动计算的 sigma 通常大于 理想 sigma，特别是在较小的 ksize 情况下。
            # 2. 随着 ksize 的增大，两者的差异逐渐减小，但仍然存在
            # 3. 自动计算的 sigma 并没有严格按照捕捉到大部分高斯分布的原则来设计，因此在一些情况下可能无法捕捉到 99% 的高斯分布。

            # 首先在特定位置设置一个峰值（1）。
            # 然后通过高斯模糊，将这个峰值扩散成一个平滑的"山丘"形状。
            heatmaps[k, ky, kx] = 1.
            kernel_size = (self.kernel_size, self.kernel_size)
            heatmaps[k] = cv2.GaussianBlur(heatmaps[k], kernel_size, 0)
            # kernel_size 是个传入参数 ?? 配置相关?? 

            # 归一化 除以 关键点处 的概率 
            # 高斯模糊后的热图在原始峰值位置 (ky, kx) 的值。
            # 由于高斯模糊会使峰值略微降低，这个值通常会小于1。
            # 除以这个值的作用是将热图重新归一化，使得原始峰值位置的值重新变为1
            # normalize the heatmap
            heatmaps[k] = heatmaps[k] / heatmaps[k, ky, kx] * 255.

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
                (K, D)
            - scores (np.ndarray): The keypoint scores in shape (K,). It
                usually represents the confidence of the keypoint prediction
        """
        heatmaps = gaussian_blur(encoded.copy(), self.kernel_size)
        K, H, W = heatmaps.shape

        keypoints, scores = get_heatmap_maximum(heatmaps)

        for k in range(K):
            heatmap = heatmaps[k]
            px = int(keypoints[k, 0])
            py = int(keypoints[k, 1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                keypoints[k] += (np.sign(diff) * 0.25 + 0.5)

        scores = scores / 255.0 + 0.5

        # Unsqueeze the instance dimension for single-instance results
        # and restore the keypoint scales
        keypoints = keypoints[None] * self.scale_factor
        scores = scores[None]

        return keypoints, scores
