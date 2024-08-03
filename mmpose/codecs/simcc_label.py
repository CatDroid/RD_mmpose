# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np

from mmpose.codecs.utils import get_simcc_maximum
from mmpose.codecs.utils.refinement import refine_simcc_dark
from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class SimCCLabel(BaseKeypointCodec):
    r"""Generate keypoint representation via "SimCC" approach.
    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_x_labels (np.ndarray): The generated SimCC label for x-axis.
            The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wx=w*simcc_split_ratio`
        - keypoint_y_labels (np.ndarray): The generated SimCC label for y-axis.
            The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wy=h*simcc_split_ratio`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]
        smoothing_type (str): The SimCC label smoothing strategy. Options are
            ``'gaussian'`` and ``'standard'``. Defaults to ``'gaussian'``
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC
            label. Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        label_smooth_weight (float): Label Smoothing weight. Defaults to 0.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.
        use_dark (bool): Whether to use the DARK post processing. Defaults to
            False.
        decode_visibility (bool): Whether to decode the visibility. Defaults
            to False.
        decode_beta (float): The beta value for decoding visibility. Defaults
            to 150.0.

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """

    label_mapping_table = dict(
        keypoint_x_labels='keypoint_x_labels',
        keypoint_y_labels='keypoint_y_labels',
        keypoint_weights='keypoint_weights',
    )

    def __init__(
        self,
        input_size: Tuple[int, int],
        smoothing_type: str = 'gaussian',
        # 联合体类型  可以是 (x_sigma, y_sigma)
        sigma: Union[float, int, Tuple[float]] = 6.0,
        simcc_split_ratio: float = 2.0,
        label_smooth_weight: float = 0.0,
        normalize: bool = True,
        use_dark: bool = False,
        decode_visibility: bool = False,
        decode_beta: float = 150.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.smoothing_type = smoothing_type
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.normalize = normalize
        self.use_dark = use_dark
        self.decode_visibility = decode_visibility
        self.decode_beta = decode_beta

        #  (x_sigma, y_sigma) 两个维度的
        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma])
        else:
            self.sigma = np.array(sigma)

        if self.smoothing_type not in {'gaussian', 'standard'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `smoothing_type` value'
                f'{self.smoothing_type}. Should be one of '
                '{"gaussian", "standard"}')

        if self.smoothing_type == 'gaussian' and self.label_smooth_weight > 0:
            raise ValueError('Attribute `label_smooth_weight` is only '
                             'used for `standard` mode.')

        if self.label_smooth_weight < 0.0 or self.label_smooth_weight > 1.0:
            raise ValueError('`label_smooth_weight` should be in range [0, 1]')

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints into SimCC labels. Note that the original
        keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_x_labels (np.ndarray): The generated SimCC label for
                x-axis.
                The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
                and (N, K) if `smoothing_type=='standard'``, where
                :math:`Wx=w*simcc_split_ratio`
            - keypoint_y_labels (np.ndarray): The generated SimCC label for
                y-axis.
                The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
                and (N, K) if `smoothing_type=='standard'``, where
                :math:`Wy=h*simcc_split_ratio`
                standard 应该也是 ?? (N, K, W)  ??
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        # 如果没有设置可见性，就默认全部可见 
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.smoothing_type == 'gaussian':
            x_labels, y_labels, keypoint_weights = self._generate_gaussian(
                keypoints, keypoints_visible)
        elif self.smoothing_type == 'standard':
            x_labels, y_labels, keypoint_weights = self._generate_standard(
                keypoints, keypoints_visible)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `smoothing_type` value'
                f'{self.smoothing_type}. Should be one of '
                '{"gaussian", "standard"}')

        encoded = dict(
            # (N, K, Wx)  Wx = Wx=w*simcc_split_ratio
            keypoint_x_labels=x_labels,
            # (N, K, Wy) Wy=h*simcc_split_ratio
            keypoint_y_labels=y_labels,
            # (N, K)
            keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, simcc_x: np.ndarray,
               simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.
        
        SimCC表示解码为关键点坐标  最后的坐标是在image space! 不是heatmap space(不是heatmap方案) 也不是 splite后的坐标

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis
                and y-axis
            simcc_x (np.ndarray): SimCC label for x-axis
            simcc_y (np.ndarray): SimCC label for y-axis

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        # 关键点坐标和置信度分数
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)

        # Unsqueeze the instance dimension for single-instance results
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        # DARK后处理   ??? 使用 refine_simcc_dark 函数对x和y坐标进行细化，提高精度 ???
        if self.use_dark:
            x_blur = int((self.sigma[0] * 20 - 7) // 3)
            y_blur = int((self.sigma[1] * 20 - 7) // 3)
            x_blur -= int((x_blur % 2) == 0)
            y_blur -= int((y_blur % 2) == 0)
            keypoints[:, :, 0] = refine_simcc_dark(keypoints[:, :, 0], simcc_x,
                                                   x_blur)
            keypoints[:, :, 1] = refine_simcc_dark(keypoints[:, :, 1], simcc_y,
                                                   y_blur)

        # 调整为输入图大小 
        keypoints /= self.simcc_split_ratio

        # print(f"simcc_lable.py decode_visibility = {self.decode_visibility} use_dark = {self.use_dark} ")
        # decode_visibility = False use_dark = False
        # 应该对 每个目标框 推理到的关键点坐标 都会跑一次这里  

        # 解码关键点的可见性 (可见性 跟 置信度不一样)
        if self.decode_visibility:
            _, visibility = get_simcc_maximum(
                simcc_x * self.decode_beta * self.sigma[0],
                simcc_y * self.decode_beta * self.sigma[1],
                apply_softmax=True)
            return keypoints, (scores, visibility)
        else:
            return keypoints, scores

    def _map_coordinates(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mapping keypoint coordinates into SimCC space."""

        keypoints_split = keypoints.copy()
        keypoints_split = np.around(keypoints_split * self.simcc_split_ratio)
        keypoints_split = keypoints_split.astype(np.int64)
        keypoint_weights = keypoints_visible.copy()

        return keypoints_split, keypoint_weights

    # keypoints_visible 最后作为weight  没有标注(vis<0.5) 和 在图片外的关键点 weight都是=0
    def _generate_standard(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Standard Label Smoothing
        strategy.

        Labels will be one-hot vectors if self.label_smooth_weight==0.0
        如果 label_smooth_weight ==0 就是 one-hot 向量
        """

        N, K, _ = keypoints.shape
        # input_size 网络输入图的大小 heatmap_size 网络输出图的大小 
        w, h = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)

        # 把坐标映射到 W*2  H*2   
        # keypoint_weights 就是 copy keypoints_visible
        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        # 每个关键点 X坐标 和 Y坐标 向量, 分别保存
        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 逐个关键点 遍历 product ! 
        for n, k in product(range(N), range(K)):
            # 跳过没有标注的点 
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # 关键点坐标 
            # get center coordinates
            mu_x, mu_y = keypoints_split[n, k].astype(np.int64)

            # 已经超出了 w*2 h*2 (因为坐标已经*2  假设 simcc_split_ratio = 2  )
            # detect abnormal coords and assign the weight 0
            if mu_x >= W or mu_y >= H or mu_x < 0 or mu_y < 0:
                keypoint_weights[n, k] = 0
                continue

            # smooth label 只是简单的 不把其他分类概率设置为0
            # 如果不做 smooth label, 就是 正确分类 1.0 其他分类概率都是0.0 

            # 其他分类概率 平均self.label_smooth_weight
            if self.label_smooth_weight > 0:
                target_x[n, k] = self.label_smooth_weight / (W - 1)
                target_y[n, k] = self.label_smooth_weight / (H - 1)

            # 正确分类 概率为  1.0 - self.label_smooth_weight
            target_x[n, k, mu_x] = 1.0 - self.label_smooth_weight
            target_y[n, k, mu_y] = 1.0 - self.label_smooth_weight

        return target_x, target_y, keypoint_weights

    def _generate_gaussian(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encoding keypoints into SimCC labels with Gaussian Label Smoothing
        strategy."""

        N, K, _ = keypoints.shape
        w, h = self.input_size
        W = np.around(w * self.simcc_split_ratio).astype(int)
        H = np.around(h * self.simcc_split_ratio).astype(int)

        keypoints_split, keypoint_weights = self._map_coordinates(
            keypoints, keypoints_visible)

        target_x = np.zeros((N, K, W), dtype=np.float32)
        target_y = np.zeros((N, K, H), dtype=np.float32)

        # 3-sigma rule
        radius = self.sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)

        # 遍历 每个图片N, 每个关键点K
        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            # (x, y)
            mu = keypoints_split[n, k]

            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            # mu - radius >= W  mu >= W + radius 
            # 也就是 只有关键点超出了 3*sigma 才会把weight设置为0   相当于图片上下左右扩大了3*sigma 
            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            mu_x, mu_y = mu

            # x 和 y 都是固定 从 0~W-1 或者 0~H-1 的数列 这里计算出 每个x/y bin的概率 
            target_x[n, k] = np.exp(-((x - mu_x)**2) / (2 * self.sigma[0]**2))
            target_y[n, k] = np.exp(-((y - mu_y)**2) / (2 * self.sigma[1]**2))

        if self.normalize:
            # 是否归一化(一维正态分布概率和为1) 因为sigma都是一样的 
            norm_value = self.sigma * np.sqrt(np.pi * 2)
            target_x /= norm_value[0]
            target_y /= norm_value[1]

        return target_x, target_y, keypoint_weights
