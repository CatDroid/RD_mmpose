# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Optional, Tuple, Union

import numpy as np


def generate_3d_gaussian_heatmaps(
    heatmap_size: Tuple[int, int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: Union[float, Tuple[float], np.ndarray],
    image_size: Tuple[int, int],
    heatmap3d_depth_bound: float = 400.0,
    joint_indices: Optional[list] = None,
    max_bound: float = 1.0,
    use_different_joint_weights: bool = False,
    dataset_keypoint_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3d gaussian heatmaps of keypoints.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H, D]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, C)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float or List[float]): A list of sigma values of the Gaussian
            heatmap for each instance. If sigma is given as a single float
            value, it will be expanded into a tuple
        image_size (Tuple[int, int]): Size of input image.
        heatmap3d_depth_bound (float): Boundary for 3d heatmap depth.
            Default: 400.0.
        joint_indices (List[int], optional): Indices of joints used for heatmap
            generation. If None (default) is given, all joints will be used.
            Default: ``None``.
        max_bound (float): The maximal value of heatmap. Default: 1.0.
        use_different_joint_weights (bool): Whether to use different joint
            weights. Default: ``False``.
        dataset_keypoint_weights (np.ndarray, optional): Keypoints weight in
            shape (K, ).

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K * D, H, W) where [W, H, D] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)
    """

    W, H, D = heatmap_size

    # select the joints used for target generation
    if joint_indices is not None:
        keypoints = keypoints[:, joint_indices, ...]
        keypoints_visible = keypoints_visible[:, joint_indices, ...]
    N, K, _ = keypoints.shape

    heatmaps = np.zeros([K, D, H, W], dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    if isinstance(sigma, (int, float)):
        sigma = (sigma, ) * N

    for n in range(N):
        # 3-sigma rule
        radius = sigma[n] * 3

        # joint location in heatmap coordinates
        mu_x = keypoints[n, :, 0] * W / image_size[0]  # (K, )
        mu_y = keypoints[n, :, 1] * H / image_size[1]
        mu_z = (keypoints[n, :, 2] / heatmap3d_depth_bound + 0.5) * D

        keypoint_weights[n, ...] = keypoint_weights[n, ...] * (mu_z >= 0) * (
            mu_z < D)
        if use_different_joint_weights:
            keypoint_weights[
                n] = keypoint_weights[n] * dataset_keypoint_weights
        # xy grid
        gaussian_size = 2 * radius + 1

        # get neighboring voxels coordinates
        x = y = z = np.arange(gaussian_size, dtype=np.float32) - radius
        zz, yy, xx = np.meshgrid(z, y, x)

        xx = np.expand_dims(xx, axis=0)
        yy = np.expand_dims(yy, axis=0)
        zz = np.expand_dims(zz, axis=0)
        mu_x = np.expand_dims(mu_x, axis=(-1, -2, -3))
        mu_y = np.expand_dims(mu_y, axis=(-1, -2, -3))
        mu_z = np.expand_dims(mu_z, axis=(-1, -2, -3))

        xx, yy, zz = xx + mu_x, yy + mu_y, zz + mu_z
        local_size = xx.shape[1]

        # round the coordinates
        xx = xx.round().clip(0, W - 1)
        yy = yy.round().clip(0, H - 1)
        zz = zz.round().clip(0, D - 1)

        # compute the target value near joints
        gaussian = np.exp(-((xx - mu_x)**2 + (yy - mu_y)**2 + (zz - mu_z)**2) /
                          (2 * sigma[n]**2))

        # put the local target value to the full target heatmap
        idx_joints = np.tile(
            np.expand_dims(np.arange(K), axis=(-1, -2, -3)),
            [1, local_size, local_size, local_size])
        idx = np.stack([idx_joints, zz, yy, xx],
                       axis=-1).astype(int).reshape(-1, 4)

        heatmaps[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = np.maximum(
            heatmaps[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]],
            gaussian.reshape(-1))

    heatmaps = (heatmaps * max_bound).reshape(-1, H, W)

    return heatmaps, keypoint_weights


def generate_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: Union[float, Tuple[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints.

    支持多个人 一个关键点用一个2d-heatmap表示  一个heatmap上的多个人的(同个)关键点的 高斯分布 用max方式 混合 
    高斯分布使用非归一化的, 这样, 中心点的概率值是1

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float or List[float]): A list of sigma values of the Gaussian
            heatmap for each instance. If sigma is given as a single float
            value, it will be expanded into a tuple

        不同的’实例‘ 可以使用不同的sigma, kernel size(或者说高斯图的大小)都是3*sigma * 2 + 1 (高斯图, 覆盖99%高斯分布)

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    # K个关键点 每个关键点用一个HxW的heatmap表示 (N可以是 >1)
    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    # 注意这里copy了, 后面只修改 keypoint_weights 但是不会修改 keypoints_visible 
    keypoint_weights = keypoints_visible.copy()

    if isinstance(sigma, (int, float)):
        sigma = (sigma, ) * N

    for n in range(N):
        # 3-sigma rule
        radius = sigma[n] * 3

        # 高斯核 包含3sigma 也就是覆盖99%的高斯分布 
        # xy grid
        gaussian_size = 2 * radius + 1
        x = np.arange(0, gaussian_size, 1, dtype=np.float32)
        y = x[:, None]

        # 中心点 (用在高斯图)
        x0 = y0 = gaussian_size // 2

        for k in range(K):
            # skip unlabled keypoints
            # 不是不可见点，而是没有标注? 
            if keypoints_visible[n, k] < 0.5:
                continue

            # 四舍五入：添加 0.5 再转换为整数实际上是在执行四舍五入操作 有助于减少由于坐标舍入引起的误差
            # get gaussian center coordinates
            mu = (keypoints[n, k] + 0.5).astype(np.int64)

            # check that the gaussian has in-bounds part
            left, top = (mu - radius).astype(np.int64)
            right, bottom = (mu + radius + 1).astype(np.int64)

            # 只有关键点位置 超出了 heatmap外的高斯分布3*sigma范围外 就把这点的权重设置伪0
            # (mu - radius).x >= W
            # mu.x >= W + radius.x 
            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            # 二维高斯分布 没有归一化
            # 1. 只需要一个相对强度 不需要精确的概率密度
            # 2. 减少计算量
            # 3. 数值稳定 ??? 但是指数里面也有除以sigma ???
            #
            # 生成一个高斯图(不是高斯核) 并且没有归一化(希望中心点是1)
            # The gaussian is not normalized,
            # we want the center value to equal 1
            gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma[n]**2))

            # 只使用了一部分的高斯图
            # valid range in gaussian
            g_x1 = max(0, -left)
            g_x2 = min(W, right) - left
            g_y1 = max(0, -top)
            g_y2 = min(H, bottom) - top

            # 对应heatmap上的位置
            # valid range in heatmap
            h_x1 = max(0, left)
            h_x2 = min(W, right)
            h_y1 = max(0, top)
            h_y2 = min(H, bottom)

            # heatmap_region 引用原来的heatmaps[k]的一部分 只是一个视图
            heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
            gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]

            # 为什么不直接替换到 原来heatmap[k]上的?? 因为一个Heatmap上可以多个人 的 同一个关键点
            _ = np.maximum(
                heatmap_region, gaussian_regsion, out=heatmap_region)

    return heatmaps, keypoint_weights


def generate_unbiased_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints using `Dark Pose`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`Dark Pose`: https://arxiv.org/abs/1910.06278
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 无偏方案 3-sigma只是用来判断关键点是否在图中
    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    x = np.arange(0, W, 1, dtype=np.float32)
    y = np.arange(0, H, 1, dtype=np.float32)[:, None]

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = keypoints[n, k]
        # check that the gaussian has in-bounds part
        left, top = mu - radius
        right, bottom = mu + radius + 1

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        # 无偏的高斯分布 
        #   1. 还是确保中心的概率是1.0 (没有归一化) 
        #   2. 生成整个W*H heatmap大小的高斯图(不只是kernel size大小的)
        gaussian = np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2 * sigma**2))

        _ = np.maximum(gaussian, heatmaps[k], out=heatmaps[k])

    return heatmaps, keypoint_weights


def generate_udp_gaussian_heatmaps(
    heatmap_size: Tuple[int, int],
    keypoints: np.ndarray,
    keypoints_visible: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate gaussian heatmaps of keypoints using `UDP`_.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        sigma (float): The sigma value of the Gaussian heatmap

    Returns:
        tuple:
        - heatmaps (np.ndarray): The generated heatmap in shape
            (K, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (N, K)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # 3-sigma rule
    radius = sigma * 3

    # xy grid
    gaussian_size = 2 * radius + 1
    x = np.arange(0, gaussian_size, 1, dtype=np.float32)
    y = x[:, None]

    for n, k in product(range(N), range(K)):
        # skip unlabled keypoints
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = (keypoints[n, k] + 0.5).astype(np.int64)
        # check that the gaussian has in-bounds part
        left, top = (mu - radius).astype(np.int64)
        right, bottom = (mu + radius + 1).astype(np.int64)

        if left >= W or top >= H or right < 0 or bottom < 0:
            keypoint_weights[n, k] = 0
            continue

        mu_ac = keypoints[n, k]
        x0 = y0 = gaussian_size // 2
        x0 += mu_ac[0] - mu[0]
        y0 += mu_ac[1] - mu[1]
        gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

        # valid range in gaussian
        g_x1 = max(0, -left)
        g_x2 = min(W, right) - left
        g_y1 = max(0, -top)
        g_y2 = min(H, bottom) - top

        # valid range in heatmap
        h_x1 = max(0, left)
        h_x2 = min(W, right)
        h_y1 = max(0, top)
        h_y2 = min(H, bottom)

        heatmap_region = heatmaps[k, h_y1:h_y2, h_x1:h_x2]
        gaussian_regsion = gaussian[g_y1:g_y2, g_x1:g_x2]

        _ = np.maximum(heatmap_region, gaussian_regsion, out=heatmap_region)

    return heatmaps, keypoint_weights
