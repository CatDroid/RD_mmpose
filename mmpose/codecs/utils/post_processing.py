# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def get_simcc_normalized(batch_pred_simcc, sigma=None):
    """Normalize the predicted SimCC.

    Args:
        batch_pred_simcc (torch.Tensor): The predicted SimCC.
        sigma (float): The sigma of the Gaussian distribution.

    Returns:
        torch.Tensor: The normalized SimCC.
    """
    B, K, _ = batch_pred_simcc.shape
    #print(f"batch_pred_simcc.shape = {batch_pred_simcc.shape}")
    # batch_pred_simcc.shape = torch.Size([3, 3, 512]) [目标框, 关键点数目, 坐标]

    #print(f"sigma = {sigma}") # None
    # 如果有传入sigma 按照高斯分布归一化??? (代表传入的是高斯分布?)
    # Scale and clamp the tensor
    if sigma is not None:
        batch_pred_simcc = batch_pred_simcc / (sigma * np.sqrt(np.pi * 2))

    # 所有小于0的值设为0
    batch_pred_simcc = batch_pred_simcc.clamp(min=0)

    # 二值掩码: 如果batch_pred_simcc在最后一个维度上的最大值大于1，则该位置掩码为True，否则为False
    # Compute the binary mask
    mask = (batch_pred_simcc.amax(dim=-1) > 1).reshape(B, K, 1)

    # print(f"(batch_pred_simcc.amax(dim=-1) > 1) = {(batch_pred_simcc.amax(dim=-1) > 1).shape}") 
    # torch.Size([3, 3])  [目标框, 关键点数目] 
    # .reshape(B, K, 1) 扩展一个维度 可以用来广播  或者用 .unsqueeze(-1)

    # 将batch_pred_simcc除以它在最后一个维度上的最大值
    # Normalize the tensor using the maximum value
    norm = (batch_pred_simcc / batch_pred_simcc.amax(dim=-1).reshape(B, K, 1))

    # 使用torch.where函数，根据掩码mask的值来选择归一化后的值或原值。
    # 掩码为True的位置使用归一化后的值，掩码为False的位置保持原值不变  
    # ???? 
    # 掩码的作用正是为了确保只有在最后一个维度的最大值超过1时，才对该维度进行归一化处理

    # Apply normalization
    batch_pred_simcc = torch.where(mask, norm, batch_pred_simcc)

    return batch_pred_simcc


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray,
                      apply_softmax: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.
        
        SimCC 表示中获取最大响应的位置和值

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """

    # 每个目标框 调用一次
    # simcc_x = (1, 3, 512)   (N, K, Wx)
    # simcc_y = (1, 3, 512)   (N, K, Wy)

    assert isinstance(simcc_x, np.ndarray), ('simcc_x should be numpy.ndarray')
    assert isinstance(simcc_y, np.ndarray), ('simcc_y should be numpy.ndarray')
    assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
        f'Invalid shape {simcc_x.shape}')
    assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
        f'Invalid shape {simcc_y.shape}')
    assert simcc_x.ndim == simcc_y.ndim, (
        f'{simcc_x.shape} != {simcc_y.shape}')

    if simcc_x.ndim == 3:
        N, K, Wx = simcc_x.shape
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    else:
        N = None

    #print(f"simcc_x = {simcc_x.shape}") # (3, 512) 主要是N=1 reshape没有影响
    #print(f"simcc_y = {simcc_y.shape}") # (3, 512) 但是如果N不是1? 后面的 argmax axis=1 应该是不对的?

    # print(f"apply_softmax = {apply_softmax}")
    # apply_softmax = False
    # simcc_label.py 
    # decode 如果不设置 decode_visibility(需要返回可见性，而不只是置信度), 不会传入 apply_softmax=True
    if apply_softmax:
        # apply_softmax：布尔值，指示是否在 heatmap 上应用 softmax 函数，默认为 False
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        # 为什么softmax还要减去最大值?? 
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

    # locs：最大 heatmap 响应的位置
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)

    # vals：最大 heatmap 响应的值 (不一定要归一化? 比如最大概率是1?)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    #print(f"max_val_x = {max_val_x}") [1.3291085 1.1008768 1.3155584]
    #print(f"max_val_y = {max_val_y}") [0.6109808 1.0368669 0.8285344]

    # 布尔索引 x和y的置信度 取小的概率值 
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # 原来有N的维度 恢复N这个维度
    if N:
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)

    return locs, vals


def get_heatmap_3d_maximum(heatmaps: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap dimension: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, D, H, W) or
            (B, K, D, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 3) or (B, K, 3)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4 or heatmaps.ndim == 5, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 4:
        K, D, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, D, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    z_locs, y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(D, H, W))
    locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 3)
        vals = vals.reshape(B, K)

    return locs, vals


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def gaussian_blur1d(simcc: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    N, K, Wx = simcc.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        simcc[n, k] *= origin_max / np.max(simcc[n, k])
    return simcc


def batch_heatmap_nms(batch_heatmaps: Tensor, kernel_size: int = 5):
    """Apply NMS on a batch of heatmaps.

    Args:
        batch_heatmaps (Tensor): batch heatmaps in shape (B, K, H, W)
        kernel_size (int): The kernel size of the NMS which should be
            a odd integer. Defaults to 5

    Returns:
        Tensor: The batch heatmaps after NMS.
    """

    assert isinstance(kernel_size, int) and kernel_size % 2 == 1, \
        f'The kernel_size should be an odd integer, got {kernel_size}'

    padding = (kernel_size - 1) // 2

    maximum = F.max_pool2d(
        batch_heatmaps, kernel_size, stride=1, padding=padding)
    maximum_indicator = torch.eq(batch_heatmaps, maximum)
    batch_heatmaps = batch_heatmaps * maximum_indicator.float()

    return batch_heatmaps
