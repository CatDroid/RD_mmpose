# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToPILImage


class SimCCVisualizer:

    def draw_instance_xy_heatmap(self,
                                 heatmap: torch.Tensor,
                                 overlaid_image: Optional[np.ndarray],
                                 n: int = 20,
                                 mix: bool = True,
                                 weight: float = 0.5):
        """Draw heatmaps of GT or prediction.

        Args:
            heatmap (torch.Tensor): Tensor of heatmap.
            overlaid_image (np.ndarray): The image to draw.
            n (int): Number of keypoint, up to 20.
            mix (bool):Whether to merge heatmap and original image.
            weight (float): Weight of original image during fusion.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        # mmpose/visualization/local_visualizer.py self._draw_instance_xy_heatmap

        heatmap2d = heatmap.data.max(0, keepdim=True)[0]
        # 返回 [{x:0,y:0}, {x:0,y:0}, ...] 和关键点数目
        xy_heatmap, K = self.split_simcc_xy(heatmap)
        K = K if K <= n else n
        # heatmap = [K, H, W]
        # blank_size = (H, W)
        blank_size = tuple(heatmap.size()[1:])
        # 字典 存放 x和y方向的 1d_heatmaps 
        maps = {'x': [], 'y': []}
        # 遍历数组 
        # maps = {'x': [0,0,0..], 'y':[0,0,0..] }  # 把关键点的x坐标 和 y坐标 的1d-heatmap 分开存放
        for i in xy_heatmap:
            # 一个关键点的 x和y坐标 鸽子调用一次 draw_1d_heatmaps
            x, y = self.draw_1d_heatmaps(i['x']), self.draw_1d_heatmaps(i['y'])
            maps['x'].append(x)
            maps['y'].append(y)
        white = self.creat_blank(blank_size, K)
        # 将2d的heatmap转换成伪彩色 
        map2d = self.draw_2d_heatmaps(heatmap2d)
        if mix:
            # dst = cv.addWeighted(src1, alpha, src2, beta, gamma)
            # dst(i,j)=src1(i,j)×α+src2(i,j)×β+γ
            # 将 image 和 伪彩色heatmap 两幅图像进行 线性加权求和
            # gamma: 加到最终输出图像上的标量值（用于亮度调整）
            map2d = cv.addWeighted(overlaid_image, 1 - weight, map2d, weight,
                                   0)

        # 把map2d画到white画布上 并且开始位置是 heatmapsize*0.1, create_blank有预留 也就是左上边 都会留位置
        self.image_cover(white, map2d, int(blank_size[1] * 0.1),
                         int(blank_size[0] * 0.1))
        
        # 把1d-hetmap(maps)画到画布(white)上
        white = self.add_1d_heatmaps(maps, white, blank_size, K)
        return white

    # Union联合体 支持ndarray和Tensor 
    def split_simcc_xy(self, heatmap: Union[np.ndarray, torch.Tensor]):
        """Extract one-dimensional heatmap from two-dimensional heatmap and
        calculate the number of keypoint."""
        size = heatmap.size()
        k = size[0] if size[0] <= 20 else 20
        maps = []
        for _ in range(k):
            xy_dict = {}
            # 取出一个关键点的2d-heatmap
            single_heatmap = heatmap[_]
            xy_dict['x'], xy_dict['y'] = self.merge_maps(single_heatmap)
            maps.append(xy_dict)
        # map = [{x:(1,W),y:(H,1)}, {x:(1,W),y:(H,1)}, ...] 长度是关键点的数目
        return maps, k

    def merge_maps(self, map_2d):
        """Synthesis of one-dimensional heatmap."""
        # ?? map_2d 的维度为 (H, W) 
        # max(0, keepdim=True) 表示(沿着)在第0维度（行）上进行最大值计算，并保留维度。(1, W) 
        # x 每列最大的 保存成 （1，W)   这个就是1-d heatmap simcc的表示方式 'x坐标'表示成为'x向量'
        x = map_2d.data.max(0, keepdim=True)[0]
        # y 每行最大的 保存成  (H, 1)  'y坐标'表示成为'y向量'
        y = map_2d.data.max(1, keepdim=True)[0]
        # (1, W) 和 (H, 1)
        return x, y

    def draw_1d_heatmaps(self, heatmap_1d):
        """Draw one-dimensional heatmap."""
        size = heatmap_1d.size()
        # 比如是(1, W) 或者 是(H, 1)
        length = max(size)
        np_heatmap = ToPILImage()(heatmap_1d).convert('RGB')
        #  相当于灰度图 装成 RGB图
        cv_img = cv.cvtColor(np.asarray(np_heatmap), cv.COLOR_RGB2BGR)
        # 是x坐标向量 还是y坐标向量 
        if size[0] < size[1]:
            # 最尽邻居插值到 15宽 (方便显示)
            cv_img = cv.resize(cv_img, (length, 15))
        else:
            cv_img = cv.resize(cv_img, (15, length))
        # 灰度图转换成伪彩色
        single_map = cv.applyColorMap(cv_img, cv.COLORMAP_JET)
        return single_map

    def creat_blank(self,
                    size: Union[list, tuple],
                    K: int = 20,
                    interval: int = 10):
        """Create the background."""
        #  size = (H, W) heatmap的宽高 
        #  K 关键点数量
        blank_height = int(
            max(size[0] * 2, size[0] * 1.1 + (K + 1) * (15 + interval)))
        #  最大 不超过 heatmap高的两倍   高*1.1 + (K+1个关键点 每个占用15+10的高度 15是填充伪彩色 10是过度带)
        blank_width = int(
            max(size[1] * 2, size[1] * 1.1 + (K + 1) * (15 + interval)))

        # RGB通道的 uint8 全白色的 画布
        blank = np.zeros((blank_height, blank_width, 3), np.uint8)
        blank.fill(255)
        return blank

    def draw_2d_heatmaps(self, heatmap_2d):
        """Draw a two-dimensional heatmap fused with the original image."""
        np_heatmap = ToPILImage()(heatmap_2d).convert('RGB')
        cv_img = cv.cvtColor(np.asarray(np_heatmap), cv.COLOR_RGB2BGR)
        map_2d = cv.applyColorMap(cv_img, cv.COLORMAP_JET)
        return map_2d

    def image_cover(self, background: np.ndarray, foreground: np.ndarray,
                    x: int, y: int):
        """Paste the foreground on the background."""
        # 把 foreground 画到 background 的 (x,y)开始的位置上 
        fore_size = foreground.shape
        background[y:y + fore_size[0], x:x + fore_size[1]] = foreground
        return background

    def add_1d_heatmaps(self,
                        maps: dict,
                        background: np.ndarray,
                        map2d_size: Union[tuple, list],
                        K: int,
                        interval: int = 10):
        """Paste one-dimensional heatmaps onto the background in turn."""

        # map2d_size = (H, W) of heatmap 

        y_startpoint, x_startpoint = [int(1.1*map2d_size[1]),
                                      int(0.1*map2d_size[0])],\
                                     [int(0.1*map2d_size[1]),
                                      int(1.1*map2d_size[0])]
        # 因为 画布(background)有预留0.1*heatmapsize在左和上边 所以x向量 开始画的坐标在 (x,y) = (0.1*heatmap_W  1.1*heatmap_H) 在画布的下方绘制 
        # 预留空间 K+1个 (interval + 15)
        # 这里从 interval * 2 开始  后续每个偏移 interval + 10  最后一个就是 interval * 2 + (K-1)*(interval + 10) = (K+1)*interval + (K-1)*10  .. +15

        # 第一个应该是从 10+15/2=10+7.5=17.5开始绘制 ? cv绘制坐标是左上角 ? 这样 每个绘制15个像素  跳interval(10) + 10=20, 相当于间隔只有5个分辨率? --看起来是这样
        x_startpoint[1] += interval * 2
        y_startpoint[0] += interval * 2
        add = interval + 10
        for i in range(K):
            # 遍历所有关键点 maps['x'] 和 mapx['y'] 都有K个向量 

            # x_startpoint[0] 就是 绘制的x坐标 x_startpoint[1] 就是绘制的y坐标  
            self.image_cover(background, maps['x'][i], x_startpoint[0],
                             x_startpoint[1])
            cv.putText(background, str(i),
                       (x_startpoint[0] - 30, x_startpoint[1] + 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 文字只是显示 第几个关键点 没有什么作用
            
            self.image_cover(background, maps['y'][i], y_startpoint[0],
                             y_startpoint[1])
            cv.putText(background, str(i),
                       (y_startpoint[0], y_startpoint[1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # x向量绘制 只要增加y坐标 
            x_startpoint[1] += add
            y_startpoint[0] += add
        return background[:x_startpoint[1] + y_startpoint[1] +
                          1, :y_startpoint[0] + x_startpoint[0] + 1]
