# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 reduction='mean',
                 use_sigmoid=False):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid \
            else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction='none')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = (loss * target_weight)
        else:
            loss = self.criterion(output, target)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight


@MODELS.register_module()
class JSDiscretLoss(nn.Module):
    """Discrete JS Divergence loss for DSNT with Gaussian Heatmap.

    Modified from `the official implementation
    <https://github.com/anibali/dsntnn/blob/master/dsntnn/__init__.py>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
    """

    def __init__(
        self,
        use_target_weight=True,
        size_average: bool = True,
    ):
        super(JSDiscretLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.size_average = size_average
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def kl(self, p, q):
        """Kullback-Leibler Divergence."""

        eps = 1e-24
        kl_values = self.kl_loss((q + eps).log(), p)
        return kl_values

    def js(self, pred_hm, gt_hm):
        """Jensen-Shannon Divergence."""

        m = 0.5 * (pred_hm + gt_hm)
        js_values = 0.5 * (self.kl(pred_hm, m) + self.kl(gt_hm, m))
        return js_values

    def forward(self, pred_hm, gt_hm, target_weight=None):
        """Forward function.

        Args:
            pred_hm (torch.Tensor[N, K, H, W]): Predicted heatmaps.
            gt_hm (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        Returns:
            torch.Tensor: Loss value.
        """

        if self.use_target_weight:
            assert target_weight is not None
            assert pred_hm.ndim >= target_weight.ndim

            for i in range(pred_hm.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.js(pred_hm * target_weight, gt_hm * target_weight)
        else:
            loss = self.js(pred_hm, gt_hm)

        if self.size_average:
            loss /= len(gt_hm)

        return loss.sum()


@MODELS.register_module()
class KLDiscretLoss(nn.Module):
    """Discrete KL Divergence loss for SimCC with Gaussian Label Smoothing.
    Modified from `the official implementation.

    <https://github.com/leeyegy/SimCC>`_.
    Args:
        beta (float): Temperature factor of Softmax. Default: 1.0.
        label_softmax (bool): Whether to use Softmax on labels.
            Default: False.
        label_beta (float): Temperature factor of Softmax on labels.
            Default: 1.0.
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        mask (list[int]): Index of masked keypoints.
        mask_weight (float): Weight of masked keypoints. Default: 1.0.
    """

    def __init__(self,
                 beta=1.0,
                 label_softmax=False,
                 label_beta=10.0,
                 use_target_weight=True,
                 mask=None,
                 mask_weight=1.0):
        super(KLDiscretLoss, self).__init__()
        self.beta = beta
        self.label_softmax = label_softmax
        self.label_beta = label_beta
        self.use_target_weight = use_target_weight
        self.mask = mask
        self.mask_weight = mask_weight

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        """Criterion function."""
        # predict都会softmax， 根据beta温度软化
        log_pt = self.log_softmax(dec_outs * self.beta)
        if self.label_softmax:
            # GT根据参数 是否需要softmatx , 并且区分了 beta 和 label_beta (旧代码没有区分predict和label的温度?bug?)
            labels = F.softmax(labels * self.label_beta, dim=1)

        # log_pt = torch.Size([96, 512])  (B*K, width*spite) 或者  (B*K, height*spite)
        # labels = torch.Size([96, 512])

        # 为什么在width/height上(1d-heatmap)上做平均? 而不是求和得到这个1d-heatmap的KL散度(规约)
        # 沿着dim=1（即类别维度）计算平均值
        # 对每个样本('关键点x坐标' 作为一个样本)，我们得到了所有类别(bin上)的平均KL散度 
        #  一个样本 在所有类别(width个bin 或者 height个bin)上的 kl散度均值 
        loss = torch.mean(self.kl_loss(log_pt, labels), dim=1)

        # loss = torch.Size([96])  (B*K, )
        return loss

    def forward(self, pred_simcc, gt_simcc, target_weight):
        """Forward function.

        Args:
            pred_simcc (Tuple[Tensor, Tensor]): Predicted SimCC vectors of
                x-axis and y-axis.
            gt_simcc (Tuple[Tensor, Tensor]): Target representations.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.

        # 根据 distiller.py 的调用 传入是个元祖  第一个是x [B, K, width*spit], 第二个是y  
        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # target_weight (torch.Tensor[N, K]  每个关键点的weight [N, K]   N就是Batch

        """
        N, K, _ = pred_simcc[0].shape
        loss = 0

        # 是否使用weight
        if self.use_target_weight:
            # 输出是一个一维张量，包含了 target_weight 中的所有元素。
            # 新张量的长度等于原张量中元素的总数。
            # torch.Size([32, 3]) ->  torch.Size([96])  (B*K, )
            weight = target_weight.reshape(-1)
        else:
            weight = 1.


        for pred, target in zip(pred_simcc, gt_simcc):
            # 遍历元祖  pred_simcc gt_simcc是个元祖 分别代表x和y
            # B=32
            # K=3
            # input_size=(256, 256), * simcc_split_ratio=2.0,  = 512
            
            # [B, K, width*split  ] 
            # [B, K, height*split ]

            # pred   = torch.Size([32, 3, 512])
            # target = torch.Size([32, 3, 512])

            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))

            # pred   = torch.Size([96, 512])  # 相当于 合并了3个关键点的x或者y坐标  32*3 B*K 
            # target = torch.Size([96, 512])  # (B*K, width)

            # weight = torch.Size([96])  # (B*K, )
            t_loss = self.criterion(pred, target).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            # 所有关键点(有效)求和 
            loss = loss + t_loss.sum()

        # 平均到每种关键点(不是K*2 x,y坐标 也不是每个关键点)
        return loss / K


@MODELS.register_module()
class InfoNCELoss(nn.Module):
    """InfoNCE loss for training a discriminative representation space with a
    contrastive manner.

    `Representation Learning with Contrastive Predictive Coding
    arXiv: <https://arxiv.org/abs/1611.05424>`_.

    Args:
        temperature (float, optional): The temperature to use in the softmax
            function. Higher temperatures lead to softer probability
            distributions. Defaults to 1.0.
        loss_weight (float, optional): The weight to apply to the loss.
            Defaults to 1.0.
    """

    def __init__(self, temperature: float = 1.0, loss_weight=1.0) -> None:
        super(InfoNCELoss, self).__init__()
        assert temperature > 0, f'the argument `temperature` must be ' \
                                f'positive, but got {temperature}'
        self.temp = temperature
        self.loss_weight = loss_weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes the InfoNCE loss.

        Args:
            features (Tensor): A tensor containing the feature
                representations of different samples.

        Returns:
            Tensor: A tensor of shape (1,) containing the InfoNCE loss.
        """
        n = features.size(0)
        features_norm = F.normalize(features, dim=1)
        logits = features_norm.mm(features_norm.t()) / self.temp
        targets = torch.arange(n, dtype=torch.long, device=features.device)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        return loss * self.loss_weight


@MODELS.register_module()
class VariFocalLoss(nn.Module):
    """Varifocal loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        alpha (float): A balancing factor for the negative part of
            Varifocal Loss. Defaults to 0.75.
        gamma (float): Gamma parameter for the modulating factor.
            Defaults to 2.0.
    """

    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.,
                 reduction='mean',
                 alpha=0.75,
                 gamma=2.0):
        super().__init__()

        assert reduction in ('mean', 'sum', 'none'), f'the argument ' \
            f'`reduction` should be either \'mean\', \'sum\' or \'none\', ' \
            f'but got {reduction}'

        self.reduction = reduction
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def criterion(self, output, target):
        label = (target > 1e-4).to(target)
        weight = self.alpha * output.sigmoid().pow(
            self.gamma) * (1 - label) + target
        output = output.clip(min=-10, max=10)
        vfl = (
            F.binary_cross_entropy_with_logits(
                output, target, reduction='none') * weight)
        return vfl

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss = (loss * target_weight)
        else:
            loss = self.criterion(output, target)

        loss[torch.isinf(loss)] = 0.0
        loss[torch.isnan(loss)] = 0.0

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss * self.loss_weight
