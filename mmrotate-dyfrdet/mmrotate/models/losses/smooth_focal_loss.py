# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES


def smooth_focal_loss(pred,
                      target,
                      weight=None,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None):
    """Smooth Focal Loss proposed in Circular Smooth Label (CSL).

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The reduction method used to
            override the original reduction method of the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@ROTATED_LOSSES.register_module()
class SmoothFocalLoss(nn.Module):
    """Smooth Focal Loss. Implementation of `Circular Smooth Label (CSL).`__

    __ https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40

    Args:
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SmoothFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * smooth_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls


import math
def Gaussian(y, mu, var):
    eps = 0.3
    epsilon = 1e-6
    result = (y-mu)/(var+epsilon)
    result = (result**2)/2*(-1)
    exp = torch.exp(result)
    result = exp/(math.sqrt(2*math.pi))/(var + eps)
    result = torch.clamp(result, min=0.0, max=1.0)
    return result

def NLL_loss(bbox_gt, bbox_pred, bbox_var):
        prob = Gaussian(bbox_gt, bbox_pred, bbox_var)

        return prob

@weighted_loss
def nll_gaussian(pred, target):
    p_mu, p_var = pred
    balance = 2.0
    eps = torch.finfo(torch.float32).eps
    if target.numel() == 0:
        return p_mu.sum() * 0
    prob = Gaussian(target, p_mu, p_var)
    loss_unc = -torch.log(prob + eps) / balance

    return loss_unc
# @ROTATED_LOSSES.register_module()
# class UncBoxLoss(nn.Module):
#     def __init__(self,use_gpu=True, loss_weight = 1.0, reduction = 'mean'):
#         super(UncBoxLoss, self).__init__()
#         self.use_gpu = use_gpu
#         self.loss_weight = loss_weight
#         self.reduction = reduction

#     def forward(self, p_mu, p_var, target, weight = None, avg_factor = None, reduction_override = None, **kwargs):
#         """
#         :param p_mu: [N,4]
#         :param p_var: [N,4]
#         :param target: [N,4]
#         :param weight:
#         :param avg_factor: N
#         :param reduction_override:'mean'
#         :return:
#         """
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         assert p_mu.shape[0] == p_var.shape[0] == target.shape[0]
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss = self.loss_weight * nll_gaussian(
#             [p_mu,p_var],
#             target,
#             weight,
#             reduction = reduction,
#             avg_factor = avg_factor,
#             **kwargs)
#         return loss
@ROTATED_LOSSES.register_module()
class UncBoxLoss(nn.Module):
    def __init__(self,use_gpu=True, loss_weight = 1.0, reduction = 'mean', omega = False, vareps = 0.0, rho = 0, p = 0):
        super(UncBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.omega = omega
        self.vareps = vareps
        self.rho = rho
        self.p = p
    def generate_omega_weight(self, var):
        weight = torch.ones_like(var)
        weight[var <= self.vareps] = 1.0
        d = (var - self.rho) / (1.0 - self.rho)
        weight[var > self.rho] = self.vareps + (1 - self.vareps) * (1 - d[var > self.rho] ** self.p)
        return weight

    def forward(self, p_mu, p_var, target, weight = None, avg_factor = None, reduction_override = None, **kwargs):
        """
        :param p_mu: [N,4]
        :param p_var: [N,4]
        :param target: [N,4]
        :param weight:
        :param avg_factor: N
        :param reduction_override:'mean'
        :return:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert p_mu.shape[0] == p_var.shape[0] == target.shape[0]
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.omega:
            omega_weight = self.generate_omega_weight(p_var)
            loss = self.loss_weight * omega_weight * nll_gaussian(
                [p_mu,p_var],
                target,
                weight,
                reduction = reduction,
                avg_factor = avg_factor,
                **kwargs)
        else:
            loss = self.loss_weight * nll_gaussian(
                [p_mu,p_var],
                target,
                weight,
                reduction = reduction,
                avg_factor = avg_factor,
                **kwargs)
        return loss

@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

@ROTATED_LOSSES.register_module()
class UncBoxLoss_SmoothL1(nn.Module):
    def __init__(self,use_gpu=True, beta = 0.11111111111, loss_weight = 1.0, reduction = 'mean', ratio = 0.5):
        super(UncBoxLoss_SmoothL1, self).__init__()
        self.use_gpu = use_gpu
        self.loss_weight = loss_weight
        self.beta = beta
        self.reduction = reduction
        self.ratio = ratio

    def forward(self, p_mu, p_var, target, weight = None, avg_factor = None, reduction_override = None, **kwargs):
        """
        :param p_mu: [N,4]
        :param p_var: [N,4]
        :param target: [N,4]
        :param weight:
        :param avg_factor: N
        :param reduction_override:'mean'
        :return:
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert p_mu.shape[0] == p_var.shape[0] == target.shape[0]
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_unc = self.loss_weight * nll_gaussian(
            [p_mu,p_var],
            target,
            weight,
            reduction = reduction,
            avg_factor = avg_factor,
            **kwargs)

        loss_l1 = self.loss_weight * smooth_l1_loss(
            p_mu, target, weight, beta = self.beta, reduction=reduction, avg_factor=avg_factor)

        loss = self.ratio * loss_unc + (1 - self.ratio) * loss_l1
        return loss