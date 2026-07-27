# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
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


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
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
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox



import math
def Gaussian(y, mu, var):
    eps = 0.3
    epsilon = 1e-6
    result = (y-mu)/(var+epsilon)
    result = (result**2)/2*(-1)
    exp = torch.exp(result)
    result = exp/(math.sqrt(2*math.pi))/(var + eps)

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

@LOSSES.register_module()
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