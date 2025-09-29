import copy

import torch
import torch.nn as nn
from torch.nn.functional import one_hot


def get_bb_corners(bboxes_coords: torch.Tensor) -> torch.Tensor:

    xmin = bboxes_coords[..., 0] - bboxes_coords[..., 2] / 2
    ymin = bboxes_coords[..., 1] - bboxes_coords[..., 3] / 2
    xmax = bboxes_coords[..., 0] + bboxes_coords[..., 2] / 2
    ymax = bboxes_coords[..., 1] + bboxes_coords[..., 3] / 2

    bb_corners = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return bb_corners


def iou(bboxes1_coords: torch.Tensor, bboxes2_coords: torch.Tensor) -> torch.Tensor:
    xmin = torch.max(bboxes1_coords[..., 0], bboxes2_coords[..., 0])
    ymin = torch.max(bboxes1_coords[..., 1], bboxes2_coords[..., 1])
    xmax = torch.min(bboxes1_coords[..., 2], bboxes2_coords[..., 2])
    ymax = torch.min(bboxes1_coords[..., 3], bboxes2_coords[..., 3])

    area_bb1 = (bboxes1_coords[..., 2] - bboxes1_coords[..., 0]) * (bboxes1_coords[..., 3] - bboxes1_coords[..., 1])
    area_bb2 = (bboxes2_coords[..., 2] - bboxes2_coords[..., 0]) * (bboxes2_coords[..., 3] - bboxes2_coords[..., 1])

    # clamp(min=0) for the special case: intersection=0
    intersection = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
    union = area_bb1 + area_bb2 - intersection

    # add 1e-6 to avoid division by 0
    return intersection / (union + 1e-6)


class YOLO_Loss(nn.Module):

    def __init__(self, S, C, B, D, L_coord, L_noobj):
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.D = D
        self.L_coord = L_coord
        self.L_noobj = L_noobj

        self.register_buffer('pred_bb_ind', torch.arange(start=self.C, end=self.C + self.B * 5).reshape(self.B, 5))

    def forward(self, y_pred, y_gt):
        n = y_pred.shape[0]
        exists_obj_i = y_gt[..., 0:1]
        gt_bboxes_coords = y_gt[..., None, self.C + 1:]
        pred_bboxes_sqrt_coords = y_pred[..., self.pred_bb_ind[:, 1:]]

        gt_bboxes_scaled_coords = copy.deepcopy(gt_bboxes_coords.data)
        gt_bboxes_scaled_coords[..., :2] /= self.S
        gt_bboxes_coords_corners = get_bb_corners(gt_bboxes_scaled_coords)

        pred_bboxes_scaled_coords = copy.deepcopy(pred_bboxes_sqrt_coords.data)
        pred_bboxes_scaled_coords[..., :2] /= self.S
        pred_bboxes_scaled_coords[..., 2:] *= pred_bboxes_scaled_coords[..., 2:]
        pred_bboxes_coords_corners = get_bb_corners(pred_bboxes_scaled_coords)

        iou_scores = iou(gt_bboxes_coords_corners, pred_bboxes_coords_corners)
        max_iou_score, max_iou_index = torch.max(iou_scores, dim=-1)

        rmse_scores = torch.sqrt(torch.sum((gt_bboxes_scaled_coords - pred_bboxes_scaled_coords) ** 2, dim=-1))
        min_rmse_scores, min_rmse_index = torch.min(rmse_scores, dim=-1)
        rmse_mask = max_iou_score == 0

        best_index = max_iou_index
        best_index[rmse_mask] = min_rmse_index[rmse_mask]
        is_best_box = one_hot(best_index, self.B)

        exists_obj_ij = exists_obj_i * is_best_box
        exists_noobj_ij = 1 - exists_obj_ij

        # Localization Loss
        localization_center_loss = self.L_coord * torch.sum(exists_obj_ij[..., None] * (
                (gt_bboxes_coords[..., 0:2] - pred_bboxes_sqrt_coords[..., 0:2]) ** 2))

        localization_dims_loss = self.L_coord * torch.sum(exists_obj_ij[..., None] * (
                (torch.sqrt(gt_bboxes_coords[..., 2:4]) - pred_bboxes_sqrt_coords[..., 2:4]) ** 2))

        localization_loss = localization_center_loss + localization_dims_loss

        # Objectness Loss
        pred_bbox_cscores = y_pred[..., self.pred_bb_ind[:, 0]]

        objectness_obj_loss = torch.sum(exists_obj_ij * (iou_scores - pred_bbox_cscores) ** 2)
        objectness_noobj_loss = self.L_noobj * torch.sum(exists_noobj_ij * pred_bbox_cscores ** 2)

        objectness_loss = objectness_obj_loss + objectness_noobj_loss

        # Classification Loss
        pred_bboxes_class = y_pred[..., :self.C]
        gt_bboxes_class = y_gt[..., 1:self.C + 1]

        classification_loss = torch.sum(exists_obj_i * (gt_bboxes_class - pred_bboxes_class) ** 2)

        # Average YOLO Loss per instance
        total_loss = (localization_loss + objectness_loss + classification_loss) / n
        return total_loss
