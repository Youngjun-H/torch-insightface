"""
Loss functions for ArcFace
"""

import math

import torch


class CombinedMarginLoss(torch.nn.Module):
    """
    Combined Margin Loss for ArcFace
    Supports ArcFace (m1=1.0, m3=0.0) and CosFace (m3>0)
    """

    def __init__(
        self,
        s,
        m1,
        m2,
        m3,
        interclass_filtering_threshold=0,
    ):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        # Easy margin: target logit이 낮을 때는 margin을 적용하지 않음 (초기 학습 단계에서 안정적)
        self.easy_margin = True

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones(
                    [index_positive.size(0), logits.size(1)], device=logits.device
                )
                # labels[index_positive]가 [N, 1] shape일 수 있으므로 squeeze 또는 view 사용
                # scatter_는 index가 [N] 또는 [N, 1] shape이어야 하는데,
                # labels가 [N, 1]이면 squeeze해서 [N]으로 만든 후 unsqueeze(1)로 [N, 1]로 만들어야 함
                labels_positive = labels[index_positive]
                if labels_positive.dim() > 1:
                    labels_positive = labels_positive.squeeze(1)
                # scatter_는 dim=1일 때 index가 [N, 1] shape을 기대함
                mask.scatter_(1, labels_positive.unsqueeze(1), 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        if self.m1 == 1.0 and self.m3 == 0.0:
            # ArcFace: target logit에만 margin 적용
            # 표준 ArcFace 구현: target logit만 arccos → margin 추가 → cos
            with torch.no_grad():
                # Target logit만 추출
                target_logit = logits[index_positive, labels[index_positive].view(-1)]

                # 수치 안정성을 위해 -1+eps ~ 1-eps 범위로 제한
                eps = 1e-7
                target_logit_clamped = target_logit.clamp(-1 + eps, 1 - eps)

                # Target logit에만 margin 적용: arccos → margin 추가 → cos
                target_angle = target_logit_clamped.arccos()

                # Easy margin: target logit이 theta보다 작을 때는 margin을 적용하지 않음
                # theta = cos(pi - m2)
                if self.easy_margin:
                    # target_logit > theta일 때만 margin 적용
                    theta = self.theta  # cos(pi - m2)
                    target_angle_with_margin = torch.where(
                        target_logit > theta, target_angle + self.m2, target_angle
                    )
                else:
                    # Standard ArcFace: 항상 margin 적용
                    target_angle_with_margin = target_angle + self.m2

                target_logit_with_margin = target_angle_with_margin.cos()

                # Target logit만 교체 (나머지 logits는 그대로 유지)
                # dtype 일치: logits가 bf16이면 target_logit_with_margin도 bf16으로 변환
                target_logit_with_margin = target_logit_with_margin.to(
                    dtype=logits.dtype
                )
                logits[index_positive, labels[index_positive].view(-1)] = (
                    target_logit_with_margin
                )

            # 모든 logits에 scale 적용
            logits = logits * self.s

        elif self.m3 > 0:
            # CosFace
            target_logit = logits[index_positive, labels[index_positive].view(-1)]
            final_target_logit = target_logit - self.m3
            # dtype 일치: logits가 bf16이면 final_target_logit도 bf16으로 변환
            final_target_logit = final_target_logit.to(dtype=logits.dtype)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise ValueError(
                f"Unsupported margin configuration: m1={self.m1}, m3={self.m3}"
            )

        return logits
