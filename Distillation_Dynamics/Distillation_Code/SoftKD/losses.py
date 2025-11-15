import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, args):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert args.distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = args.distillation_type
        self.tau = args.distillation_tau

        self.layer_ids_s = args.s_id
        self.layer_ids_t = args.t_id
        self.alpha = args.distillation_alpha
        self.beta = args.distillation_beta

    def forward(self, inputs, outputs, labels):
        block_outs_s = outputs[1]
        student_outputs = outputs[0]
        base_loss = self.base_criterion(student_outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        # don't backprop through the teacher
        with torch.no_grad():
            teacher_outputs, block_outs_t = self.teacher_model(inputs, self.layer_ids_t)

        loss_base = (1-self.alpha) * base_loss


        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(student_outputs, teacher_outputs.argmax(dim=1))
        loss_dist = self.alpha * distillation_loss

        return loss_base, loss_dist



