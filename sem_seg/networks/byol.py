import torch
from torch import nn
import torch.nn.functional as F
from .network_utils import get_num_features
import copy


class BYOL_Head(nn.Module):
    def __init__(self, backbone: nn.Module, target_momentum=0.996):
        super().__init__()
        # representation head
        self.online_network = backbone
        self.target_network = copy.deepcopy(backbone)

        # Projection Head
        self.online_projector = ProjectorHead(backbone.name)
        self.target_projector = ProjectorHead(backbone.name)

        # Predictor Head
        self.predictor = MLPHead(self.online_projector.out_channels, 512, 128)

        self.m = target_momentum

    def initialize_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @staticmethod
    def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        loss = 2 - 2 * (x_norm * y_norm).sum(dim=-1)
        return loss.mean()


class ProjectorHead(nn.Module):
    def __init__(self, backbone_class: str, layer: str = 'feats5'):
        super().__init__()
        num_feats = get_num_features(backbone_class)
        layer_index = -3 if layer == 'feats3' else -2 if layer == 'feats4' else -1
        num_features = num_feats[layer_index]
        self.projection = MLPHead(num_features, 512, 128)
        self.out_channels = 128
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_pooled = self.avg_pool(x)
        h = x_pooled.view(x_pooled.shape[0], x_pooled.shape[1])   # removing the last dimension
        return self.projection(h)


class MLPHead(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        return self.net(x)


class BYOL:
    def __init__(self, backbone: nn.Module, target_momentum=0.996):
        super().__init__(backbone, target_momentum)

