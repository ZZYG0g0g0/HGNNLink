import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData

class HGTModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, metadata, heads=8, num_layers=2):
        super(HGTModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 多层 HGTConv
        for i in range(num_layers):
            conv = HGTConv(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                metadata,
                heads=heads,
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 全连接层
        self.fc1 = nn.Linear(hidden_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x_dict, edge_index_dict):
        for conv, bn in zip(self.convs, self.bns):
            x_dict = conv(x_dict, edge_index_dict)
            for key in x_dict.keys():
                x = x_dict[key]
                x = bn(x)
                x = F.relu(x)
                x_dict[key] = x

        # 返回更新后的 x_dict，以便 Classifier 使用
        return x_dict


class Classifier(torch.nn.Module):
    def forward(self, x_req: Tensor, x_code: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_req = x_req[edge_label_index[0]]
        edge_feat_code = x_code[edge_label_index[1]]
        return torch.sigmoid((edge_feat_req * edge_feat_code).sum(dim=-1))


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super(Model, self).__init__()
        self.hgt = HGTModel(in_channels, out_channels, metadata)  # 使用 HGTModel 替代 HAN
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "req": data["req"].x,
            "code": data["code"].x,
        }
        x_dict = self.hgt(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["req"],
            x_dict["code"],
            data["req", "link", "code"].edge_label_index,
        )
        return pred