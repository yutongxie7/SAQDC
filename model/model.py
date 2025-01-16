# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class SetConv(nn.Module):
#     def __init__(self,  predicate_feats,  hid_units):
#         super(SetConv, self).__init__()
#
#         self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
#         self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
#         self.out_mlp1 = nn.Linear(hid_units * 1, hid_units)
#         self.out_mlp2 = nn.Linear(hid_units, 1)
#
#     def forward(self, predicates, predicate_mask):
#
#
#         hid_predicate = F.relu(self.predicate_mlp1(predicates))
#         hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
#         hid_predicate = hid_predicate * predicate_mask
#         hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
#         predicate_norm = predicate_mask.sum(1, keepdim=False)
#         hid_predicate = hid_predicate / predicate_norm
#
#
#         hid = hid_predicate
#         hid = F.relu(self.out_mlp1(hid))
#         out = torch.sigmoid(self.out_mlp2(hid))
#
#         return out
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define model architecture
class SetConv(nn.Module):
    def __init__(self, predicate_feats, hid_units):
        super(SetConv, self).__init__()

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 1, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 2)  # 输出层改为 2，表示两个预测值

    def forward(self, predicates, predicate_mask):
        # 对谓词特征进行嵌入处理
        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask  # 掩码操作
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)  # 对谓词求和
        predicate_norm = predicate_mask.sum(1, keepdim=False)  # 归一化
        hid_predicate = hid_predicate / predicate_norm

        # 全连接层
        hid = F.relu(self.out_mlp1(hid_predicate))
        out = torch.sigmoid(self.out_mlp2(hid))  # 输出两个归一化值

        return out
