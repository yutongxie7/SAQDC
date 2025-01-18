import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.util import *
from model.data import get_train_datasets, load_data, make_dataset
from model.model import SetConv

def unnormalize_torch(vals, min_val, max_val):

    # 反归一化
    vals = (vals * (max_val - min_val)) + min_val

    # 判断是否需要处理负值
    if min_val < 0:
        # 对于负值，使用相应的反归一化逻辑
        neg_mask = vals < 0
        # 对于负值，使用 1 / exp(-vals) 来恢复
        vals = torch.where(neg_mask, -torch.exp(-vals), torch.exp(vals))
    else:
        # 对于非负值，直接使用指数还原
        vals = torch.exp(vals)

    return vals
def unnormalize_torch_pre(vals, min_val, max_val,min_val_ar, max_val_ar):
    tensor1 = vals[:, 0].unsqueeze(1)  # 获取第一列并增加维度
    tensor2 = vals[:, 1].unsqueeze(1)  # 获取第二列并增加维度
    # 反归一化
    tensor1 = (tensor1 * (max_val - min_val)) + min_val
    # 判断是否需要处理负值
    if min_val < 0:
        # 对于负值，使用相应的反归一化逻辑
        neg_mask = tensor1 < 0
        # 对于负值，使用 1 / exp(-vals) 来恢复
        tensor1 = torch.where(neg_mask, -torch.exp(-tensor1), torch.exp(tensor1))
    else:
        # 对于非负值，直接使用指数还原
        tensor1 = torch.exp(tensor1)

    tensor2 = (tensor2 * (max_val_ar - min_val_ar)) + min_val_ar
    # 判断是否需要处理负值
    if min_val_ar < 0:
        # 对于负值，使用相应的反归一化逻辑
        neg_mask = tensor2 < 0
        # 对于负值，使用 1 / exp(-vals) 来恢复
        tensor2 = torch.where(neg_mask, -torch.exp(-tensor2), torch.exp(tensor2))
    else:
        # 对于非负值，直接使用指数还原
        tensor2 = torch.exp(tensor2)

    return tensor1,tensor2

def qerror_loss(preds, targets, targets_ar,min_val, max_val,min_val_ar, max_val_ar):
    qerror = []
    preds = unnormalize_torch_pre(preds, min_val, max_val,min_val_ar, max_val_ar)
    targets = unnormalize_torch(targets, min_val, max_val)
    targets_ar = unnormalize_torch(targets_ar, min_val_ar, max_val_ar)
    rspn=preds[0]
    ar=preds[1]
    for i in range(len(targets)):

        if (rspn[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(abs(rspn[i] - targets[i]))
        else:
            qerror.append(abs(targets[i] - rspn[i]))

    for i in range(len(targets_ar)):

        if (ar[i] > targets_ar[i]).cpu().data.numpy()[0]:
            qerror.append(abs(ar[i] - targets_ar[i]))
        else:
            qerror.append(abs(targets_ar[i] - ar[i]))
    return torch.mean(torch.cat(qerror))

def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.
    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        predicates,  targets, predicate_masks = data_batch

        if cuda:
            predicates,targets = predicates.cuda(),targets.cuda()
            predicate_masks= predicate_masks.cuda()
        predicates, targets = Variable(predicates), Variable(
            targets)
        predicate_masks = Variable(predicate_masks)

        t = time.time()
        outputs = model(predicates,predicate_masks)
        t_total += time.time() - t

        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])

    return preds, t_total


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        print(preds_unnorm[i],"和",labels_unnorm[i])
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i][0] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i][0] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i][0]))


def train_(workload_name, num_queries, num_epochs, batch_size, hid_units, cuda):

    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, min_val_ar, max_val_ar,labels_train,labels_train_ar, labels_test, labels_test_ar,max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    column2vec, op2vec = dicts
    predicate_feats = len(column2vec) + len(op2vec) + 1
    model = SetConv( predicate_feats,  hid_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if cuda:
        model.cuda()
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    model.train()
    for epoch in range(num_epochs):
        loss_total = 0.
        for batch_idx, data_batch in enumerate(train_data_loader):
            predicates, targets, targets_ar, predicate_masks = data_batch
            if cuda:
                predicates, targets ,targets_ar= predicates.cuda(),  targets.cuda() ,targets_ar.cuda()
                predicate_masks = predicate_masks.cuda(),
            predicates,  targets ,targets_ar= Variable(predicates),  Variable(
                targets) ,Variable(
                targets_ar)
            predicate_masks = Variable(predicate_masks)

            optimizer.zero_grad()
            outputs = model(predicates, predicate_masks)


            loss = qerror_loss(outputs, targets.float(),targets_ar.float(), min_val, max_val,min_val_ar,max_val_ar)

            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))

   # Save the model
    model_path = "model/setconv_model1.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")

    train_("synthetic", 150000, 200, 1024, 256, False)


if __name__ == "__main__":
    main()
