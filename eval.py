import argparse
import time
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from model.util import *
from model.data import get_train_datasets, load_data, make_dataset
from model.model import SetConv
import csv
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
    first_elements, second_elements = zip(*vals)
    tensor1 = torch.tensor(first_elements)
    tensor2 = torch.tensor(second_elements)

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


def predict(model, data_loader, cuda):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        predicates,  targets,targets_ar, predicate_masks = data_batch

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





def eval(workload_name, num_queries, batch_size, hid_units, cuda):

    num_materialized_samples = 200
    dicts, column_min_max_vals, min_val, max_val,min_val_ar, max_val_ar, labels_train, labels_train_ar,labels_test,label_test_ar, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    column2vec, op2vec = dicts
    model = load_model("model/setconv_model1.pth", 8, 256, False)


    file_name = "workloads/" + workload_name
    joins, predicates, tables, samples, label,label_ar = load_data(file_name, num_materialized_samples)


    predicates_test = encode_data(predicates, column_min_max_vals, column2vec, op2vec)
    labels_test, min_val, max_val = normalize_labels(label)

    labels_test_ar, min_val_ar, max_val_ar = normalize_labels(label_ar)

    max_num_predicates = max([len(p) for p in predicates_test])


    # Get test set predictions

    test_data = make_dataset(predicates_test, labels_test,labels_test_ar, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    # Unnormalize
    preds_test_unnorm = unnormalize_torch_pre(preds_test, min_val, max_val,min_val_ar,max_val_ar)
    preds_rspn=preds_test_unnorm[0]
    preds_ar=preds_test_unnorm[1]
    # Print metrics
    # Write predictions
    file_name = "results/predictions_" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    rspn_pr_result= [int(x) for x in preds_rspn.tolist()]
    ar_pr_result = [int(x) for x in preds_ar.tolist()]
    print(rspn_pr_result)
    print(ar_pr_result)
    label = [int(x) for x in label]
    label_ar=[int(x) for x in label_ar]
    print("labels_rspn",label)
    print("laebls_ar",label_ar)
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_rspn[i]) + "," + str(preds_ar[i]) + "\n")
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))
    print("evaling finished")
    data = pd.read_csv('results/predictions_synthetic.csv', header=None)
    # 创建空的列表用于存储行数


    # 假设 rspn_pr_result 和 label 列表是这样的


    # 创建并写入 CSV 文件
    folder_path = './model_RSPN'
    file_path = os.path.join(folder_path, 'output.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["rspn_pr_result", "label_rspn"])  # 写入表头
        for i in range(len(rspn_pr_result)):
            writer.writerow([rspn_pr_result[i], label[i]])  # 写入每行数据

    folder_path = './model_AR'

    file_path = os.path.join(folder_path, 'output.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ar_pr_result", "label_ar"])  # 写入表头
        for i in range(len(rspn_pr_result)):
            writer.writerow([ar_pr_result[i], label_ar[i]])  # 写入每行数据

def load_model(model_path, predicate_feats, hid_units, cuda):
    model = SetConv(predicate_feats, hid_units)
    model.load_state_dict(torch.load(model_path))

    if cuda:
        model.cuda()
    model.eval()  # Set the model to evaluation mode
    return model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")

    eval("synthetic", 150000,  1024, 256, False)

    # Perform predictions

if __name__ == "__main__":
    main()
