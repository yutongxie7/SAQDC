import argparse
import time
import os
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

def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(abs(preds[i] - targets[i]))
        else:
            qerror.append(abs(targets[i] - preds[i]))
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





def eval(workload_name, num_queries, batch_size, hid_units, cuda):

    num_materialized_samples = 1000
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_predicates, train_data, test_data = get_train_datasets(
        num_queries, num_materialized_samples)
    column2vec, op2vec = dicts
    model = load_model("model/setconv_model.pth", 8, 256, False)


    file_name = "workloads/" + workload_name
    joins, predicates, tables, samples, label = load_data(file_name, num_materialized_samples)

    # Get feature encoding and proper normalization
    # samples_test = encode_samples(tables, samples, table2vec)
    predicates_test = encode_data(predicates, column_min_max_vals, column2vec, op2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)


    max_num_predicates = max([len(p) for p in predicates_test])


    # Get test set predictions
    test_data = make_dataset(predicates_test, labels_test, max_num_predicates)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    preds_test, t_total = predict(model, test_data_loader, cuda)
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)
    # Print metrics
    # Write predictions
    file_name = "results/predictions_" + workload_name + ".csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + str(label[i]) + "\n")
    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))
    print("evaling finished")
def load_model(model_path, predicate_feats, hid_units, cuda):
    model = SetConv(predicate_feats, hid_units)
    model.load_state_dict(torch.load(model_path))
    if cuda:
        model.cuda()
    model.eval()  # Set the model to evaluation mode
    return model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testset", help="synthetic, scale, or job-light")
    parser.add_argument("--queries", help="number of training queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--epochs", help="number of epochs (default: 10)", type=int, default=10)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--cuda", help="use CUDA", action="store_true")

    eval("synthetic", 150000,  1024, 256, False)

    # Perform predictions

if __name__ == "__main__":
    main()
