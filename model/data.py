import csv
import torch
from torch.utils.data import dataset

from model.util import *


def load_data(file_name, num_materialized_samples):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []
    label_ar=[]
    # Load queries
    with open(file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            predicates.append(row[0].split(','))

            if int(row[1]) == 0:
                row[1]=2

            label.append(row[1])
            label_ar.append(row[2])
    print("Loaded queries")
    predicates = [list(chunks(d, 3)) for d in predicates]
    return joins, predicates, tables, samples, label ,label_ar
def load_and_encode_train_data(num_queries, num_materialized_samples):
    file_name_queries = "data/train"
    file_name_column_min_max_vals = "data/column_min_max_vals.csv"
    joins, predicates, tables, samples, label ,label_ar = load_data(file_name_queries, num_materialized_samples)
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)
    with open(file_name_column_min_max_vals, 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    predicates_enc = encode_data(predicates, column_min_max_vals, column2vec, op2vec)
    label_norm, min_val, max_val = normalize_labels(label) #这里对label_rspn进行归一化
    label_norm_ar, min_val_ar, max_val_ar = normalize_labels(label_ar)  # 这里对label_ar进行归一化
    # Split in training and validation samples
    num_train = int(num_queries * 0.9)
    num_test = num_queries - num_train

    predicates_train = predicates_enc[:num_train]
    labels_train = label_norm[:num_train]
    labels_train_ar = label_norm_ar[:num_train]

    predicates_test = predicates_enc[num_train:num_train + num_test]

    labels_test = label_norm[num_train:num_train + num_test]
    labels_test_ar= label_norm_ar[num_train:num_train + num_test]
    print("Number of training samples: {}".format(len(labels_train)))

    print("Number of validation samples: {}".format(len(labels_test)))


    max_num_predicates = max(max([len(p) for p in predicates_train]), max([len(p) for p in predicates_test]))

    dicts = [ column2vec, op2vec]
    train_data = [ predicates_train]
    test_data = [predicates_test]
    return dicts, column_min_max_vals, min_val, max_val,min_val_ar, max_val_ar, labels_train, labels_train_ar,labels_test, labels_test_ar,max_num_predicates, train_data, test_data


def make_dataset( predicates, labels, labels_ar,max_num_predicates):
    predicate_masks = []
    predicate_tensors = []
    for predicate in predicates:
        predicate_tensor = np.vstack(predicate)
        num_pad = max_num_predicates - predicate_tensor.shape[0]
        predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
        predicate_tensor = np.pad(predicate_tensor, ((0, num_pad), (0, 0)), 'constant')
        predicate_mask = np.pad(predicate_mask, ((0, num_pad), (0, 0)), 'constant')
        predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
        predicate_masks.append(np.expand_dims(predicate_mask, 0))
    predicate_tensors = np.vstack(predicate_tensors)
    predicate_tensors = torch.FloatTensor(predicate_tensors)
    predicate_masks = np.vstack(predicate_masks)
    predicate_masks = torch.FloatTensor(predicate_masks)


    target_tensor = torch.FloatTensor(labels)
    target_tensor_ar = torch.FloatTensor(labels_ar)

    return dataset.TensorDataset( predicate_tensors,  target_tensor,target_tensor_ar,  predicate_masks)


def get_train_datasets(num_queries, num_materialized_samples):
    dicts, column_min_max_vals, min_val, max_val,min_val_ar, max_val_ar, labels_train,labels_train_ar, labels_test, labels_test_ar, max_num_predicates, train_data, test_data = load_and_encode_train_data(
        num_queries, num_materialized_samples)
    train_dataset = make_dataset(*train_data, labels=labels_train,labels_ar=labels_train_ar,
                                 max_num_predicates=max_num_predicates)
    print("Created TensorDataset for training data")
    test_dataset = make_dataset(*test_data, labels=labels_test,labels_ar=labels_test_ar,
                                max_num_predicates=max_num_predicates)
    print("Created TensorDataset for validation data")
    return dicts, column_min_max_vals, min_val, max_val, min_val_ar, max_val_ar,labels_train, labels_train_ar, labels_test, labels_test_ar, max_num_predicates, train_dataset, test_dataset
#上面返回label_rspn和label_ar 以及各自对应的上下限  还有训练数据集