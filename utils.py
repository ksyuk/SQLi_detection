import csv
import time

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from transformers import BertForSequenceClassification


def load_dataset(dataset_path):
    dataset = []
    removed = 0
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) == 2:
                dataset.append(row)
            else:
                removed += 1
    print(f"removed: {removed}")
    return dataset


def count_model_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    with open('result.txt', 'a') as f:
        f.write(f'parameters: {num_params}\n')
        f.write('\n')


def train_model(network, dataloader, device, dataset_size, report_freq, hyperparams):
    print("traing...")
    network.train()

    optimizer = hyperparams["optimizer"]
    loss_fn = hyperparams["loss_fn"]
    if hyperparams["lr_scheduler"]:
        lr_scheduler = StepLR(optimizer, hyperparams["lr_scheduler"]["step_size"], hyperparams["lr_scheduler"]["gamma"])

    total_loss, accuracy = 0, 0
    count, epoch_count = 0, 0
    loss_values = []
    report_count = []

    start_time = time.time()

    while True:
        for i, (features, labels) in enumerate(dataloader, 1):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            out = network(features)
            if type(out) == int:
                continue
            loss = loss_fn(out, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(out, 1)
            accuracy += (predicted==labels).sum()
            count += len(labels)

            if i % report_freq == 0:
                print(f"{count}: accuracy={accuracy.item()/count}")
                loss_values.append(loss.item())
                report_count.append(count)

            if count > dataset_size * (epoch_count+1):
                print(f"{epoch_count+1} time epoch end")
                epoch_count += 1
                if hyperparams["lr_scheduler"]:
                    lr_scheduler.step()
                break

        if epoch_count == hyperparams["epoch"]:
            break

    end_time = time.time()
    with open('result.txt', 'a') as f:
        f.write(f'training_time: {end_time - start_time}\n')
        f.write('\n')

    plt.plot(report_count, loss_values)
    plt.xlabel('Count')
    plt.ylabel('Loss')
    plt.show()


def test_model(model, test_loader, device):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    with torch.no_grad():
        if model.__class__ == BertForSequenceClassification:
            for batch_data in test_loader:
                out = model(**batch_data)
                _, preds = torch.max(out.logits, 1)

                for label, pred in zip(batch_data['labels'], preds):
                    label = label.item()
                    pred = pred.item()
                    if label == 1:
                        if pred == 1:
                            true_positives += 1
                        else:
                            with open('false_negatives.txt', 'a') as f:
                                f.write(f'query: {raw_query}\n')
                            false_negatives += 1
                    else:
                        if pred == 0:
                            true_negatives += 1
                        else:
                            false_positives += 1
                            with open('false_positives.txt', 'a') as f:
                                f.write(f'query: {raw_query}\n')
        else:
            for queries, labels, raw_queries in test_loader:
                queries, labels = queries.to(device), labels.to(device)
                out = model(queries)
                _, preds = torch.max(out, 1)

                for label, pred, raw_query in zip(labels, preds, raw_queries):
                    label = label.item()
                    pred = pred.item()
                    if label == 1:
                        if pred == 1:
                            true_positives += 1
                        else:
                            false_negatives += 1
                            with open('false_negatives.txt', 'a') as f:
                                f.write(f'query: {raw_query}\n')
                    else:
                        if pred == 0:
                            true_negatives += 1
                        else:
                            false_positives += 1
                            with open('false_positives.txt', 'a') as f:
                                f.write(f'query: {raw_query}\n')

    with open('result.txt', 'a') as f:
        f.write(f'true_positives: {true_positives}\n')
        f.write(f'false_positives: {false_positives}\n')
        f.write(f'true_negatives: {true_negatives}\n')
        f.write(f'false_negatives: {false_negatives}\n')
        f.write('\n')
        f.write(f'accuracy: {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}\n')
        f.write(f'precision: {true_positives / (true_positives + false_positives)}\n')
        f.write(f'recall: {true_positives / (true_positives + false_negatives)}\n')
        f.write(f'f1: {2 * true_positives / (2 * true_positives + false_positives + false_negatives)}\n')
