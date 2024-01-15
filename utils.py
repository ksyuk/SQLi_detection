import csv

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


def load_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            dataset.append(row)
    return dataset


def train_model(network, dataloader, device, dataset_size, report_freq, hyperparams):
    print("traing...")
    network.train()

    optimizer = hyperparams["optimizer"] or torch.optim.Adam(network.parameters(), lr=hyperparams["learning_rate"])
    loss_fn = hyperparams["loss_fn"] or nn.NLLLoss()
    if hyperparams["lr_scheduler"]:
        lr_scheduler = StepLR(optimizer, hyperparams["lr_scheduler"]["step_size"], hyperparams["lr_scheduler"]["gamma"])

    total_loss, accuracy = 0, 0
    count, epoch_count = 0, 0
    loss_values = []

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

            if count > dataset_size * (epoch_count+1):
                print(f"{epoch_count+1} time epoch end")
                loss_values.append(loss.item())
                epoch_count += 1
                if hyperparams["lr_scheduler"]:
                    lr_scheduler.step()
                break

        if epoch_count == hyperparams["epoch"]:
            break

    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return total_loss/count, accuracy.item()/count


def test_model(model, test_loader, device):
    with torch.no_grad():
        for query, label in test_loader:
            query, label = query.to(device), label.to(device)
            pred = model(query)

            if label == "1":
                if pred == "1":
                    true_positives += 1
                else:
                    false_negatives += 1
                    with open('false_negatives.txt', 'a') as f:
                        f.write(f'query: {query}\n')
                        f.write(f'output: {pred}\n')
                        f.write('\n')
            else:
                if pred == "0":
                    true_negatives += 1
                else:
                    false_positives += 1
                    with open('false_positives.txt', 'a') as f:
                        f.write(f'query: {query}\n')
                        f.write(f'output: {pred}\n')
                        f.write('\n')
