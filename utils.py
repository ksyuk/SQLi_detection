import csv

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


def train(network, dataloader, device, dataset_size, report_freq, hyperparams):
    print("traing...")
    network.train()

    optimizer = hyperparams["optimizer"] or torch.optim.Adam(network.parameters(), lr=hyperparams["learning_rate"])
    if hyperparams["lr_scheduler"]:
        lr_scheduler = StepLR(optimizer, hyperparams["lr_scheduler"]["step_size"], hyperparams["lr_scheduler"]["gamma"])
    loss_fn = hyperparams["loss_fn"] or nn.NLLLoss()

    total_loss, accuracy, count = 0, 0, 0
    epoch_count = 0
    while True:
        for i, (features, labels) in enumerate(dataloader, 1):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            out = network(features)
            loss = loss_fn(out, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss
            _, predicted = torch.max(out, 1)
            accuracy += (predicted==labels).sum()
            count += len(labels)

            if i % report_freq == 0:
                print(f"{count}: accuracy={accuracy.item()/count}")

            if count / (epoch_count+1) > dataset_size:
                print(f"{epoch_count+1} time epoch end")
                epoch_count += 1
                if hyperparams["lr_scheduler"]:
                    lr_scheduler.step()
                break

        if epoch_count == hyperparams["epoch"]:
            break

    return total_loss.item()/count, accuracy.item()/count