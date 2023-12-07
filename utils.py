import csv

import torch
import torch.nn as nn

def load_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            dataset.append(row)
    return dataset


def train_epoch(network, dataloader, learning_rate, optimizer, loss_fn, epoch, device, dataset_size, report_freq=200):
    print("traing...")
    network.train()

    learning_rate = learning_rate
    optimizer = optimizer or torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = loss_fn or nn.NLLLoss()

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
                print("one epoch end")
                epoch_count += 1
                break

        if epoch_count == epoch:
            break

    return total_loss.item()/count, accuracy.item()/count