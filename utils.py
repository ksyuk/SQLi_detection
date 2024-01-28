import csv
import statistics
import time

import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from transformers import BertForSequenceClassification


def load_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) == 2:
                dataset.append(row)
    return dataset


def count_model_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    with open('result.txt', 'a') as f:
        f.write(f'parameters: {num_params}\n')
        f.write('\n')


def plot_loss(report_count, loss_values):
    plt.plot(report_count, loss_values, color='dimgray')
    plt.xlabel('Count')
    plt.ylabel('Loss')
    plt.show()


def plot_inference_time_histogram(inference_time, bins='auto'):
    plt.hist(inference_time, bins=bins, color='dimgray')
    plt.xlabel('Inference Time')
    plt.ylabel('Count')
    plt.show()


def evaluate(network, dataloader):
    network.eval()

    with torch.no_grad():
        total, correct = 0, 0
        for features, labels in dataloader:
            out = network(features)
            if type(out) == int:
                continue
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum()

    network.train()
    return correct.item()/total


def train_model(network, train_dataloader, val_dataloader, dataset_size, report_freq, hyperparams):
    print('training...')

    optimizer = hyperparams['optimizer']
    loss_fn = hyperparams['loss_fn']
    if hyperparams['lr_scheduler']:
        lr_scheduler = StepLR(optimizer, hyperparams['lr_scheduler']['step_size'], hyperparams['lr_scheduler']['gamma'])

    total_loss, accuracy, best_val_accuracy = 0, 0, 0
    count, epoch_count = 0, 0
    loss_values = []
    report_count = []

    start_time = time.time()

    while True:
        for i, (features, labels) in enumerate(train_dataloader, 1):
            optimizer.zero_grad()
            out = network(features)
            if type(out) == int:
                continue
            loss = loss_fn(out, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out, 1)
            accuracy += (predicted==labels).sum()
            count += len(labels)

            if i % report_freq == 0:
                print(f'{count}: accuracy={accuracy.item()/count}')
                loss_values.append(loss.item())
                report_count.append(count)

                val_accuracy = evaluate(network, val_dataloader)
                if val_accuracy > best_val_accuracy and hyperparams['epoch'] - epoch_count == 1:
                    best_val_accuracy = val_accuracy
                    torch.save(network.state_dict(), 'model.pth')

            if count > dataset_size * (epoch_count+1):
                print(f'{epoch_count+1} time epoch end')
                epoch_count += 1
                if hyperparams['lr_scheduler']:
                    lr_scheduler.step()
                break

        if epoch_count == hyperparams['epoch']:
            break

    end_time = time.time()
    with open('result.txt', 'a') as f:
        f.write(f'training_time: {end_time - start_time}\n')
        f.write('\n')

    plot_loss(report_count, loss_values)


def calculate_classification_metric(classification_counts, labels, preds, raw_queries):
    for label, pred, raw_query in zip(labels, preds, raw_queries):
        if isinstance(label, torch.Tensor):
            label = label.item()
            pred = pred.item()
        if label == 1:
            if pred == 1:
                classification_counts['true_positives'] += 1
            else:
                with open('false_negatives.txt', 'a') as f:
                    f.write(f'{raw_query}\n')
                classification_counts['false_negatives'] += 1
        else:
            if pred == 0:
                classification_counts['true_negatives'] += 1
            else:
                classification_counts['false_positives'] += 1
                with open('false_positives.txt', 'a') as f:
                    f.write(f'{raw_query}\n')


def write_classification_metric(classification_counts, inference_time):
    true_positives = classification_counts['true_positives']
    false_positives = classification_counts['false_positives']
    true_negatives = classification_counts['true_negatives']
    false_negatives = classification_counts['false_negatives']
    with open('result.txt', 'a') as f:
        f.write(f'inference time median: {statistics.median(inference_time)}\n')
        f.write(f'inference time max: {max(inference_time)}\n')
        f.write('\n')
        f.write(f'true_positives: {true_positives}\n')
        f.write(f'false_positives: {false_positives}\n')
        f.write(f'true_negatives: {true_negatives}\n')
        f.write(f'false_negatives: {false_negatives}\n')
        f.write('\n')
        f.write(f'accuracy: {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}\n')
        f.write(f'precision: {true_positives / (true_positives + false_positives)}\n')
        f.write(f'recall: {true_positives / (true_positives + false_negatives)}\n')
        f.write(f'f1: {2 * true_positives / (2 * true_positives + false_positives + false_negatives)}\n')


def test_model(model, test_loader):
    classification_counts = {
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0
    }
    inference_time = []

    with torch.no_grad():
        if model.__class__ == BertForSequenceClassification:
            for batch_data in test_loader:
                raw_queries = batch_data.pop('raw_queries', None)
                start_time = time.time()
                out = model(**batch_data)
                inference_time.append(time.time() - start_time)

                _, preds = torch.max(out.logits, 1)

                calculate_classification_metric(classification_counts, batch_data['labels'], preds, raw_queries)
        else:
            for queries, labels, raw_queries in test_loader:
                start_time = time.time()
                try:
                    out = model(queries)
                except RuntimeError:
                    print(raw_queries)
                inference_time.append(time.time() - start_time)

                _, preds = torch.max(out, 1)

                calculate_classification_metric(classification_counts, labels, preds, raw_queries)

    write_classification_metric(classification_counts, inference_time)
    plot_inference_time_histogram(inference_time)
