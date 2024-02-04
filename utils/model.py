import time

import torch
from torch.optim.lr_scheduler import StepLR
from transformers import BertForSequenceClassification

from .plot import plot_loss
from .result import calculate_classification_metric, write_classification_metric


def evaluate_model(network, dataloader):
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

                val_accuracy = evaluate_model(network, val_dataloader)
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
