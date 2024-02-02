import statistics
import torch


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


def count_model_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    with open('result.txt', 'a') as f:
        f.write(f'parameters: {num_params}\n')
        f.write('\n')


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

