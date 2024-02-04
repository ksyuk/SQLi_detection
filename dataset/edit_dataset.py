import csv
import random


def load_dataset(dataset_path):
    dataset = []
    with open(dataset_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) == 2:
                dataset.append(row)
    return dataset


def split_dataset(filename):
    dataset = load_dataset(filename)
    random.shuffle(dataset)

    test_dataset = dataset[:6500]
    val_dataset = dataset[6500:13000]
    train_dataset = dataset[13000:]

    with open("train.csv", 'a') as file:
        file.write("query,label\n")
        for query, label in train_dataset:
            query = query.replace('"', '""')
            file.write(f'"{query}",{label}\n')

    with open("validation.csv", 'a') as file:
        file.write("query,label\n")
        for query, label in val_dataset:
            query = query.replace('"', '""')
            file.write(f'"{query}",{label}\n')

    with open("test.csv", 'a') as file:
        file.write("query,label\n")
        for query, label in test_dataset:
            query = query.replace('"', '""')
            file.write(f'"{query}",{label}\n')


def count_label(filename):
    dataset = load_dataset(filename)

    zero_count = 0
    one_count = 0
    for _, label in dataset:
        if label == "1":
            one_count += 1
        else:
            zero_count += 1

    print(f"zero: {zero_count}, one: {one_count}")
    print(f"sum: {zero_count + one_count}")


# ラベルが1のものと0のものを同数にする
def make_up_by_label(filename):
    dataset = load_dataset(filename)
    random.shuffle(dataset)

    zero_count = 0
    with open("tmp.csv", 'a') as file:
        file.write("query,label\n")
        for query, label in dataset:
            # Add an extra double quote before each existing double quote
            query = query.replace('"', '""')
            if label == "1":
                file.write(f'"{query}",{label}\n')
            else:
                if zero_count > 23925:
                    continue
                file.write(f'"{query}",{label}\n')
                zero_count += 1


# sqliv3.csvの末尾にあるカンマを削除する
def remove_trailing_commas(filename):
    """Removes two trailing commas from each line in a file."""

    with open(filename, 'r') as file:
        lines = file.readlines()

    with open(filename, 'w') as file:
        for line in lines:
            if line.endswith(',,\n'):
                line = line[:-3] + '\n'
            file.write(line)


if __name__ == '__main__':
    # remove_trailing_commas('./sqliv3.csv')

    # make_up_by_label("./dataset.csv")

    # split_dataset("./tmp.csv")

    # count_label("./tmp.csv")
    count_label("./train.csv")
    count_label("./validation.csv")
    count_label("./test.csv")
