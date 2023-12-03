import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec

class Sentences():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            a = [i for i in line.strip()]
            yield a

class TrainData(Dataset):
    def __init__(self, url_path, word_model, fill_one, fill_zero, batch_size):
        self.url_path = url_path
        self.word_model = word_model
        self.fill_one = fill_one
        self.fill_zero = fill_zero
        self.batch_size = batch_size
    
    def ulr_iter(self):
        for line in open(self.url_path):
            y_label, query = line.strip().split(" ")
            yield y_label, query

    def __getitem__(self, index):
        url_vec = []
        y = []

        for y_label, url in self.url_iter(self.url_path):
            single_url_vec = self._get_single_url_vector(url)
            y.append([int(y_label), 1 - int(y_label)])  # normal->[1,0]
            url_vec.append(single_url_vec)

            if len(url_vec) >= self.batch_size:
                x_train = torch.Tensor(url_vec).view(len(url_vec), 1, 256, 32)
                y_train = torch.Tensor(y)
                return x_train, y_train

    def _get_single_url_vector(self, url):
        single_url_vec = []

        for char in url:
            try:
                single_url_vec.append(word_model[char])
            except BaseException:
                single_url_vec.append(fill_one)

            if len(single_url_vec) >= 256:
                break

        while len(single_url_vec) < 256:
            single_url_vec.append(fill_zero)

        return single_url_vec

    def __len__(self):
        return len(open(self.url_path).readlines()) // self.batch_size


path = "../train_test_data/word2vec_train"
sentences = Sentences(path)
word_model = Word2Vec(sentences, size=32, window=10, min_count=5).wv
fill_one = [1 for _ in range(32)]
fill_zero = [0 for _ in range(32)]
url_path = "../train_test_data/train_data_2e5"
batch_size = 32

train_data = TrainData(url_path, word_model, fill_one, fill_zero, batch_size)
train_loader = DataLoader(train_data, batch_size=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 8 * 1, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.pool(F.relu(self.conv8(x)))
        x = self.dropout1(x)
        x = x.view(-1, 512 * 8 * 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

# Assuming that load_TrainData is a PyTorch DataLoader
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'cnn256-32_model_1214.pt')

