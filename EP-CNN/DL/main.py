import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from gensim.models import Word2Vec

class TrainData(Dataset):
    def __init__(self, url_path, word_model, fill_one, fill_zero, wordVecNum, p1_size):
        self.url_path = url_path
        self.word_model = word_model
        self.fill_one = fill_one
        self.fill_zero = fill_zero
        self.wordVecNum = wordVecNum
        self.p1_size = p1_size
        self.data = list(self.load_data())

    def load_data(self):
        url_vec = []
        y = []

        for y_label, url in self.url_iter():
            single_url_vec = self._get_single_url_vector(url)
            y.append([int(y_label), 1 - int(y_label)])  # normal->[1,0]
            url_vec.append(single_url_vec)

        x_train = torch.Tensor(url_vec).view(len(url_vec), 1, self.wordVecNum, 32)
        y_train = torch.Tensor(y)
        return x_train, y_train

    def url_iter(self):
        for line in open(self.url_path):
            y_label, query = line.strip().split(" ")
            yield y_label, query

    def _get_single_url_vector(self, url):
        single_url_vec = []

        for char in url:
            try:
                single_url_vec.append(self.word_model[char])
            except BaseException:
                single_url_vec.append(self.fill_one)

            if len(single_url_vec) >= self.wordVecNum:
                break

        while len(single_url_vec) < self.wordVecNum:
            single_url_vec.append(self.fill_zero)

        return single_url_vec

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Parameters
url_path = "./train_data/"  # Add your url path here
word_model = {}  # Add your word model here
fill_one = [1 for _ in range(32)]
fill_zero = [0 for _ in range(32)]
wordVecNum = 256
p1_size = 128

dataset = TrainData(url_path, word_model, fill_one, fill_zero, wordVecNum, p1_size)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the Word2Vec model
word_model = Word2Vec.load("word_train32.model")

# Define the CNN model
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(512 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create the model
model = TextCNN()
model.train()

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

# Train the model
for epoch in range(30):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'my8_weights.pth')
