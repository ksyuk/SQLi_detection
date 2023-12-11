from sklearn import svm
from gensim.models import Word2Vec
from joblib import dump

# Load word model
word_model = Word2Vec.load("train.model")

# Define URL iterator
class URLIterator:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path) as file:
            for line in file:
                y_label, query = line.strip().split(" ")
                yield y_label, query

# Function to get vector representation of URL
def get_url_vector(query):
    url_vec = [word_model[char] if char in word_model else 1 for char in query[:256]]
    url_vec.extend([0] * (256 - len(url_vec)))
    return url_vec

# Load training data
url_path = "x_train_10000.txt"
y_train = []
x_train = []
for y_label, query in URLIterator(url_path):
    y_train.append(int(y_label))
    x_train.append(get_url_vector(query))

# Load test data
url_test_path = "test.txt"
y_test = []
x_test = []
for y_label, query in URLIterator(url_test_path):
    y_test.append(int(y_label))
    x_test.append(get_url_vector(query))

# Train and save the model
clf = svm.SVC(C=0.8, kernel='rbf', gamma=0.3, decision_function_shape='ovr', probability=True)
clf.fit(x_train, y_train)
dump(clf, 'clf.model')

# Print training accuracy
print(clf.score(x_train, y_train))
