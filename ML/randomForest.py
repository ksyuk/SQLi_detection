from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec

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
    url_vec = []
    for char in query:
        url_vec.append(word_model[char] if char in word_model else 1)
        if len(url_vec) >= 256:
            break
    url_vec.extend([0] * (256 - len(url_vec)))
    return url_vec

# Load training data
url_path = "x_train_10e5.txt"
y_train = []
x_train = []
for y_label, query in URLIterator(url_path):
    y_train.append(int(y_label))
    x_train.append(get_url_vector(query))

# Train classifier
cls = RandomForestClassifier(n_estimators=101, criterion='entropy', max_features=16, max_depth=20, n_jobs=-1)
cls.fit(x_train, y_train)

# Load test data
url_test_path = "x_test_10e5.txt"
y_test = []
x_test = []
for y_label, query in URLIterator(url_test_path):
    y_test.append(int(y_label))
    x_test.append(get_url_vector(query))

# Evaluate classifier
score = cls.score(x_test, y_test)
print(score)
