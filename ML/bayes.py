from sklearn.metrics import f1_score, make_scorer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from gensim.models import Word2Vec
from sklearn.externals import joblib

word_model = Word2Vec.load("train.model")


class url_iter():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            y_lable, query = line.strip().split(" ")
            yield y_lable, query


url_path = "x_train_10000.txt"
y = []
x = []
for y_lable, query in url_iter(url_path):
    y.append(int(y_lable))
    single_url_vec = []
    for i in query:
        try:
            single_url_vec.append(word_model[i])
        except BaseException:
            single_url_vec.append(1)
        if (len(single_url_vec) >= 256):
            break
    while (len(single_url_vec) < 256):
        single_url_vec.append(0)
    x.append(single_url_vec)

clf = BernoulliNB()
clf.fit(x, y)
joblib.dump(clf, 'Bernoulli.model')
clf2 = MultinomialNB()
clf3 = GaussianNB()
clf2.fit(x, y)
clf3.fit(x, y)
joblib.dump(clf2, 'Multinomial.model')
joblib.dump(clf3, 'Gaussian.model')

# print("Training acc:%f" % (clf.score(x,y)))

url_test_path = "x_test_10e5.txt"
y_test = []
x_test = []
for y_lable, query in url_iter(url_test_path):
    y_test.append(int(y_lable))
    single_url_vec = []
    for i in query:
        try:
            single_url_vec.append(word_model[i])
        except BaseException:
            single_url_vec.append(1)
        if (len(single_url_vec) >= 256):
            break
    while (len(single_url_vec) < 256):
        single_url_vec.append(0)
    x_test.append(single_url_vec)
"""
s_test=clf.score(x_test,y_test)
print("Test acc:{}".format(s_test))

 """

f1 = make_scorer(f1_score)

ber = f1(clf, x_test, y_test)
mul = f1(clf2, x_test, y_test)
ga = f1(clf3, x_test, y_test)
print(ber)
print(mul)
print(ga)
