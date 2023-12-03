from sklearn.tree import DecisionTreeClassifier
from gensim.models import Word2Vec
from sklearn.externals import joblib


word_model=Word2Vec.load("train.model")


class url_iter():
  def __init__(self, path):
    self.path = path
  def __iter__(self):
    for line in open(self.path):
      y_lable,query=line.strip().split(" ")
      yield y_lable,query

url_path="x_train_10e5.txt"
y=[]
x=[]
for y_lable,query in url_iter(url_path):
    y.append(int(y_lable))
    single_url_vec=[]
    for i in query:
        try:
            single_url_vec.append(word_model[i])
        except:
            single_url_vec.append(1)
        if(len(single_url_vec)>=256):
            break
    while(len(single_url_vec)<256):
        single_url_vec.append(0)
    x.append(single_url_vec)

cls = DecisionTreeClassifier(criterion="entropy",max_depth=15)

cls.fit(x,y)


#joblib.dump(cls,'cls.model')
print("Training acc:%f" % (cls.score(x,y)))

url_test_path="x_test_10e5.txt"
y_test=[]
x_test=[]
for y_lable,query in url_iter(url_test_path):
    y_test.append(int(y_lable))
    single_url_vec=[]
    for i in query:
        try:
            single_url_vec.append(word_model[i])
        except:
            single_url_vec.append(1)
        if(len(single_url_vec)>=256):
            break
    while(len(single_url_vec)<256):
        single_url_vec.append(0)
    x_test.append(single_url_vec)

#cls=joblib.load('cls.model')
s_test=cls.score(x_test,y_test)
print("Test acc:{}".format(s_test))