from collections import Counter
from itertools import product

import numpy as np
from sklearn.metrics import accuracy_score


def get_unique_proba(labels):
  total = len(labels)
  c = {k: v/total for k,v in Counter(labels).items()}
  return c


def gini_impurity(labels):
  unique = np.array(list(get_unique_proba(labels).values()))
  impurity = 1 - np.sum(np.power(unique, 2))
  return impurity


def partition(data, q):
  true_p = q.check(data).nonzero()[0]
  false_p = (~q.check(data)).nonzero()[0]
  return true_p, false_p


def weighted_gini_impurity(true, false):
  p = len(true) / (len(false) + len(true))
  i_t = gini_impurity(true)
  i_f = gini_impurity(false)
  return p * i_t + (1-p) * i_f


class Question():
  def __init__(self, column, value):
    self.column = column
    self.value = value

  def check(self, x):
    return x[:, self.column] < self.value

  def __repr__(self):
    return "Q: x[%s] < %s" % (self.column, self.value)


def generate_questions(data, num_features):
  columns = np.random.choice(range(data.shape[-1]), num_features, replace=False)
  for c in columns:
    unique_values = np.unique(data[:,c])
    if len(unique_values) < 2:
      continue
    for uv in unique_values:
      yield Question(c, uv)


class DecisionNode():
  def __init__(self, true, false, q):
    self.true = true
    self.false = false
    self.q = q


  def make(self, data):
    true_p, false_p = partition(data, self.q)
    
    true_r = self.true.make(data[true_p])
    flase_r = self.false.make(data[false_p])

    results = np.zeros(data.shape[0])
    results[true_p] = true_r
    results[false_p] = flase_r
    return results

  def __get_str__(self, level=1):
    b_str = str(self.q)
    t_str = '  ' * level + 'T: ' + self.true.__get_str__(level+1)
    f_str = '  ' * level + 'F: ' + self.false.__get_str__(level+1)
    return b_str + '\n' + t_str + '\n' + f_str

  def __repr__(self):
    return self.__get_str__()


class LeafNode():
  def __init__(self, probas):
    self.result = max(probas.items(), key=lambda x: x[1])[0]

  def make(self, data):
    return np.array([self.result] * data.shape[0])

  def __get_str__(self, level):
    return str(self)

  def __repr__(self):
    return str(self.result)


class Tree():
  def __init__(self):
    self.tree = None
    self.num_features = None

  def preprocess_and_assert(func):
    def call(self, data, labels = None, *kargs, **kwargs):
      #preprocess
      data = np.array(data)
      if labels is not None:
        labels = np.array(labels)

      #assert
      self.assert_data_format(data, labels)

      #call
      if not labels is not None:
        return func(self, data)
      return func(self, data, labels, *kargs, **kwargs)
    return call

  def assert_data_format(self, data, labels=None):
    assert len(data.shape) == 2
    if self.num_features:
      assert data.shape[1] == self.num_features
    if labels is not None:
      assert len(labels.shape) == 1
      assert labels.shape[0] == data.shape[0]

  @preprocess_and_assert
  def predict(self, data):
    return self.tree.make(data)

  def __max_num_features__(self, mf):
    if isinstance(mf, int):
      return mf
    if isinstance(mf, float):
      return int(np.round(self.num_features * mf))
    if mf == 'sqrt':
      return int(np.round(np.sqrt(self.num_features)))
    if mf == 'log2':
      return int(np.round(np.log2(self.num_features)))
    return self.num_features


  @preprocess_and_assert
  def fit(self, data, labels, max_depth=10, min_samples=1, \
      max_num_features='sqrt'):
    self.num_features = data.shape[1]
    self.max_depth = max_depth
    self.min_samples = min_samples
    self.max_num_features = self.__max_num_features__(max_num_features)
    self.tree = self.__build_node__(data, labels)
    # print(self.tree)

  @preprocess_and_assert
  def evaluate(self, data, labels):
    predictions = self.predict(data)
    return accuracy_score(labels, predictions)

  @staticmethod
  def __evaluate_question__(data, labels, initial_impurity, q):
    # Partitonate indexes by question 
    true_idx, false_idx = partition(data, q)

    # Compute weighted sum of gini impurity for both subsets
    weighted_partition_impurity = \
        weighted_gini_impurity(labels[true_idx], labels[false_idx])

    # Information gain after answering question
    information_gain = initial_impurity - weighted_partition_impurity

    return (information_gain, q)

  def __build_node__(self, data, labels, depth=0):
    # Compute initial gini impurity
    impurity = gini_impurity(labels)

    information_gain, question = None, None
    # If labels subset has impurity
    if impurity > 0 and depth < self.max_depth and data.shape[0] > self.min_samples:
      # Evaluate every posible question checking information gain on data subset
      questions_evaluation = \
          [Tree.__evaluate_question__(data, labels, impurity, q) 
              for q in generate_questions(data, self.max_num_features)]

      # There may be no question to evaluate (Unseparable data -> Leaf Node)
      if questions_evaluation:
        # Save question which gives max information gain
        information_gain, question = \
            max(questions_evaluation, key=lambda x:x[0])

      # There may be no information gain on every question (Unseparable data -> Leaf Node)
      if information_gain:
        # Decision Node build on top of new evaluation on new data subsets
        true_idx, false_idx = partition(data, question)
        true = self.__build_node__(data[true_idx], labels[true_idx], depth+1)
        false = self.__build_node__(data[false_idx], labels[false_idx], depth+1)
        return DecisionNode(true, false, question)

    # if data.shape[0] <= self.min_samples and impurity != 0:
      # print("Failed on min_samples", data.shape[0], "depth", depth+1)
    # Leaf Node :labels subset has no imprity, no questions to ask or 0 information gain)
    return LeafNode(get_unique_proba(labels))


class RandomForest():
  def __sample_size__(self, ss, total):
    if isinstance(ss, int):
      return ss
    if isinstance(ss, float):
      return np.max([10, int(np.round(ss * total))])
    return np.max([10, int(np.round(total / self.n_estimators * 2))])

  def fit(self, data, labels, n_estimators=10, max_depth=5, min_samples=1, max_num_features='sqrt', \
      sample_size='auto'):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.max_num_features = max_num_features
    self.min_samples = min_samples
    self.sample_size = self.__sample_size__(sample_size, data.shape[0])
    self.forest = self.__build_forest__(data, labels)

  def predict(self, data):
    predictions = np.zeros((data.shape[0], self.n_estimators))
    for i in range(self.n_estimators):
      predictions[:, i] = self.forest[i].predict(data)
    predictions = predictions.astype(int)
    return np.apply_along_axis(np.bincount, 1, predictions).argmax(axis=1)

  def evaluate(self, data, labels):
    predictions = self.predict(data)
    return accuracy_score(labels, predictions)

  def __bagging__(self, data, labels):
    for i in range(self.n_estimators):
      bag_idx = np.random.choice(range(data.shape[0]), self.sample_size, replace=True)
      yield data[bag_idx], labels[bag_idx]

  def __build_forest__(self, data, labels):
    forest = []
    for b_data, b_labels in self.__bagging__(data, labels):
      t = Tree()
      t.fit(b_data, b_labels, max_depth=self.max_depth, min_samples=self.min_samples, \
          max_num_features=self.max_num_features)
      forest.append(t)
    return forest



# data = np.array([[1,1,1,2,2,2,2],[1,3,2,3,2,3,1]]).T
# labels = np.array([0,1,1,0,0,1,1])

# t = Tree()
# t.fit(data, labels)
# test = list(product([1,2], [1,2,3]))
# print(test)
# t.predict(test)
# t.evaluate(test, [0, 1, 1, 1, 0, 0])

import pandas as pd

data = pd.read_csv('data.csv').values
labels = np.array([_ == 'M' for _ in data[:, -1]]).astype(int)
data = data[:, :-1]

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.2)
t = RandomForest()
for ss in [None, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
  t.fit(x_train, y_train, n_estimators=100, max_depth=7, min_samples=5, sample_size=ss)
  print(ss, t.evaluate(x_train, y_train), t.evaluate(x_test, y_test))

t = Tree()
for md in [1, 3, 5, 7, 10]:
  for ms in [1, 15, 30, 50, 70, 85, 110]:
    t.fit(x_train, y_train, max_depth=md, min_samples=ms, \
        max_num_features='sqrt')

    print(md, ms, t.evaluate(x_train, y_train), t.evaluate(x_test, y_test))
