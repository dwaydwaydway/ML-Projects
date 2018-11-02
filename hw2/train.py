import csv
import numpy as np
import pandas as pd
import io
import math
import sys
from scipy.stats.stats import pearsonr   
def read_file(name):
  temp = open(name, 'r', encoding='big5')
  return csv.reader(temp , delimiter=",")

class prob:
  def __init__(self, train_x, c0, c1):
    self.c0 = c0
    self.c1 = c1
    
    self.c0_count = len(c0[0])
    self.c1_count = len(c1[0])
    
    self.c0_mean = [[np.mean(feature)] for feature in c0]
    self.c1_mean = [[np.mean(feature)] for feature in c1]
  
    self.c0_cov = np.dot((c0 - self.c0_mean), (c0 - self.c0_mean).transpose()) / self.c0_count
    self.c1_cov = np.dot((c1 - self.c1_mean), (c1 - self.c1_mean).transpose()) / self.c1_count
    
  def gaussian(self, x, mean, cov):
    try:
      D = len(mean)
      temp1 = np.dot((x - mean).transpose(), np.linalg.inv(cov))
      temp2 = np.dot(temp1, (x - mean))
      first = math.exp(-0.5 * temp2[0][0])
      second = max(float((2 * math.pi) ** (D/2)) * float(math.sqrt(abs(np.linalg.det(cov)))), 2.2250738585072014e-308)
      return max(first / second, 2.2250738585072014e-308)
    except ZeroDivisionError:
      print(first)
      print(second)
  
  def c0_probability(self, x):
    pc0 = self.c0_count / (self.c0_count + self.c1_count)
    pc1 = self.c1_count / (self.c0_count + self.c1_count)
    cov = pc0 * self.c0_cov + pc1 * self.c1_cov
    px_c0 = self.gaussian(x, self.c0_mean, cov)
    px_c1 = self.gaussian(x, self.c1_mean, cov)
    
    return float(px_c0) * float(pc0) / (float(px_c0) * float(pc0) + float(px_c1) * float(pc1))

def parseX(file):
  array = []
  for i in range(0, 23):
    array.append([])
  for m, row in enumerate(file):
    if m == 0:
      continue
    for j in range(0,23):
        array[j].append(float(row[j]))
  return array

def parseY(file):
  train_y = []
  for i, row in enumerate(file):
    if i == 0:
      continue
    train_y.append([float(row[0])])
  return train_y
  
def find_c0(train_x, train_y):
  c0 = []
  for i, feature in enumerate(train_x.transpose()):
    if train_y[i][0] == 0.0:
      c0.append(feature)
  return np.array(c0).transpose()

def find_c1(train_x, train_y):
  c1 = []
  for i, feature in enumerate(train_x.transpose()):
    if train_y[i][0] == 1.0:
      c1.append(feature)
  return np.array(c1).transpose()

def modify(out):
  for p, feature in enumerate(out):
    if p == 0 or p > 10:
      for i, data in enumerate(feature):
        feature[i] /= 100000
    else:
      for i, data in enumerate(feature):
        feature[i] /= 10
                  
train_x_file = read_file(sys.argv[1])
train_y_file = read_file(sys.argv[2])
test_file = read_file(sys.argv[3])

train_x = parseX(train_x_file)
test_x = parseX(test_file)
train_y = parseY(train_y_file)

w = []
for i in range(0, 23 + 23 + 23):
  w.append([-0.001])
  
modify(train_x)
modify(test_x)

train_x = np.array(train_x)
train_x = np.concatenate((train_x, train_x ** 3, train_x ** 5), axis=0)

test_x = np.array(test_x)
test_x = np.concatenate((test_x, test_x ** 3, test_x ** 5), axis=0)

train_y = np.array(train_y)
w = np.array(w).transpose()

'''
feature: m
data: n

train_y = n x 1
train_x = (m + bias) x n
w = 1 x (m + bias)
'''

c0 = find_c0(train_x, train_y)
c1 = find_c1(train_x, train_y)

judge = prob(train_x, c0, c1)

right = 0
wrong = 0

for i, feature in enumerate(train_x.transpose()):
  data = np.array([[i] for i in feature])
  predict = 1.0
  if judge.c0_probability(data) >= 0.5:
    predict = 0.0
  if predict == train_y[i]:
    right += 1
  else:
    wrong += 1
    
ans = []
for i, feature in enumerate(test_x.transpose()):
  data = np.array([[i] for i in feature])
  temp = []
  temp.append("id_" + str(i))
  if judge.c0_probability( data ) >= 0.5:
    temp.append("0")
  else:
    temp.append("1")
  ans.append(temp)
  
filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in ans:
    s.writerow(i) 
text.close()
