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

normalize = set()

def parseX(file):
  B = {1.0 : 14, 2.0 : 15}
  C = {0.0 : 16, 1.0 : 17, 2.0 : 18, 3.0 : 19, 4.0 : 20, 5.0 : 21, 6.0 : 22}
  D = {0.0 : 23, 1.0 : 24, 2.0 : 25, 3.0 : 26}
  E = {-2.0 : 27, -1.0 : 28, 0.0 : 29, 1.0 : 30, 2.0 : 31, 3.0 : 32, 4.0 : 33, 5.0 : 34, 6.0 : 35, 7.0 : 36, 8.0 : 37}
    
  array = []
  for i in range(0, 94+14):
    array.append([])
  
  for m, row in enumerate(file):
    remain = set(range(0, 94))
    if m == 0:
      continue
    for j in range(0,23):
      if j == 0:
        array[j].append(float(row[j]) / 1000000.0)
        array[94].append((float(row[j]) / 1000000.0) ** 2)
        remain.remove(int(j))
        normalize.add(j)
        normalize.add(94)
      elif j == 4:
        array[j-3].append(float(row[j]) / 10.0)
        array[95].append((float(row[j]) / 10.0) ** 2)
        remain.remove(int(j-3))
        normalize.add(j)
        normalize.add(95)
      elif j in range(11, 23):
        array[j-9].append(float(row[j]) / 1000000.0)
        array[96 + j-11].append((float(row[j]) / 1000000.0) ** 2)
        remain.remove(int(j-9))
        normalize.add(j)
        normalize.add(96 + j-11)

      elif j == 1:
        array[B[float(row[j])]].append(1.0)
        remain.remove(int(B[float(row[j])]))
      elif j == 2:
        array[C[float(row[j])]].append(1.0)
        remain.remove(int(C[float(row[j])]))
      elif j == 3:
        array[D[float(row[j])]].append(1.0)
        remain.remove(int(D[float(row[j])]))
      elif j in range(5, 11):
        i = float(j - 5)
        array[int(E[float(row[j])] + 11 * i)].append(1.0)
        remain.remove(int(E[float(row[j])] + 11 * i))
    for i in remain:
       array[i].append(0.0)
        
  return array

def modify(out):
  
    for p, feature in enumerate(out):
      if p in normalize:
        mean = np.mean(feature)
        std = np.std(feature)
        for i, data in enumerate(feature):
          feature[i] = (feature[i] - mean) / std
          
def sigmoid(x, derivative=False):
  if derivative:
    return x*(1.0-x)
  else: 
    ret = np.minimum(0.99999999999, 1 / (1 + np.exp(-x)))
    ret = np.maximum(0.00000000001, ret)
    return ret

train_x_file = read_file(sys.argv[1])
train_y_file = read_file(sys.argv[2])
test_file = read_file(sys.argv[3])

train_x = parseX(train_x_file)
test_x = parseX(test_file)

train_y = []
w = []
for i in range(0, 94+14):
  w.append([-0.1])
  
one_index = []

for i, row in enumerate(train_y_file):
  if i == 0:
    continue
  train_y.append([float(row[0])])
  if float(row[0]) == 1.0:
    one_index.append(i)
    
  
# modify(train_x)
# modify(test_x)

  
model = np.load('model.npz')

LR = float(20)
grad__square_sum = np.zeros((1, 94+14), dtype=float)
grad__square_sum += 0.00000000001

train_x = np.array(train_x)

test_x = np.array(test_x)

train_y = np.array(train_y)
w = np.array(w).transpose()

w = model['w']
'''
feature: m
data: n

train_y = n x 1
train_x = (m + bias) x n
w = 1 x (m + bias)
'''
sigmoid_list = []

# for i in range(200000):
#   f = sigmoid( np.dot( w, train_x ) )
#   loss = -( np.dot( np.log(f), train_y ) + np.dot( np.log(1 - f), (1 - train_y) ) )
 
#   gradiant = -(np.dot((train_y.transpose() - f), train_x.transpose()))
  
#   grad__square_sum += gradiant ** 2
#   w -= LR *  (gradiant) / np.sqrt(grad__square_sum)
#   if i == 100000 - 1:
#     sigmoid_list = list(f.transpose())
#   if i % 10000 == 0:
#     print ('iteration: %d | Loss: %f  ' % ( i, loss))
#     print(f)

ans = []
for i, feature in enumerate( np.array(test_x).transpose() ):
  temp = []
  temp.append("id_" + str(i))
  if sigmoid( np.dot( w, feature ) ) >= 0.5:
    temp.append("1")
  else:
    temp.append("0")
  ans.append(temp)
  
filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in ans:
    s.writerow(i) 
text.close()
