import csv
import numpy as np
import pandas as pd
import os
import sys
import math
from scipy.stats.stats import pearsonr
test, outfile = sys.argv[1], sys.argv[2]
'''
train_data = []
for i in range(0, 12): # 12 months 
  train_data.append([])
  for j in range(0, 18):
    train_data[i].append([])
    
temp = open(infile, 'r', encoding='big5') 
train_file = csv.reader(temp , delimiter=",")

month = -1
for i, row in enumerate(train_file):
  if i == 0:
    continue
  if i % 360 == 1:
    month = month + 1
  for j in range(3,27):
    if row[j] != "NR":
      train_data[month][(i-1)%18].append(float(row[j]))
    else:
      train_data[month][(i-1)%18].append(float(0))
      
x_data = []
y_data = []

for month in train_data:
  month = np.array(list(month)).transpose()
  for i in range(471):
    temp = []
    for j in range(9):
      temp = temp + list(month[i + j])
    x_data.append(temp)
    y_data.append(month[i+9][9])


# ans = real y
ans = []
for i in range(9, len(x_data), 9):
  for j in range(9, 162, 18):
    ans.append(x_data[i][j])
ans += [None, None, None, None, None, None, None, None, None]

# trim off wierd data 
for i, data_set in enumerate(x_data):
  flag = False
  
  for j in range(162):
    if data_set[j] < 0:
      flag = True
      break
      
    if data_set[j] == 0:
      if data_set[j : j + 5] == [0, 0, 0, 0, 0]:
        flag = True
        break
        
  if flag == True:
    del(x_data[i])
    del(y_data[i])
    del(ans[i])
    
# x_transpose = a list of 18 containing all data of a feature    
x_transpose = list(np.array(x_data).transpose())
temp = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i, stuff in enumerate(x_transpose):
  temp[i%18] += list(stuff)

# std = a list of std of features
std = []
mean = []
for i in range(18):
	std.append(np.std(temp[i]))
    mean.append(np.mean(temp[i]))

# trim off extreme data
for i, data_set in enumerate(x_data):
  flag = False
  
  for j in range(162):
    if data_set[j] < (mean[j%18] - 4.5*std[j%18]) or data_set[j] > (mean[j%18] + 4.5*std[j%18]):
      flag = True
      
  if flag == True:
    del(x_data[i])
    del(y_data[i])
    del(ans[i])

x_trans = np.array(x_data).transpose()
valid = []
valid2 = []
for i in range(162):
  if (pearsonr(ans[:len(ans) - ans.count(None)], x_trans[i][:len(ans) - ans.count(None)])[0]) ** 2 > 0.03:
    valid.append(i)
  if (pearsonr(ans[:len(ans) - ans.count(None)], x_trans[i][:len(ans) - ans.count(None)])[0]) ** 2 > 0.23:
    valid2.append(i)


x_square = []
for data_set in x_data:
  temp = []
  for j, data in enumerate(data_set):
    if j in valid:
      temp.append(data)
    if j in valid2:
      temp.append(data ** 2)
  x_square.append(temp)
  
x_square = np.array(x_square)

w = np.full( len(valid) + len(valid2), 0.01)
bias = -1
LRW = 1000
LRB = 1000
w_grad = 0
b_grad = 0

y_data = np.array(y_data)

for i in range(200000):
  loss =  y_data - (np.dot(x_square, w) + bias) 
  cost = np.sum((loss) ** 2) / len(x_square)
  
  b_derivative = -2 * np.sum(loss) / len(x_square)
  w_derivative = (-2 * np.dot(loss.transpose(), x_square)) / len(x_square)
  
  w_grad += (w_derivative) ** 2
  b_grad += (b_derivative) ** 2
  
  w -= LRW *  w_derivative  / np.sqrt(w_grad)
  bias -= LRB * b_derivative  / np.sqrt(b_grad)    
'''

temp = open(test, 'r', encoding='big5') 
test_file = csv.reader(temp , delimiter=",")

model = np.load('model.npz')

def chunks(l, n):
	l = list(l)
	for i in range(0, len(l), n):
		yield np.array(l[i:i + n]).transpose()

per_id = list(chunks(test_file, 18))
test_data = []
for block in per_id:
	store = []
	for i, hour in enumerate(block):
		if i < 2:
			continue
		for num in hour:
			if num == "NR":
				store.append(float(0))
			else:
				store.append(float(num))
	test_data.append(store)

predict = []

for i, id_num in enumerate(test_data):
	id_num_square = []
	for k in range(len(id_num)):
		if k < len(id_num) - 1 and id_num[k] == 0 and id_num[k + 1] == 0:
			m = k
			while m < len(id_num) and id_num[m] == 0:
				if m < 80:
					id_num[m] = id_num[m+18]
				else:
					id_num[m] = id_num[m-18]
				m += 1
	for j, data in enumerate(id_num):
		if j in model['valid']:
			id_num_square.append(data)
		if j in model['valid2']:
			id_num_square.append(data ** 2)
	id_num_sqr = np.array(id_num_square)
	ans = np.dot(id_num_sqr, model['w']) + model['bias']
	temp = []
	temp.append("id_"+str(i))
	temp.append(str(ans))
	predict.append(temp)

text = open(outfile, "w+")
writer = csv.writer(text,delimiter=',',lineterminator='\n')
writer.writerow(["id","value"])
for i in predict:
    writer.writerow(i) 
text.close()
