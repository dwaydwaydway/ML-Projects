# Machine Learning Projects
Here are all the ML related projects that I have done. Most of them are projects in lecture: NTU Machine Learning(Fall, 2018).
## PM2.5 Preduction
To predict the PM2.5 concentration of the next hour using data of the nine previous hours. 

Ranked top 15% on private leaderboard ([Kaggle Link](https://www.kaggle.com/t/39e5638799ce440d89c19297afef9cf2))
### Technique
* Linear Regression
### Environment
* Python 3.6
* Numpy: 1.15
* Scipy: 1.1.0
* Pandas: 0.23.4
### Usage
```sh
$ bash hw1.sh [input file] [output file]
```
## Credit Card Default Payment Prediction
To predict the credit card default payment using various information. 

Ranked top 22% on private leaderboard ([Kaggle Link](https://www.kaggle.com/t/019e3be1832d48eaaa0fbe24430adb4b))
### Technique
* Logistic Regression
* Generative Model
### Environment
* Python 3.6
* Numpy: 1.15
* Scipy: 1.1.0
* Pandas: 0.23.4
### Usage
* Logistic Regression
```sh
$ bash hw2_best.sh [train_x file] [train_y file] [test_x file] [output file]
```
* Generative Model
```sh
$ bash hw2.sh [train_x file] [train_y file] [test_x file] [output file]
```

## Image Sentiment Classification
To classify images of different sentiments 

Ranked top 39% on private leaderboard ([Kaggle Link](https://www.kaggle.com/t/d7a2678990d546b3a4a54f8191321b42))
### Technique
* CNN
### Environment
* Python 3.6
* Numpy: 1.15
* Scipy: 1.1.0
* Pandas: 0.23.4
* Keras: 2.1.6
### Usage
* Training 
```sh
$ bash  hw3_train.sh <training data>
```
* Testing
```sh
$ bash  hw3_test.sh  <testing data>  <prediction file>
```
## Malicious Comments Identification
To classify images of different sentiments 

Ranked top 1% on private leaderboard ([Kaggle Link](https://www.kaggle.com/t/6fb2e644f95f450c9419dd74701ec391))
### Technique
* RNN
* Word Embedding
* Bag of Words
### Environment
* Python 3.6
* Numpy: 1.15
* Scipy: 1.1.0
* Pandas: 0.23.4
* Keras: 2.1.6
* Jieba 0.39
* Gensim 3.6.0
* Emoji 0.5.1
### Usage
* Training 
```sh
$ bash hw4_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
```
* Testing
```sh
$ bash hw4_test.sh <test_x file> <dict.txt.big file> <output file>
```
