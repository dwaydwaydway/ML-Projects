# Machine Learning Projects
Here are all the ML related projects that I have done.
## Malicious Comments Identification
To detect malicious comments on Dcard

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
$ bash train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
```
* Testing
```sh
$ bash test.sh <test_x file> <dict.txt.big file> <output file>
```
