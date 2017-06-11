from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import sys, os
try:
    import ujson as json
except:
    import json

def load_file(filename):
	X_list = []
	y_list = []
	with open(filename,'r') as fin:
		for line in fin:
			data = json.loads(line)
			x = data['features_a'] + (data['features_g'])
			X_list.append(x)
			y_list.append(data['label'])
	return X_list, y_list



def label_to_numerical(y_train_label,y_test_label):
	le = preprocessing.LabelEncoder()
	le.fit(y_train_label + y_test_label)
	y_train = le.transform(y_train_label)
	y_test = le.transform(y_test_label)
	return y_train,y_test


def Gaussian_naiveBayes(X_train,y_train,X_test):
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	y_test_predict = gnb.predict(X_test)
	return y_test_predict 

def BernoulliNB_naiveBayes(X_train,y_train,X_test):
	bnb = BernoulliNB()
	bnb.fit(X_train, y_train)
	y_test_predict = bnb.predict(X_test)
	return y_test_predict 




def calculatePerformance(y_test_true,y_test_predict):
	precision = metrics.precision_score(y_test_true, y_test_predict, average='weighted')
	recall = metrics.recall_score(y_test_true, y_test_predict, average='weighted')
	f1_score = metrics.f1_score(y_test_true, y_test_predict, average='weighted') 
	return precision,recall,f1_score


def weightAvg(val1,size1,val2,size2):
	return (val1*size1+val2*size2)/(size1+size2);



def main():
	X_train_watch,y_train_label_watch = load_file('../../data/feature/data_Watch_1_small_train_feature.json')
	X_test_watch,y_test_label_watch = load_file('../../data/feature/data_Watch_1_small_test_feature.json')
	y_train_watch,y_test_watch = label_to_numerical(y_train_label_watch,y_test_label_watch)
	y_test_predict_watch = BernoulliNB_naiveBayes(X_train_watch,y_train_watch,X_test_watch)
	precision_watch,recall_watch,f1_score_watch = calculatePerformance(y_test_watch,y_test_predict_watch)
	print precision_watch,recall_watch,f1_score_watch
	watch_size =  np.shape(y_test_watch)[0]


	X_train_phones,y_train_label_phones = load_file('../../data/feature/data_Phones_1_small_train_feature.json')
	X_test_phones,y_test_label_phones = load_file('../../data/feature/data_Phones_1_small_test_feature.json')
	y_train_phones,y_test_phones = label_to_numerical(y_train_label_phones,y_test_label_phones)
	y_test_predict_phones = BernoulliNB_naiveBayes(X_train_phones,y_train_phones,X_test_phones) 
	precision_phones,recall_phones,f1_score_phones = calculatePerformance(y_test_phones,y_test_predict_phones)
	print precision_phones,recall_phones,f1_score_phones
	phones_size = np.shape(y_test_phones)[0]


	print 'precision: '+ str(weightAvg(precision_watch,watch_size,precision_phones,phones_size))
	print 'recall: '+str(weightAvg(recall_watch,watch_size,recall_phones,phones_size))
	print 'f1_score: '+str(weightAvg(f1_score_watch,watch_size,f1_score_phones,phones_size))



if __name__ == "__main__":
	main()