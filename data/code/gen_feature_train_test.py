# python gen_feature_train_test.py data_Watch_1_small 0.2

import sys
import json
from pprint import pprint
import numpy as np

if __name__ == '__main__':
	np.seterr("ignore")
	filename = sys.argv[1]
	test_percentage = sys.argv[2]
	count = 0
	#try:
	with open('../'+ filename +'.json') as data_file: 
			data = np.array(data_file.readlines())   
			fout_train = open('../'+ filename + '_train.json', 'w+')
			fout_test = open('../'+ filename + '_test.json', 'w+')
			length = len(data)
			np.random.seed(10)
			shuffle_indices = np.random.permutation(np.arange(length))
			data_shuffled = data[shuffle_indices]
			print test_percentage
			test_index = -1 * int(float(test_percentage) * float(length))
			data_train, data_test = data_shuffled[:test_index], data_shuffled[test_index:]

			for line in data_train: fout_train.write(line)
			for line in data_test: fout_test.write(line)

			fout_train.close()
			fout_test.close()
