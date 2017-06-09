# python gen_instances_small.py data_Watch_1 200

import json
import numpy as np
import sys

if __name__ == '__main__':
	np.seterr("ignore")
	filename = sys.argv[1]
	keep_num = int(sys.argv[2])
	with open('../'+ filename +'.json') as data_file: 
		fout = open('../'+ filename +'_small.json','w')
		contents = data_file.readlines()
		for item in contents:
			line = json.loads(item)
			if len(line['gdata']) < keep_num or len(line['adata']) < keep_num: continue
			line['gdata'] = line['gdata'][:keep_num]
			line['adata'] = line['adata'][:keep_num]
			json.dump(line, fout)
			fout.write("\n")
	fout.close()