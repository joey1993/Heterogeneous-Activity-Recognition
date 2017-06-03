import sys
import json
from pprint import pprint
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis
# from statsmodels.tsa.ar_model import AR

def compute_feat(instance):
	
	adata_list = instance['adata']
	Xa = []
	Ya = []
	Za = []
	features_a = []
	features_g = []
	for data in adata_list:
		Xa.append(float(data['data'][0]))
		Ya.append(float(data['data'][1]))
		Za.append(float(data['data'][2]))

	if len(Xa) !=0 and len(Ya) !=0 and len(Za) !=0:
		iqrXa = np.subtract(*np.percentile(Xa, [75, 25]))
		iqrYa = np.subtract(*np.percentile(Ya, [75, 25]))
		iqrZa = np.subtract(*np.percentile(Za, [75, 25]))


		features_a =  [
		np.mean(Xa), np.std(Xa), np.var(Xa), np.median(Xa), np.max(Xa), np.min(Xa), skew(Xa), kurtosis(Xa), iqrXa,
		np.mean(Ya), np.std(Ya), np.var(Ya), np.median(Ya), np.max(Ya), np.min(Ya), skew(Ya), kurtosis(Ya), iqrYa,
		np.mean(Za), np.std(Za), np.var(Za), np.median(Za), np.max(Za), np.min(Za), skew(Za), kurtosis(Za), iqrZa,
		pearsonr(Xa, Ya)[0], pearsonr(Xa, Za)[0], pearsonr(Ya, Za)[0]]

	# print np.mean(Xa), np.std(Xa), np.median(Xa), np.max(Xa), np.min(Xa), pearsonr(Xa, Ya), entropy(Xa)

	# print len(features), features

	gdata_list = instance['gdata']
	Xg = []
	Yg = []
	Zg = []
	for data in gdata_list:
		Xg.append(float(data['data'][0]))
		Yg.append(float(data['data'][1]))
		Zg.append(float(data['data'][2]))

	# print Xg
	if len(Xg) !=0 and len(Yg) !=0 and len(Zg) !=0:
		# print "aaa", Xg
		iqrXg = np.subtract(*np.percentile(Xg, [75, 25]))
		iqrYg = np.subtract(*np.percentile(Yg, [75, 25]))
		iqrZg = np.subtract(*np.percentile(Zg, [75, 25]))
		
		features_g = [
		np.mean(Xg), np.std(Xg), np.var(Xg), np.median(Xg), np.max(Xg), np.min(Xg), skew(Xg), kurtosis(Xg), iqrXg,
		np.mean(Yg), np.std(Yg), np.var(Yg), np.median(Yg), np.max(Yg), np.min(Yg), skew(Yg), kurtosis(Yg), iqrYg,
		np.mean(Zg), np.std(Zg), np.var(Zg), np.median(Zg), np.max(Zg), np.min(Zg), skew(Zg), kurtosis(Zg), iqrZg,
		pearsonr(Xg, Yg)[0], pearsonr(Xg, Zg)[0], pearsonr(Yg, Zg)[0]]
	
	# print len(features_a), len(features_g)
	return features_a, features_g

if __name__ == '__main__':
	np.seterr("ignore")
	filename = sys.argv[1]
	try:
		with open('../'+ filename +'.json') as data_file:    
				fout = open('../feature/'+ filename + '_feature.json', 'w+')
				for line in data_file:
					instance = json.loads(line)
					features_a, features_g = compute_feat(instance)
					out_json = {
		                'label': instance['label'],
						'begin_time': instance['begin_time'],
		                'end_time': instance['end_time'],
		                'user': instance['user'],
		                'features_a': features_a,
		                'features_g': features_g
					}
					json.dump(out_json, fout)
					fout.write("\n")
					# print(json.dumps(out_json), file=fout)
					# break  	
	except:
		print "No such file, please try again"


