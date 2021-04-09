import pandas as pd
import numpy as np
from pprint import pprint

dataset = pd.read_csv('data.csv', names = ['STT', 'Height', 'HairLength', 'Voice', 'Label'])

def entropy(target_col):
	elements, counts = np.unique(target_col, return_counts = True)
	en = 0
	for i in range(len(elements)):
		pi = counts[i]/np.sum(counts)
		temp = -pi*np.log2(pi)
		en = en + temp
	return en

def infoGain(data, split_attribute_name, target_name):
	total_entropy = entropy(data[target_name])
	
	vals, counts = np.unique(data[split_attribute_name], return_counts = True)
	print(vals)
	print(counts)
	
	total_elements = np.sum(counts)
	print(total_elements)
	
	weighted_entropy = 0
	for i in range(len(vals)):
		print(vals[i])
		print(split_attribute_name)
		
		weighted_elements = (counts[i]/total_elements)
		print(weighted_elements)
		
		dt_split_attribute_vals = data[data[split_attribute_name] == vals[i]]
		print(dt_split_attribute_vals)
		
		entropy_elements = entropy(dt_split_attribute_vals[target_name])
		
		weighted_entropy = weighted_entropy + weighted_elements*entropy_elements
		print(weighted_entropy)
	
	information_gain = total_entropy - weighted_entropy
	print(information_gain)
	return information_gain
	
def ID3(data, original_data, features, target_attribute_name, parent_node_class):
	if len(np.unique(data[target_attribute_name])) <= 1:
		print('Du lieu thuan nhat tra ve nut la')
		return np.unique(data[target_attribute_name])[0]
	elif len(features) == 0:
		return 0
	else:
		parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],
							return_counts = True)[1])]
		print('parent node class: ', parent_node_class)
		print('tap features: ', features)
		
		item_values = [infoGain(data, features, target_attribute_name) for feature in features]
		print('item_values: ', item_values)
		best_feature_index = np.argmax(item_values)
		best_feature = features[best_feature_index]
		print('best_feature', best_feature)
		
		tree = {best_feature:{}}
		features = [i for i in features if i != best_feature]
		
		print('Tap thuoc tinh sau khi loai la: ', features)
		
		for value in np.unique(data[best_feature]):
			print('Xet gia tri: ', value, '\n')
			sub_data = data.where(data[best_feature] == value).dropna()
			
			print('Subdata la: \n', sub_data)
			print('parent_node_class la: ', parent_node_class, '\n')
			subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)
			
			tree[best_feature][value] = subtree
		return(tree)

training_data = dataset.iloc[1:15].reset_index(drop=True)
tree = ID3(training_data, training_data, training_data.columns[1:-1], 'Label', None)
pprint(tree)
