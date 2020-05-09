import numpy as np
import cv2
import argparse
import os
import glob
from tqdm import tqdm
np.random.seed(40)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Final Perception Homework')
	parser.add_argument('--data', default='/cmlscratch/arjgpt27/projects/ENPM673/DL/data_original/train')
	parser.add_argument('--output', default='/cmlscratch/arjgpt27/projects/ENPM673/DL/dataset')
	args = parser.parse_args()

	cat_data = glob.glob(os.path.join(args.data, 'cat*'), recursive=True)
	dog_data = glob.glob(os.path.join(args.data, 'dog*'), recursive=True)

	indexes_cat = np.arange(0, len(cat_data))
	indexes_dog = np.arange(0, len(dog_data))
	train_indexes_cat = np.random.choice(indexes_cat, int(0.75*len(cat_data)), replace=False)
	train_indexes_dog = np.random.choice(indexes_dog, int(0.75*len(dog_data)), replace=False)

	# Cat
	if not os.path.exists(args.output + '/train/0'):
		os.makedirs(args.output + '/train/0')

	if not os.path.exists(args.output + '/val/0'):
		os.makedirs(args.output + '/val/0')

	# Dog
	if not os.path.exists(args.output + '/train/1'):
		os.makedirs(args.output + '/train/1')

	if not os.path.exists(args.output + '/val/1'):
		os.makedirs(args.output + '/val/1')

	for i in range(0, len(cat_data)):

		image = cv2.imread(cat_data[i])
		filename = cat_data[i].split('/')[-1]
		if i in train_indexes_cat:
			cv2.imwrite(os.path.join(args.output + '/train/0', filename), image)
		else:
			cv2.imwrite(os.path.join(args.output + '/val/0', filename), image)

	for i in range(0, len(dog_data)):

		image = cv2.imread(dog_data[i])
		filename = dog_data[i].split('/')[-1]
		if i in train_indexes_dog:
			cv2.imwrite(os.path.join(args.output + '/train/1', filename), image)
		else:
			cv2.imwrite(os.path.join(args.output + '/val/1', filename), image)


