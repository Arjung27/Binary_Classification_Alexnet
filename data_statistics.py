import numpy as np
import cv2
import argparse
import os
import glob
from tqdm import tqdm

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Final Perception Homework')
	parser.add_argument('--data', default='/cmlscratch/arjgpt27/projects/ENPM673/DL/data_original/train')
	args = parser.parse_args()

	data = glob.glob(os.path.join(args.data, '*.jpg'), recursive=False)
	sample_img = cv2.imread(data[0])
	blue = np.zeros((224, 224))
	green = np.zeros((224, 224))
	red = np.zeros((224, 224))
	images = np.zeros((224, 224, 3))
	index = np.random.choice(np.arange(0, len(data)), 1000, replace=False)

	for i in tqdm(range(0, len(index))):
		
		image = cv2.imread(data[index[i]])/255
		image = cv2.resize(image, (224, 224))
		images = np.vstack((images, image))
		# blue = np.dstack((image[:,:,0], blue))
		# green = np.dstack((image[:,:,1], green))
		# red = np.dstack((image[:,:,2], red))
	
	print(images.shape)

	print("Red statistics: {}, {}".format(np.mean(images[224:,:,2]), np.std(images[224:,:,2])))
	print("Green statistics: {}, {}".format(np.mean(images[224:,:,1]), np.std(images[224:,:,1])))
	print("Blue statistics: {}, {}".format(np.mean(images[224:,:,0]), np.std(images[224:,:,0])))
	# print("Red statistics: {}, {}".format(np.mean(red[:,:,1:]), np.std(red[:,:,1:])))
	# print("Blue statistics: {}, {}".format(np.mean(blue[:,:,1:]), np.std(blue[:,:,1:])))
	# print("Green statistics: {}, {}".format(np.mean(green[:,:,1:]), np.std(green[:,:,1:])))