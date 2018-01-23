import os
import sys

# Load in the Pascal VOC Dataset
def load_pascal_voc_classification(full_path, text_files_dir, image_files_dir):

	# The image classification list returned in the FORMAT specified above 
	image_classification_list = []

	num_classes = 0

	# For every text file in the folder
	for filename in os.listdir(text_files_dir):

		# Each file represents another class
		# There are a total of 20 in the Pascal VOC dataset
		num_classes += 1 

		# Small array for each file
		arr = []

		# Read all of the data from the text file into a list 
		# Each element of the list represents a line in the text file
		f = open(text_files_dir + "/" + filename, 'r')
		file_lines = f.readlines()

		# Removes the newline '\n'
		file_lines = [x.strip() for x in file_lines] 

		# Only grab the positive images for each class and remove 
		# the positive image indicator '1' and preceding spaces
		for idx, line in enumerate(file_lines):
			if "-1" in line:
				continue
			else:
				line = line[:len(line)-3]
				arr.append(line)

		# Put all of it into the list of tuples 
		# First element is the image name, second element is the one hot vector representation
		for element in arr:
			one_hot = [0] * 20
			one_hot[num_classes - 1] = 1
			image_classification_list.append([element, one_hot])

	num_classes = 0

	return image_classification_list