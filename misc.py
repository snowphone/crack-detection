import re
import os
from scipy.misc import imread

grade = ("시작", "초기", "중기", "말기", "최종")


def label(nameList):
	finder = re.compile(r"시작|초기|중기|말기|최종")
	ret = []
	for name in nameList:
		onehot = [0 for _ in range(len(grade))]
		idx = grade.index(finder.search(name).group())
		onehot[idx] = 1
		ret.append(onehot)
	return ret


def load(folderPath):
	x_imgName = [name for name in os.listdir(
		folderPath) if name.find("jpg") != -1 or name.find("JPG") != -1]
	#정규화
	x_images = [imread(folderPath + name) / 256 for name in x_imgName]
	y_labels = label(x_imgName)
	return x_images, y_labels

def load_file_name(folderPath):
	return [name for name in os.listdir(folderPath) 
			if name.find("jpg") != -1 or name.find("JPG") != -1]

