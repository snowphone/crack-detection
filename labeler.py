import re

def label(nameList):
	finder = re.compile(r"시작|초기|중기|말기|최종")
	return [finder.search(name).group() for name in nameList]

def label_onehot(nameList):
	classifier = ["시작", "초기", "중기", "말기", "최종"]
	finder = re.compile(r"시작|초기|중기|말기|최종")
	ret = []
	for name in nameList:
		onehot = [0 for _ in range(len(classifier))]
		idx = classifier.index(finder.search(name).group())
		onehot[idx] = 1
		ret.append(onehot)
	return ret
		
