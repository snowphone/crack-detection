import re

def label(nameList):
	finder = re.compile(r"시작|초기|중기|말기|최종")
	return [finder.search(name).group() for name in nameList]