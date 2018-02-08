from itertools import chain

import numpy as np
from PIL import Image

BLACK = 0
WHITE = 255


def convert_to_bw(name: str, threshold=90):
	'''
	특정 값을 기준으로 어두운 픽셀을 검정색으로, 밝은 픽셀은 흰색으로 변환한
	binary image를 반환한다.
	'''
	img: Image.Image = Image.open(name)
	grey = img.convert('L')
	bw = grey.point(lambda x: WHITE if x > threshold else BLACK)
	return bw


def count_if(iterable, func):
	''' func을 만족하는 원소의 수를 반환'''
	return sum(1 for x in iterable if func(x))


def count_pixels(img: Image.Image):
	'''binary image를 입력으로 받아 (검정색 픽셀, 전체 픽셀)을 반환한다.'''
	matrix = np.asarray(img)
	ary = list(chain(*matrix))
	return count_if(ary, lambda x: x == BLACK), len(ary)


if __name__ == "__main__":
	name = "images/crack.jpg"
	ret = convert_to_bw(name)
	a = count_pixels(ret)
