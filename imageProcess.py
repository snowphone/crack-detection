from itertools import chain

import numpy as np
from PIL import Image

BLACK = 0
WHITE = 255


def convert_to_bw(name: str, threshold=90):
	img: Image.Image = Image.open(name)
	grey = img.convert('L')
	bw = grey.point(lambda x: WHITE if x > threshold else BLACK)
	return bw


def count_if(iterable, func):
	return sum(1 for x in iterable if func(x))


def count_pixels(img: Image.Image):
	matrix = np.asarray(img)
	ary = list(chain(*matrix))
	return count_if(ary, lambda x: x == BLACK), len(ary)


if __name__ == "__main__":
	name = "images/crack.jpg"
	ret = convert_to_bw(name)
	a = count_pixels(ret)
