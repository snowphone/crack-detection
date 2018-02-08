from PIL import Image
from itertools import chain
import numpy as np


def convert_to_bw(name: str, threshold=90):
	img: Image.Image = Image.open(name)
	grey = img.convert('L')
	bw = grey.point(lambda x: 255 if x > threshold else 0)
	return bw

def count_if(iterable, func):
    return sum(1 for x in iterable if func(x))

def count_pixels(img: Image.Image):
    matrix = np.asarray(img)
    ary = list(chain(*matrix))
    return count_if(ary, lambda x: x == 0), len(ary)

if __name__ == "__main__":
	ret = convert_to_bw("lena_std.tif")