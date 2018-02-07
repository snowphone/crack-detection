from PIL import Image


def convert_to_binary(name: str, threshold=90):
	img: Image.Image = Image.open(name)
	grey = img.convert('L')
	bw = grey.point(lambda x: 255 if x > threshold else 0)
	return bw

def showImage(img: Image.Image):
	img.show()

if __name__ == "__main__":
	ret = convert_to_binary("lena_std.tif")
	showImage(ret)