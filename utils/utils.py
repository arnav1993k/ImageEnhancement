import os
import cv2
import sys
import argparse


def resize(x,n):
    return cv2.resize(x,None,fx=n, fy=n, interpolation = cv2.INTER_CUBIC)
def resize_dir(image_dir,scale):
	if image_dir[-1]!='/':
		image_dir=image_dir+'/'
	resized_path = image_dir+"Resized/"
	if os.path.exists(image_dir):
		images = os.listdir(image_dir)
	else:
		print("Input directory does not exists")
		sys.exit()

	if not os.path.exists(resized_path):
	    os.mkdir(resized_path)

	for img in images:
		try:
			name = img.split("/")[-1]
			print("*** Resizing image -> {}".format(name))
			image = cv2.imread(image_dir+img,flags=cv2.IMREAD_COLOR)
			big_img = resize(image,scale)
			cv2.imwrite(resized_path+name,big_img)
		except Exception as e:
			print(e)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir', default=".", type=str)
	parser.add_argument('--scale', default='2', type=int)
	parser.add_argument('--method', default='resize_dir', type=str)
	args = parser.parse_args()
	method = args.method
	if method == "resize_dir":
		image_dir = args.image_dir
		scale = args.scale
		resize_dir(image_dir,scale)
