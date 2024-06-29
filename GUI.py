import cv2
from tkinter import*
from tkinter import filedialog
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import preprocess
import thinning
from PIL import Image, ImageDraw
import utils
import argparse
import math
import os
import crossing
from PIL import Image
from PIL import ImageTk
from PIL import Image as Img
root=Tk()
root.title("FingerPrint Authentication")
root.geometry('1370x700+0+0')
root.resizable(False,False)
root.config(bg="#fcb6b2")
Label(root, text="FINGERPRINT RECOGNITION WITH HOUGH TRANSFORM COMPUTER VISION TECHNIQUES ",font=("Times new roman",20),fg="black",bg='#fcb6b2').place(x=50,y=50)

np.random.seed(0)

cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def minutiae_at(pixels, i, j):
    values = [pixels[i + k][j + l] for k, l in cells]

    crossings = 0
    for k in range(0, 8):
        crossings += abs(values[k] - values[k + 1])
    crossings /= 2

    if pixels[i][j] == 1:
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"

def calculate_minutiaes(im):
    pixels = utils.load_image(im)
    utils.apply_to_each_pixel(pixels, lambda x: 0.0 if x > 10 else 1.0)

    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)

    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

    ellipse_size = 2
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            minutiae = minutiae_at(pixels, i, j)
            if minutiae != "none":
                draw.ellipse([(i - ellipse_size, j - ellipse_size), (i + ellipse_size, j + ellipse_size)], outline = colors[minutiae])

    del draw

    return result

def pixel_is_black(arr, x, y):
    if arr[x, y] == 1:
        return True
    return False

def pixel_has_2_to_6_black_neighbors(arr, x, y):
    if (2 <= arr[x, y-1] + arr[x+1, y-1] + arr[x+1, y] + arr[x+1, y+1] +
        arr[x, y+1] + arr[x-1, y+1] + arr[x-1, y] + arr[x-1, y-1] <= 6):
        return True
    return False


def pixel_has_1_white_to_black_neighbor_transition(arr, x, y):
   
    neighbors = [arr[x, y-1], arr[x+1, y-1], arr[x+1, y], arr[x+1, y+1],
                 arr[x, y+1], arr[x, y+1], arr[x-1, y], arr[x-1, y-1],
                 arr[x, y-1]]
    transitions = sum((a, b) == (0, 1) for a, b in zip(neighbors, neighbors[1:]))
    if transitions == 1:
        return True
    return False
    
def at_least_one_of_P2_P4_P6_is_white(arr, x, y):
    if (arr[x, y-1] and arr[x+1, y] and arr[x, y+1]) == False:
        return True
    return False

def at_least_one_of_P4_P6_P8_is_white(arr, x, y):
    if (arr[x+1, y] and arr[x, y+1] and arr[x-1, y]) == False:
        return True
    return False


def at_least_one_of_P2_P4_P8_is_white(arr, x, y):
    if (arr[x, y-1] and arr[x+1, y] and arr[x-1, y]) == False:
        return True
    return False


def at_least_one_of_P2_P6_P8_is_white(arr, x, y):
    if (arr[x, y-1] and arr[x, y+1] and arr[x-1, y]) == False:
        return True
    return False


def selectfile():
	from PIL import Image as Img
	from PIL import ImageTk
	global filename, panelA,cap
	filename=filedialog.askopenfilename(filetype=(("all files","*.*"),("all files","*.*")))
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Img.fromarray(image)
	image = ImageTk.PhotoImage(image)
	panelB = Label(image=image)
	panelB.image = image
	Label(root, text="Input Image",font=("Times new roman",16),fg="black",bg="#fcb6b2").place(x=510,y=120)
	panelB.place(x=450,y=150)

def preprocess_fingerprint():
	
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	print("image is loaded.")
	preprocess.Process(image)
	img = cv2.imread('data/mask.png')
	image = cv2.imread('data/mask.png')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Img.fromarray(image)
	image = ImageTk.PhotoImage(image)
	panelB = Label(image=image)
	panelB.image = image
	Label(root, text="PreProcessed Image",font=("Times new roman",16),fg="black",bg="#fcb6b2").place(x=790,y=120)
	panelB.place(x=750,y=150)
	
def Thinning():
	img = cv2.imread('data/mask.png', 0)
	retval, orig_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
	bin_thresh = (orig_thresh == 0).astype(int)
	thinned_thresh = bin_thresh.copy()
	while 1:
		thresh_copy = thinned_thresh.copy()
		pixels_meeting_criteria = []
		for i in range(1, thinned_thresh.shape[0] - 1):
			for j in range(1, thinned_thresh.shape[1] - 1):
				if (pixel_is_black(thinned_thresh, i, j) and
					pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
					pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
					at_least_one_of_P2_P4_P6_is_white(thinned_thresh, i, j) and
					at_least_one_of_P4_P6_P8_is_white(thinned_thresh, i, j)):
					pixels_meeting_criteria.append((i, j))
		for pixel in pixels_meeting_criteria:
			thinned_thresh[pixel] = 0
		pixels_meeting_criteria = []
		for i in range(1, thinned_thresh.shape[0] - 1):
			for j in range(1, thinned_thresh.shape[1] - 1):
				if (pixel_is_black(thinned_thresh, i, j) and
					pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
					pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
					at_least_one_of_P2_P4_P8_is_white(thinned_thresh, i, j) and
					at_least_one_of_P2_P6_P8_is_white(thinned_thresh, i, j)):
					pixels_meeting_criteria.append((i, j))
		for pixel in pixels_meeting_criteria:
			thinned_thresh[pixel] = 0
		if np.all(thresh_copy == thinned_thresh) == True:
			break
	global im
	thresh = (thinned_thresh == 0).astype(np.uint8)
	thresh *= 255
	cv2.imwrite('data/thinn55.png',thresh)
	im = Image.open('data/thinn55.png')
	image = cv2.imread('data/thinn55.png')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Img.fromarray(image)
	image = ImageTk.PhotoImage(image)
	panelB = Label(image=image)
	panelB.image = image
	Label(root, text="Thinning Image",font=("Times new roman",16),fg="black",bg="#fcb6b2").place(x=510,y=420)
	panelB.place(x=450,y=450)
	
def Minutiae():
	global im
	im = im.convert("L") 
	crossing.calculate_minutiaes(im)
	result = calculate_minutiaes(im)
	result.save('data/minutia.png',)
	image = cv2.imread('data/minutia.png')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Img.fromarray(image)
	image = ImageTk.PhotoImage(image)
	panelB = Label(image=image)
	panelB.image = image
	Label(root, text="Minutiae Image",font=("Times new roman",16),fg="black",bg="#fcb6b2").place(x=790,y=420)
	panelB.place(x=750,y=450)

def Houghman():
	import numpy as np
	import imageio
	import math

	def rgb2gray(rgb):
		return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


	def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
		thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
		width, height = img.shape
		diag_len = int(round(math.sqrt(width * width + height * height)))
		rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

		cos_t = np.cos(thetas)
		sin_t = np.sin(thetas)
		num_thetas = len(thetas)

		accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
		are_edges = img > value_threshold if lines_are_white else img < value_threshold
		y_idxs, x_idxs = np.nonzero(are_edges)

		for i in range(len(x_idxs)):
			x = x_idxs[i]
			y = y_idxs[i]

			for t_idx in range(num_thetas):
				rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
				accumulator[rho, t_idx] += 1

		return accumulator, thetas, rhos
		
	def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(1, 2, figsize=(10, 10))

		ax[0].imshow(img, cmap=plt.cm.gray)
		ax[0].set_title('Input image')
		ax[0].axis('image')

		ax[1].imshow(
			accumulator, cmap='jet',
			extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
		ax[1].set_aspect('equal', adjustable='box')
		ax[1].set_title('Hough transform')
		ax[1].set_xlabel('Angles (degrees)')
		ax[1].set_ylabel('Distance (pixels)')
		ax[1].axis('image')

		if save_path is not None:
			plt.savefig(save_path, bbox_inches='tight')
		plt.show()

	
	imgpath = filename
	img = imageio.imread(imgpath)
	if img.ndim == 3:
		img = rgb2gray(img)
	accumulator, thetas, rhos = hough_line(img)
	show_hough_line(img, accumulator, thetas, rhos,  save_path='output.png')

def match():
	import cv2
	import os
	import sys
	import numpy
	import matplotlib.pyplot as plt
	from enhance import image_enhance
	from tkinter import filedialog
	from pathlib import Path
	from skimage.morphology import skeletonize, thin
	os.chdir("C:\\Users\\Admin\\Desktop\\gokulfinalproject\\final project gokul\\Fingerprint")
	def removedot(invertThin):
		temp0 = numpy.array(invertThin[:])
		temp0 = numpy.array(temp0)
		temp1 = temp0/255
		temp2 = numpy.array(temp1)
		temp3 = numpy.array(temp2)

		enhanced_img = numpy.array(temp0)
		filter0 = numpy.zeros((10,10))
		W,H = temp0.shape[:2]
		filtersize = 6

		for i in range(W - filtersize):
			for j in range(H - filtersize):
				filter0 = temp1[i:i + filtersize,j:j + filtersize]

				flag = 0
				if sum(filter0[:,0]) == 0:
					flag +=1
				if sum(filter0[:,filtersize - 1]) == 0:
					flag +=1
				if sum(filter0[0,:]) == 0:
					flag +=1
				if sum(filter0[filtersize - 1,:]) == 0:
					flag +=1
				if flag > 3:
					temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

		return temp2


	def get_descriptors(img):
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		img = clahe.apply(img)
		img = image_enhance.image_enhance(img)
		img = numpy.array(img, dtype=numpy.uint8)
		ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
		img[img == 255] = 1

		
		skeleton = skeletonize(img)
		skeleton = numpy.array(skeleton, dtype=numpy.uint8)
		skeleton = removedot(skeleton)
		
		harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
		harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
		threshold_harris = 125
		
		keypoints = []
		for x in range(0, harris_normalized.shape[0]):
			for y in range(0, harris_normalized.shape[1]):
				if harris_normalized[x][y] > threshold_harris:
					keypoints.append(cv2.KeyPoint(y, x, 1))
		
		orb = cv2.ORB_create()
		
		_, des = orb.compute(img, keypoints)
		print("feature matrix",keypoints,des)
		return (keypoints, des)

	def file():
		import ntpath
		import os
		from PIL import Image
		from tkinter import filedialog

		def get_all_images(directory):
			image_paths = []

			for filename in os.listdir(directory):

				image_paths.append("{}/{}".format(directory, filename))
				
			return image_paths

		def get_filename(path):
			return ntpath.basename(path)

		def compare_images(input_image, output_image):

			if input_image.size != output_image.size:
				return False

			rows, cols = input_image.size

			for row in range(rows):
				for col in range(cols):
					input_pixel = input_image.getpixel((row, col))
					output_pixel = output_image.getpixel((row, col))
					if input_pixel != output_pixel:
						return False

			return True
			
		def find_duplicate_image(input_image, output_images):

			input_image = Image.open(input_image)

			for image in output_images:
				if compare_images(input_image, Image.open(image)):
					return image
		global filename, panelA,cap
		global duplicate_path
		possible_duplicates = get_all_images("database")
		duplicate_path = find_duplicate_image(filename, possible_duplicates)
		
		if duplicate_path:
			print("Matched")
			print(duplicate_path)
		
		else:
			print("not matched")
			
	def findmatch():
		global filename
		image_name = filename
		img1 = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
		kp1, des1 = get_descriptors(img1)	
		global duplicate_path
		file()
		if duplicate_path==None:
			root=Tk()
			root.title("FingerPrint Authentication")
			root.geometry('350x300+360+200')
			root.resizable(False,False)
			root.config(bg="#ec8080")
			Label(root, text="Fingerprint does not match",font=("Times new roman",20),fg="black",bg="#ec8080").place(x=30,y=120)
			root.mainloop()
		else:
			image_name1 = duplicate_path
			img2 = cv2.imread(image_name1, cv2.IMREAD_GRAYSCALE)
			kp2, des2 = get_descriptors(img2)
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches = sorted(bf.match(des1, des2), key= lambda match:match.distance)
			img4 = cv2.drawKeypoints(img1, kp1, outImage=None)
			img5 = cv2.drawKeypoints(img2, kp2, outImage=None)
			f, axarr = plt.subplots(1,2)
			axarr[0].imshow(img4)
			axarr[1].imshow(img5)
			plt.show()
			img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
			plt.imshow(img3)
			plt.show()
			root=Tk()
			root.title("FingerPrint Authentication")
			root.geometry('350x300+360+200')
			root.resizable(False,False)
			root.config(bg="#ec8080")
			Label(root, text="Fingerprint matched",font=("Times new roman",20),fg="black",bg="#ec8080").place(x=30,y=120)
			root.mainloop()
			score = 0
			for match in matches:
				score += match.distance
			score_threshold = 33
			if score/len(matches) < score_threshold:
				print("Fingerprint matches.")
			else:
				print("Fingerprint does not match.")
	findmatch()
	
btn=Button(root,text="Select Input File",width="20",font=("Times new roman",14),fg="black",bg='#f99691',command=selectfile)
btn.place(x=100,y=250)
btn1=Button(root,text="Preprocess",width="20",font=("Times new roman",14),fg="black",bg='#f99691',command=preprocess_fingerprint)
btn1.place(x=100,y=300)
btn2=Button(root,text="Thinning",width="20",font=("Times new roman",14),fg="black",bg='#f99691',command=Thinning)
btn2.place(x=100,y=350)
btn3=Button(root,text="Minutiae",width="20",font=("Times new roman",14),fg="black",bg='#f99691',command=Minutiae)
btn3.place(x=100,y=400)
btn4=Button(root,text="Houghman Transform",width="20",font=("Times new roman",14),fg="black",bg='#f99691',command=Houghman)
btn4.place(x=100,y=450)
btn5=Button(root,text="Matching",width="20",font=("Times new roman",14),fg="black",bg='#f99691',command=match)
btn5.place(x=100,y=500)
root.mainloop()