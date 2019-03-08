import cv2
import argparse
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import operator
import copy
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from skimage.segmentation import clear_border
from keras.models import load_model

#Show Image
def show_image(img,title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 450,450)
    cv2.imshow(title, img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

img = cv2.imread('img/image4.jpg', cv2.IMREAD_GRAYSCALE)
show_image(img,"Original Image")

#Image filter processing
def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9),0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
      kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
      proc = cv2.dilate(proc, kernel)
    return proc

#Find image corners
def findCorners(img):
    h,contours, hierarchy = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

#Function used to specify point
def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    show_image(img,"display_points")
    return img


def distance_between(p1, p2):
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))

def display_rects(in_img, rects, colour=255):
    img = in_img.copy()
    for rect in rects:
        cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
    show_image(img,"display_rects")
    return img

def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def infer_grid(img):
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)
			p2 = ((i + 1) * side, (j + 1) * side)
			squares.append((p1, p2))
	return squares

def getEveryDigits(img,squares):
    labels = []
    centers = []
    model = load_model('models/mnist_keras_cnn_model.h5')
    img2=img.copy()
    show_image(img2,"TEST")
    height, width = img.shape[:2]
    img2 = Image.fromarray(img2)
    for i in range(81):
        x1=squares[i][0][0]
        x2=squares[i][1][0]
        y1=squares[i][0][1]
        y2=squares[i][1][1]
        window=img[x1:x2, y1:y2]

        digit = cv2.resize(window,(28,28))
        digit = clear_border(digit)

        numPixels = cv2.countNonZero(digit)
        if numPixels<70:
            label=0
        else:
            label2 = model.predict_classes([digit.reshape(1,28,28,1)])
            label=label2[0]
        labels.append(label)
    return matrix_convert(labels)

def matrix_convert(label):
  a=0
  matrix=[]
  for i in range(0,9):
        matrix.append(label[a:a+9])
        a=a+9
  print("original Sudoku")
  for i in range(0,9):
        print(matrix[i])
  print("---------------------------------------")
  return matrix

def checkGrid(grid):
  for row in range(0,9):
      for col in range(0,9):
        if grid[row][col]==0:
          return False
  return True


#Backtracking algorithm
def solveGrid(grid):

  for i in range(0,81):
    row=i/9
    col=i%9
    if grid[row][col]==0:
      for value in range (1,10):
        if not(value in grid[row]):
          if not value in (grid[0][col],grid[1][col],grid[2][col],grid[3][col],grid[4][col],grid[5][col],grid[6][col],grid[7][col],grid[8][col]):
            square=[]
            if row<3:
              if col<3:
                square=[grid[i][0:3] for i in range(0,3)]
              elif col<6:
                square=[grid[i][3:6] for i in range(0,3)]
              else:
                square=[grid[i][6:9] for i in range(0,3)]
            elif row<6:
              if col<3:
                square=[grid[i][0:3] for i in range(3,6)]
              elif col<6:
                square=[grid[i][3:6] for i in range(3,6)]
              else:
                square=[grid[i][6:9] for i in range(3,6)]
            else:
              if col<3:
                square=[grid[i][0:3] for i in range(6,9)]
              elif col<6:
                square=[grid[i][3:6] for i in range(6,9)]
              else:
                square=[grid[i][6:9] for i in range(6,9)]
            if not value in (square[0] + square[1] + square[2]):
              grid[row][col]=value
              if checkGrid(grid):
                print("Sudoku Result")
                for i in range(0,9):
                      print(grid[i])
                print("CHECK CONTROL ")
                return grid
              else:
                  if solveGrid(grid):
                    return grid
      break
  grid[row][col]=0

def writeImg(solved,old,img,squares):
  font                   = cv2.FONT_HERSHEY_SIMPLEX
  bottomLeftCornerOfText = (10,500)
  fontScale              = 3
  fontColor              = (255,255,255)
  lineType               = 2
  img2 = Image.fromarray(img)
  for i in range(81):
    x1=squares[i][0][0]
    x2=squares[i][1][0]
    y1=squares[i][0][1]
    y2=squares[i][1][1]
    window=img[y1:y2,x1:x2]
    if old[i/9][i%9]==0:
        k=i/9
        k=k+1
        tp=(y1,(k*84))
        cv2.putText(img,str(solved[i/9][i%9]),tp,font,fontScale,fontColor,lineType)
        show_image(np.array(img2),"RESULT")
  cv2.waitKey(0)


if __name__== "__main__":
  processed = pre_process_image(img)
  corners = findCorners(processed)
  display_points(processed, corners)

  cropped = crop_and_warp(processed, corners)
  squares = infer_grid(cropped)

  old= getEveryDigits(cropped,squares)
  solved = solveGrid(copy.deepcopy(old))

  writeImg(solved,old,cropped,squares)
