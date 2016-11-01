import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# image = mpimg.imread('test_images/solidWhiteRight.jpg')
image = mpimg.imread('test_images/solidYellowLeft.jpg')

white_color_select = np.copy(image)
yellow_color_select = np.copy(image)
ysize = image.shape[0]
xsize = image.shape[1]

# region vertices
left_bottom = [130, ysize]
right_bottom = [870, ysize]
apex = [480, 290]

print('This image is:', type(image), 'with dimesions:', image.shape)

# color selection
# select while line region
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]
white_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

white_color_select[white_thresholds] = [0,0,0]

# select yellow line
red_threshold = 180
green_threshold = 180
blue_threshold_down = 0
blue_threshold_up = 200
yellow_rgb_threshold = [red_threshold, green_threshold, blue_threshold]
yellow_thresholds = (image[:,:,0] < red_threshold) | \
                    (image[:,:,1] < green_threshold) | \
                    (image[:,:,2] < blue_threshold_down) | \
                    (image[:,:,2] > blue_threshold_up)

yellow_color_select[yellow_thresholds] = [0,0,0]
combined_region = cv2.bitwise_or(yellow_color_select, white_color_select, [], [])
plt.imshow(combined_region, cmap = 'gray')
plt.show()

# change to grayscale
grayImg = cv2.cvtColor(white_color_select, cv2.COLOR_BGR2GRAY)

# gaussian blur
kernel_size = 3
gaussianImg = cv2.GaussianBlur(grayImg, (kernel_size, kernel_size), 0)

low_threshold = 70
high_threshold = 100

# canny
cannyImg = cv2.Canny(gaussianImg, low_threshold, high_threshold)

# region select
vertices = [np.array([left_bottom, right_bottom, apex], dtype=type(apex[0]))]
masked_image = region_of_interest(cannyImg, vertices)

# hough lines
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 6 #minimum number of pixels making up a line
max_line_gap = 5   # maximum gap in pixels between connectable line segments

lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
draw_lines(image, lines)

plt.imshow(image)
plt.show()
