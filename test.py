import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys
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

def process_image(image):
    result = np.copy(image)

    combined_region = np.copy(image)
    ysize = image.shape[0]
    xsize = image.shape[1]

    # region vertices
    left_bottom = [130, ysize]
    right_bottom = [940, ysize]
    apex = [480, 300]

    # color selection
    # select while line region
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    white_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                        (image[:,:,1] < rgb_threshold[1]) | \
                        (image[:,:,2] < rgb_threshold[2])

    # white_color_select[white_thresholds] = [0,0,0]

    vertices = [np.array([left_bottom, right_bottom, apex], dtype=type(apex[0]))]
    masked_image = region_of_interest(image, vertices)

    # plt.imshow(masked_image, cmap = 'gray')
    # plt.show()

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

    combined_region[yellow_thresholds & white_thresholds] = [0,0,0]
    # plt.imshow(combined_region, cmap = 'gray')
    # plt.show()

    # change to grayscale
    grayImg = cv2.cvtColor(combined_region, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    kernel_size = 3
    gaussianImg = cv2.GaussianBlur(grayImg, (kernel_size, kernel_size), 0)

    low_threshold = 70
    high_threshold = 100

    # canny
    cannyImg = cv2.Canny(gaussianImg, low_threshold, high_threshold)
    # plt.imshow(cannyImg, cmap = 'gray')
    # plt.show()
    # region select
    vertices = [np.array([left_bottom, right_bottom, apex], dtype=type(apex[0]))]
    masked_image = region_of_interest(cannyImg, vertices)

    # plt.imshow(masked_image, cmap = 'gray')
    # plt.show()
    # hough lines
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 11 #minimum number of pixels making up a line
    max_line_gap = 2 # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    # classify left and right lanes
    left_lines = []
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if((y2 - y1) / (x2 - x1) > 0):
                left_lines.append(line)
            else:
                right_lines.append(line)
    draw_lines(result, right_lines)
    return result


image = mpimg.imread('test_images/fucked.jpg')
outImg = process_image(image)
plt.imshow(outImg, cmap = 'gray')
plt.show()
sys.exit("No Error")

white_output = 'challenge_out.mp4'
clip = VideoFileClip("challenge.mp4")
white_clip = clip.fl_image(process_image)
    # i += 1
    # if(i > 20):
    #     break
white_clip.write_videofile(white_output, audio=False)
# HTML("""
# <video width="960" height="540" controls>
#   <source src="{0}">
# </video>
# """.format(white_output))
