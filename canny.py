import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

# 1d gaussian equation
def gaussian_1d(x, sigma, deriv = 0):
    # normalization constant is needed as the integral over the exponential function  is not unity
    # http://pages.stat.wisc.edu/~mchung/teaching/MIA/reading/diffusion.gaussian.kernel.pdf.pdf
    norm = 1 / (np.sqrt(2 * np.pi) * sigma)
    term = np.exp(-x ** 2 * 0.5 / sigma ** 2)
    if (sigma == 0):
        return 0.0
    elif (deriv == 1):
        return -(x / (sigma ** 2)) * term
    else:
        return norm * term

# 1d gaussian kernel
def kernel_1d(kernel_size, sigma, deriv = 0):
    kernel_1d = np.linspace(-kernel_size, kernel_size, kernel_size)
    if (deriv == 0):
        for i in range(kernel_size):
            kernel_1d[i] = gaussian_1d(kernel_1d[i], sigma)
    else:
        for i in range(kernel_size):
            kernel_1d[i] = gaussian_1d(kernel_1d[i], sigma, deriv)
    return kernel_1d

def conv_1d(image, kernel, axis, prime = 0):
    conv_1d = ndimage.convolve1d(image, kernel, output = float, axis = axis)

    if (prime == 1):
        conv_1d *= 255.0 / conv_1d.max()

    return conv_1d

def magnitude(ix_prime, iy_prime):
    magnitude = np.sqrt(ix_prime ** 2 + iy_prime ** 2)
    magnitude /= magnitude.max() * 255

    return magnitude

def theta(ix_prime, iy_prime):
    return abs(np.arctan2(iy_prime, ix_prime))

# non-max suppression
# classify pixel as edge if local maxima along direction of gradiant magnitude
def non_max_suppression(image, grad_dir):
    rows, cols = image.shape
    nms = np.zeros((rows, cols))

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            before_pixel = []
            after_pixel = []

            # compare east and west pixels
            if (0 <= grad_dir[row, col] < np.pi / 8) or (7 * np.pi / 8 <= grad_dir[row, col] < 9 * np.pi / 8) or (
                    15 * np.pi / 8 <= grad_dir[row, col] < np.pi):
                before_pixel = image[row, col - 1]
                after_pixel = image[row, col + 1]
            # compare north east and south west pixels
            elif (np.pi / 8 <= grad_dir[row, col] < 3 * np.pi / 8) or (9 * np.pi / 8 <= grad_dir[row, col] < 11 * np.pi / 8):
                before_pixel = image[row + 1, col - 1]
                after_pixel = image[row - 1, col + 1]
            # compare north and south pixels
            elif (3 * np.pi / 8 <= grad_dir[row, col] < 5 * np.pi / 8) or (11 * np.pi / 8 <= grad_dir[row, col] < 13 * np.pi / 8):
                before_pixel = image[row - 1, col]
                after_pixel = image[row + 1, col]
            # compare north west and south east pixels
            elif (5 * np.pi / 8 <= grad_dir[row, col] < 7 * np.pi / 8) or (13 * np.pi / 8 <= grad_dir[row, col] < 15 * np.pi / 8):
                before_pixel = image[row - 1, col - 1]
                after_pixel = image[row + 1, col + 1]

            if image[row, col] >= before_pixel and image[row, col] >= after_pixel:
                nms[row, col] = image[row, col]

    return nms

# double hysteresis thresholding
def dbl_hys(image):
    low_ratio = 0.10
    high_ratio = 1.5 * low_ratio
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    rows, cols = image.shape
    hys = np.zeros((rows, cols))

    high_row, high_col = np.where(image >= high_threshold)
    low_row, low_col = np.where((image <= high_threshold) & (image >= low_threshold))
    bg_row, bg_col = np.where(image < low_threshold)

    hys[high_row, high_col] = 255  # edges flagged
    hys[low_row, low_col] = 125  # weak edges flagged
    hys[bg_row, bg_col] = 0  # make background

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # weak edges are edges if any 8-connected pixel is an edge
            if (hys[row, col] == 125):
                if 255 in [[hys[row - 1, col - 1],
                            hys[row - 1, col],
                            hys[row - 1, col + 1],
                            hys[row, col - 1],
                            hys[row, col + 1],
                            hys[row + 1, col - 1],
                            hys[row + 1, col],
                            hys[row + 1, col + 1]]]:
                    hys[row, col] = 255
                else:
                    hys[row, col] = 0

    return hys

def load_image(path, filename):
    image = cv2.imread(path + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(image, cmap="gray")
    #plt.show()

    return image

def save_image(path, filename, image, save_name):
    cmap = plt.cm.gist_gray
    norm = plt.Normalize(vmin = image.min(), vmax = image.max())
    image = cmap(norm(image))
    plt.imsave(path + filename.replace('.jpg', save_name) + '.jpg', image)

def canny(path, filename, save = 0):
    image = load_image(path, filename)

    # kernel values taper around 3 sigma (where gaussian => 0)
    # https://www.crisluengo.net/archives/150/
    sigma = 3  # smaller sigma values detect finer features and the reverse is true
    kernel_size = 3
    kernel_size *= sigma
    x_axis = 1  # convolve along columns
    y_axis = 0  # convolve along rows

    g_kernel = kernel_1d(kernel_size, sigma)
    g_d_kernel = kernel_1d(kernel_size, sigma, deriv = 1)

    # smoothing
    ix = conv_1d(image, g_kernel, x_axis)
    iy = conv_1d(image, g_kernel, y_axis)

    # compute derivatives/gradients
    ix_prime = conv_1d(ix, g_d_kernel, x_axis, prime = 1)
    iy_prime = conv_1d(iy, g_d_kernel, y_axis, prime = 1)

    # magnitude and orientation of gradients
    mag = magnitude(ix_prime, iy_prime)
    grad_dir = theta(ix_prime, iy_prime)

    nms = non_max_suppression(mag, grad_dir)
    hys = dbl_hys(nms)

    if(save == 1):
        save_image(path, filename, image = ix, save_name = '_ix')
        save_image(path, filename, image = iy, save_name = '_iy')
        save_image(path, filename, image = ix_prime, save_name = '_ix_prime')
        save_image(path, filename, image = iy_prime, save_name = '_iy_prime')
        save_image(path, filename, image = nms, save_name = '_nms')
        save_image(path, filename, image = hys, save_name = '_hys')


if __name__ == '__main__':

    path = '/follow/your/own/path/'

    # apply canny edge detection to *.jpg within folder and
    # save intermediary processing results
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            canny(path, filename, save = 1)
        else:
            continue
