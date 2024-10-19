import numpy as np
import skimage as sk
import skimage.io as io
import matplotlib.pyplot as plt
import cv2 
import scipy.signal as signal
from skimage.feature import corner_harris, peak_local_max
images_dir = './images/'
# *** Part 1. Defining Correspondences

# IMPORTED METHOD CODE FROM PROJECT 2
# proj2: returns a gaussian blurred img 
def gaussian(img, sigma):
    k1d = cv2.getGaussianKernel(sigma * 6, sigma)
    kernel = k1d * k1d.T
    return signal.convolve(img, np.expand_dims(kernel, axis=2), mode="same")

# proj2: creates gaussian stack starting with sigma = 2, 4, 8, 16, 32
def gauss_stack(hybrid, N):
  gstack = [hybrid]
  for i in range(1, N):
    gauss = gaussian(hybrid, 2**i)
    gstack.append(gauss) 
  return gstack

# proj2: creates lap stack of N levels
def lap_stack(gstack, N):
  lstack = []
  for i in range(0, N-1):
    lstack.append(gstack[i]-gstack[i+1]) 
    if (i == N-2):
      lstack.append(gstack[i+1])
  return lstack

# proj 4A ~~

# define n pairs of corresponding points on the two images by hand using gpinput
def get_points(im1, im2, n):
    plt.imshow(im1)
    im1_pts = np.asarray(plt.ginput(n, 0))
    plt.imshow(im2)
    im2_pts = np.asarray(plt.ginput(n, 0))
    return im1_pts, im2_pts

# recovers homography via set of (p', p) pairs of corresponding points from two images
# im_pts are n-by-2 matrices holding (x, y) locations of n point correspondences
# H is (recovered 3x3 homography matrix)
def computeH(im1_pts, im2_pts):
    # A * h = b
    n = len(im1_pts)
    A = np.zeros((2*n, int(8)))
    b = np.zeros((2*n, int(1)))
    
    for i in range(n):
        pt1 = im1_pts[i]
        pt2 = im2_pts[i]
        # calculates B  
        b[i*2] = pt2[0]
        b[i*2+1] = pt2[1]

        # calc A
        A[i*2, 2] = 1
        A[i*2+1, 5] = 1

        A[i*2, 0] = pt1[0]
        A[i*2, 1] = pt1[1]
        A[i*2+1, 3] = pt1[0]
        A[i*2+1, 4] = pt1[1]

        A[i*2, 6] = -pt2[0] * pt1[0]
        A[i*2, 7] = -pt2[0] * pt1[1]
        A[i*2+1, 6] = -pt2[1] * pt1[0]
        A[i*2+1, 7] = -pt2[1] * pt1[1]

    H = np.linalg.lstsq(A,b, rcond=None)[0]
    H = np.vstack((H, [1]))
    H = H.reshape(3, 3)
    return H

# warps the im using the given homography matrix
def warpImage(im, H):
    imWarped = im
    # compute the BOUNDING BOX
    height, width, na = im.shape
    imCorners = np.array([[0,0], [height, width], [0, width], [height, 0]])
    
    max_x = -100000000
    max_y = -100000000
    min_x = 100000000
    min_y = 100000000

    # FOR EACH CORNER: find warped corners
    for corner in imCorners:
        x = corner[0]
        y = corner[1]
        ## warp four corners of original image, keep min and max values
        warped = H.dot(np.array([x, y, 1]))
        warped = warped/warped[2]
        warp_x = int(warped[0])
        warp_y = int(warped[1])

        if warp_y < min_y:
            min_y = warp_y
        if warp_y > max_y:
            max_y = warp_y
        if warp_x < min_x:
            min_x = warp_x
        if warp_x > max_x:
            max_x = warp_x
    ## size of box = max values (x, y) - min values (x, y)
    bound_y = max_y - min_y
    bound_x = max_x - min_x
    warpCorners = [min_x, max_x, min_y, max_y]
    # out of bounds indexes: mask to black 0 pixel
    imWarped = np.zeros((bound_x, bound_y, 3))
    for new_x in range(bound_x):
        for new_y in range(bound_y):
            # INVERSE WARP
            warped = np.linalg.inv(H).dot(np.array([new_x + min_x, new_y + min_y, 1]))
            warped = warped/warped[2]
            x = int(warped[0])
            y = int(warped[1])
            # if IN BOUNDS: translate pixel value from original to warped
            if (x >= 0 and x < height and y >= 0 and y < width):
                imWarped[new_x, new_y, :] = im[x, y, :]
    return imWarped, warpCorners

def rectification():
    img1 = sk.img_as_float(io.imread(images_dir+"book.jpg"))
    book_down_pts = np.array([[ 156,  481],
 [ 146,  886],
 [ 619,  316],
 [ 619, 1136]])
    book_front_pts = np.array([[92,  452],
 [92,1116],
 [830,452],
 [830, 1116]])
    H = computeH(book_down_pts,book_front_pts)
    img,_ = warpImage(img1,H)
    img = np.clip(img, 0, 1)  
    img = (img * 255).astype(np.uint8) 
    io.imsave(images_dir+'re2.png',img)
    
if __name__ == '__main__':
    rectification()