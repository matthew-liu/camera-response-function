import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import curve_fit


def readImagesAndTimes(times, filenames):
    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images, times


def correctImage(image, crf, expTime):
    c_image = np.zeros_like(image)

    for (y, x, c), v in np.ndenumerate(image):
        # c_image[y, x, c] = crf[c, v] / expTime  # curve
        c_image[y, x, c] = crf[v, c] / expTime  # lookup table

    return c_image


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def fit_curve(y):
    xs = np.arange(0, len(y), 1)
    # y = Ae^Bx <-> logy = logA + Bx
    y_poly = np.polyfit(xs, np.log(y), 1, w=np.sqrt(y))
    # p0 = [np.exp(y_poly[1]), y_poly[0]]
    # [a, b], _ = curve_fit(lambda x, a, b: a * np.exp(b * x), xs, y, p0=p0)

    y_fit = []
    for x in xs:
        y_fit.append(np.exp(np.polyval(y_poly, x)))
        # y_fit.append(a * np.exp(b * x))

    return y_fit


def fitCRF(crf):
    r = crf[:, 0]
    g = crf[:, 1]
    b = crf[:, 2]

    r2 = fit_curve(r)
    g2 = fit_curve(g)
    b2 = fit_curve(b)

    print(len(r2))
    crf = np.array([r2, g2, b2])
    print(crf.shape)
    return crf


def show_images(images, titles=None, columns=5, max_rows=5, fileName=None):
    """Shows images in a titled format

    Args:
        images(list[np.array]): Images to show
        titles(list[string]): Titles for each of the images
        columns(int): How many columns to use in the tiling
        max_rows(int): If there are more than columns * max_rows images, only the first n of them will be shown.
    """

    images = images[:min(len(images), max_rows * columns)]

    plt.figure(figsize=(20, 10))
    for ii, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, ii + 1)
        plt.axis('off')
        if titles is not None and ii < len(titles):
            plt.title(str(titles[ii]))
        plt.imshow(image)

    if fileName is not None:
        plt.savefig(fileName)
    else:
        plt.show()


# List of exposure times in seconds
times = np.array([0.005, 0.02, 0.1, 0.2, 0.3], dtype=np.float32)

# List of image filenames
fileNames = ["im/5.png", "im/20.png", "im/100.png", "im/200.png", "im/300.png"]

images, times = readImagesAndTimes(times, fileNames)

# Align input images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# Obtain Camera Response Function (CRF) as lookup table
calibrateDebevec = cv2.createCalibrateDebevec()
crf = calibrateDebevec.process(images, times).squeeze()

# crf = np.loadtxt('crf.txt')
# np.savetxt('crf.txt', crf)

# crf = fitCRF(crf)

c_images = []
for i in range(len(images)):
    print(f'correcting image[{i}]...')
    c_image = adjust_gamma(correctImage(images[i], crf, times[i]), 3)
    c_images.append(c_image)
    # plt.figure()
    # plt.imshow(c_image)
    # plt.savefig('corrected' + str(i) + '.png')

show_images(images + c_images, fileNames, fileName='result')

