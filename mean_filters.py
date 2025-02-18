import numpy as np
import numpy.typing as npt
from skimage import io
import matplotlib.pyplot as plt
from scipy import signal


def add_padding(img: npt.NDArray, pad_x=0, pad_y=0, fill="zeros") -> npt.NDArray[np.uint8]:
    h, w = img.shape

    match fill:
        case "zeros":
            pad_img: npt.NDArray[np.uint8] = np.zeros(
                shape=(h + 2*pad_x, w + 2*pad_y))
        case "ones":
            pad_img: npt.NDArray[np.uint8] = np.ones(
                shape=(h + 2*pad_x, w + 2*pad_y))
        case _:
            raise ValueError("Invalid fill value for padding")

    pad_img[pad_x:-pad_x, pad_y:-pad_y] = img

    return pad_img


def arithmetic_filter(img: npt.NDArray[np.uint8], kH: int, kW: int) -> npt.NDArray[np.uint8]:
    kernel: npt.NDArray[np.float32] = (1/(kH * kW)) * np.ones(shape=(kH, kW))

    return np.rint(signal.convolve2d(img, kernel))


def geometric_filter(img: npt.NDArray[np.uint8], kH: int, kW: int) -> npt.NDArray[np.uint8]:
    h, w = img.shape

    pad_img = add_padding(img, pad_x=kH, pad_y=kW)
    filtered_img = np.zeros(shape=(h, w))

    for i in range(h):
        for j in range(w):
            filtered_img[i, j] = np.power(np.prod(
                pad_img[i:i + kH, j:j + kW]), (1/(kW*kH)))

    return np.rint(filtered_img)


def harmonic_filter(img: npt.NDArray[np.uint8], kH: int, kW: int) -> npt.NDArray[np.uint8]:
    h, w = img.shape

    pad_img = add_padding(img, pad_x=kH, pad_y=kW, fill="ones")
    filtered_img = np.zeros(shape=(h, w))

    for i in range(h):
        for j in range(w):
            window = pad_img[i:i+kH, j:j+kW]

            filtered_img[i, j] = (kH*kW)/np.sum(1 / window[window != 0])

    return np.rint(filtered_img)


def contraharmonic_filter(img: npt.NDArray[np.uint8], kH: int, kW: int, q: int) -> npt.NDArray[np.uint8]:
    h, w = img.shape

    pad_img = add_padding(img, pad_x=kH, pad_y=kW)
    filtered_img = np.zeros(shape=(h, w))

    for i in range(h):
        for j in range(w):
            window = pad_img[i:i+kH, j:j+kW]

            numerator = np.sum(np.power(window[window != 0], q+1))

            denominator = np.sum(np.power(window[window != 0], q))

            if denominator == 0 or np.isnan(numerator) or np.isnan(denominator):
                filtered_img[i, j] = window[kH // 2, kW // 2]
            else:
                filtered_img[i, j] = numerator/denominator

    return np.rint(filtered_img)


original_img: npt.NDArray[np.uint8] = io.imread(
    "./img/original_lena.gif")[0, :, :]
gauss_img: npt.NDArray[np.uint8] = io.imread("./img/gauss_lena.gif")[0, :, :]
salt_pepper_img: npt.NDArray[np.uint8] = io.imread(
    "./img/saltpepper_lena.gif")[0, :, :]


KH: int = 3
KW: int = 3

fig = plt.figure()

ROWS: int = 2
COLUMNS: int = 4

fig.add_subplot(ROWS, COLUMNS, 1)

plt.imshow(original_img, cmap="gray")
plt.title("Original Image")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 2)

plt.imshow(gauss_img, cmap="gray")
plt.title("Image with Gaussian Noise")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 3)

plt.imshow(gauss_img, cmap="gray")
plt.title("Image with Salt and Pepper Noise")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 4)

plt.imshow(arithmetic_filter(gauss_img, KH, KW), cmap="gray")
plt.title("After Arithmetic Mean Filter")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 5)

plt.imshow(geometric_filter(gauss_img, KH, KW), cmap="gray")
plt.title("Geometric Mean Filter")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 6)

plt.imshow(harmonic_filter(salt_pepper_img, KH, KW), cmap="gray")
plt.title("Harmonic Mean Filter")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 7)

plt.imshow(contraharmonic_filter(salt_pepper_img, KH, KW, 1.5), cmap="gray")
plt.title("Contraharmonic Mean Filter")
plt.axis("off")


fig.add_subplot(ROWS, COLUMNS, 8)

plt.imshow(contraharmonic_filter(salt_pepper_img, KH, KW, -1.5), cmap="gray")
plt.title("Contraharmonic Mean Filter")
plt.axis("off")

plt.show()
