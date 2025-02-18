import numpy as np
import numpy.typing as npt
from skimage import io
import matplotlib.pyplot as plt
from scipy import signal


def change_angles(angles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    angles[(angles <= 0.392) | (angles > 2.74)] = 0
    angles[(angles <= 1.178) | (angles > 0.392)] = 0.78
    angles[(angles <= 1.963) | (angles > 1.178)] = 1.57
    angles[(angles <= 2.74) | (angles > 1.963)] = 2.35

    return angles


img: npt.NDArray[np.uint8] = io.imread("./img/original_lena.gif")
img = np.reshape(img, newshape=(512, 512))

plt.imshow(img, cmap='gray')
plt.show()

L: int = 2
K: int = 2
KH: int = 2 * L + 1
KW: int = 2 * K + 1

SD: float = 1.4

x, y = np.mgrid[(-KH // 2 + 1):(KH // 2 + 1), (-KW//2 + 1):(KW // 2 + 1)]

gauss_kernel: npt.NDArray[np.float32] = (
    1/(2 * np.pi * (SD**2))) * np.exp(-(x**2 + y**2)/(2 * SD**2))

img_denoised: npt.NDArray[np.uint8] = signal.convolve2d(img, gauss_kernel)
img_denoised = np.rint(img_denoised)

plt.imshow(img_denoised, cmap="gray")
plt.show()

v_kernel: npt.NDArray[np.int8] = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
h_kernel: npt.NDArray[np.int8] = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

v_der: npt.NDArray[np.uint8] = signal.convolve2d(img_denoised, v_kernel)
h_der: npt.NDArray[np.uint8] = signal.convolve2d(img_denoised, h_kernel)

v_der = np.rint(v_der)
h_der = np.rint(h_der)

grad_mod: npt.NDArray[np.uint8] = np.sqrt(v_der ** 2 + h_der ** 2)
theta: npt.NDArray[np.float32] = np.atan2(v_der, h_der)

theta = change_angles(theta)


plt.imshow(theta, cmap="gray")
plt.show()

non_max_suppr: npt.NDArray[np.uint8] = \
    np.zeros(shape=(grad_mod.shape[0] - KH, grad_mod.shape[1] - KW))

H: int = non_max_suppr.shape[0]
W: int = non_max_suppr.shape[1]

L: int = 0
K: int = 0
KH: int = 2 * L + 1
KW: int = 2 * K + 1

angles: npt.NDArray[np.int8] = np.array([0, 0.78, 1.57, 2.35])

for i in range(H):
    for j in range(W):
        if theta[i+KH, j+KW] == angles[0]:
            if grad_mod[i + KH, j + KW] > grad_mod[i + KH, j + KW + 1] \
                    and grad_mod[i + KH, j + KW] > grad_mod[i + KH, j + KW - 1]:
                non_max_suppr[i, j] = 255
        elif theta[i+KH, j+KW] == angles[1]:
            if grad_mod[i + KH, j + KW] > grad_mod[i + KH + 1, j + KW] \
                    and grad_mod[i + KH, j + KW] > grad_mod[i + KH - 1, j + KW]:
                non_max_suppr[i, j] = 255
        elif theta[i+KH, j+KW] == angles[2]:
            if grad_mod[i + KH, j + KW] > grad_mod[i + KH - 1, j + KW - 1] \
                    and grad_mod[i + KH, j + KW] > grad_mod[i + KH + 1, j + KW + 1]:
                non_max_suppr[i, j] = 255
        elif theta[i+KH, j+KW] == angles[3]:
            if grad_mod[i + KH, j + KW] > grad_mod[i + KH - 1, j + KW + 1] \
                    and grad_mod[i + KH, j + KW] > grad_mod[i + KH + 1, j + KW - 1]:
                non_max_suppr[i, j] = 255

fig = plt.figure()


fig.add_subplot(1, 2, 1)
plt.imshow(non_max_suppr, cmap='gray')
# plt.show()

weak = np.zeros(shape=(grad_mod.shape[0] - KH, grad_mod.shape[1] - KW))

for i in range(H):
    for j in range(W):
        if grad_mod[i+KH, j+KW] < 1.41:
            non_max_suppr[i, j] = 0
        elif grad_mod[i+KH, j+KW] > 11:
            weak[i, j] = 2
        else:
            weak[i, j] = 1

for i in range(H-KH):
    for j in range(W-KW):
        if 2 not in weak[i:i+KH, j:j+KW] and weak[i+KH, j+KW] == 1:
            non_max_suppr[i+KH, j+KW] = 0

fig.add_subplot(1, 2, 2)

non_max_suppr = np.rint(non_max_suppr)
non_max_suppr = non_max_suppr.clip(0, 255).astype("uint8")

plt.imshow(non_max_suppr, cmap='gray')
plt.show()
