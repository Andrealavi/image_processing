import numpy as np
from skimage import io
import matplotlib.pyplot as plt

img = io.imread("./img/gauss_lena.gif")
img = np.reshape(img, newshape=(512, 512))

L = 2
K = 2
kH = 2 * L + 1
kW = 2 * K + 1

out = np.zeros(shape=(img.shape[0] - kH, img.shape[1] - kW))

H = out.shape[0]
W = out.shape[1]

for i in range(H):
    for j in range(W):
        out[i, j] = np.median(img[i:i+kH, j:j+kW, 0])

fig = plt.figure(figsize=(5, 3))

fig.add_subplot(1, 2, 1)

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("Original image with noise")

fig.add_subplot(1, 2, 2)

plt.imshow(out, cmap="gray")
plt.axis("off")
plt.title("Image after median filter application")

plt.show()
