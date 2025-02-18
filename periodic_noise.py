import numpy as np
import numpy.typing as npt
from skimage import io
import matplotlib.pyplot as plt


img: npt.NDArray[np.uint8] = io.imread("./img/periodic_pidgeon.jpg")

plt.imshow(img, cmap="gray")
plt.show()
