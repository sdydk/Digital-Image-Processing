# Markov Random Field
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM
from skimage.segmentation import mark_boundaries
image = cv2.imread('Images/1689167501326.jpg')
image_gray = color.rgb2gray(image)
def unary_potential(x):
    return -x
def pairwise_potential(x1, x2):
    return -np.abs(x1 - x2)

X = image_gray.reshape(-1, 1)
y = (image_gray > 0.5).astype(np.int)

model = GraphCRF(directed=False)
model.add_unary_potentials(unary_potential, X)
model.add_pairwise_potentials(pairwise_potential, structure=np.array([[1, 1]]))

svvm = FrankWolfeSSVM(model, C=0.1)
svvm.fit(X, y)

y_pred = svvm.predict(X)
segmentation = y_pred.reshape(image_gray.shape)

plt.show(image_gray)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(image, segmentation))
plt.title("Segmentation Result")
plt.show()
if __name__ == '__main__':
    print("--------------------x")