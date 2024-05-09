import math
import cv2
import cv2 as cv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class HogExtraction():
    def __init__(self, image,cell_size=16, bin_size=8):
        self.image = image
        self.image = np.sqrt(image / np.max(image))
        self.image = image * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size
        assert type(self.bin_size) == int
        assert type(self.cell_size) == int
        assert type(self.angle_unit) == int

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centres = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centres[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centres[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centres
    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


    def extract(self):
        h, w = self.image.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)

        cell_gradient_vector = np.zeros((h // self.cell_size, w // self.cell_size, self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:(j+1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)
        hog_image = self.render_gradient(np.zeros([h, w]), cell_gradient_vector)
        hog_vector = []
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normaliza = lambda block_vector, magnitude: [element // magnitude for element in block_vector]
                    block_vector = normaliza(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image
if __name__ == '__main__':
    save_dir = 'Output'
    image = cv2.imread('Images/77.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    hog = HogExtraction(image, cell_size=8, bin_size=8)
    vector, image = hog.extract()
    print(np.array(vector).shape)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.savefig(save_dir + "\\" + 'HogExtraction.png')
    plt.show()

