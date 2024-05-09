import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def convolution(image, kernel):
    h, w = image.shape
    img_new = []
    for i in range(h - 3):
        line = []
        for j in range(w - 3):
            a = image[i:i+3, j:j+3]
            line.append(np.sum(np.multiply(kernel, a)))
        img_new.append(line)
    return np.array(img_new)


def self_erosion(image, kernel):

    h, w = image[:, :, 0].shape

    r = np.pad(image[:, :, 0], (1,1), 'constant')
    g = np.pad(image[:, :, 1], (1, 1), 'constant')
    b = np.pad(image[:, :, 2], (1, 1), 'constant')

    img_r_new, img_g_new, img_b_new = [], [], []
    for i in range(h):
        r_line, g_line, b_line = [], [], []
        for j in range(w):
            r_line.append(min(filter(lambda x: x > 0, r[i:i + kernel.shape[0], j:j + kernel.shape[0]].flatten())))
            g_line.append(min(filter(lambda x: x > 0, g[i:i + kernel.shape[0], j:j + kernel.shape[0]].flatten())))
            b_line.append(min(filter(lambda x: x > 0, b[i:i + kernel.shape[0], j:j + kernel.shape[0]].flatten())))

        img_r_new.append(r_line)
        img_g_new.append(g_line)
        img_b_new.append(b_line)
    erosion = np.array([np.array(img_r_new), np.array(img_g_new), np.array(img_b_new)]) # r,g,b -> [r,g,b]

    # numpy -> tensor for (3,256,256) -> (256, 256, 3)
    erosion = torch.tensor(erosion)
    erosion = erosion.permute(1, 2, 0).detach().numpy()

    # erosion_1 = cv2.erode(image, kernel)
    # np.array_equal(erosion, erosion_1)

    return erosion


    # image_pad = np.pad(image, (1,1), 'constant')


def self_dilation(image, kernel):
    h, w = image[:, :, 0].shape

    r = np.pad(image[:, :, 0], (1, 1), 'constant')
    g = np.pad(image[:, :, 1], (1, 1), 'constant')
    b = np.pad(image[:, :, 2], (1, 1), 'constant')

    img_r_new, img_g_new, img_b_new = [], [], []
    for i in range(h):
        r_line, g_line, b_line = [], [], []
        for j in range(w):
            r_line.append(max(filter(lambda x: x > 0, r[i:i + kernel.shape[0], j:j + kernel.shape[0]].flatten())))
            g_line.append(max(filter(lambda x: x > 0, g[i:i + kernel.shape[0], j:j + kernel.shape[0]].flatten())))
            b_line.append(max(filter(lambda x: x > 0, b[i:i + kernel.shape[0], j:j + kernel.shape[0]].flatten())))

        img_r_new.append(r_line)
        img_g_new.append(g_line)
        img_b_new.append(b_line)
    dilation = np.array([np.array(img_r_new), np.array(img_g_new), np.array(img_b_new)])  # r,g,b -> [r,g,b]

    # numpy -> tensor for (3,256,256) -> (256, 256, 3)
    dilation = torch.tensor(dilation)
    dilation = dilation.permute(1, 2, 0).detach().numpy()

    return dilation
if __name__ == '__main__':
    save_dir ='Output'
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.imread('Images/Acete.tif')
    image = cv2.resize(image, (256, 256))
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    k = np.array([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ])

    erosion = self_erosion(image, kernel)
    dilation = self_dilation(image, kernel)

    opening = self_dilation(erosion, kernel)
    closing = self_erosion(dilation, kernel)

    gradient = cv2.absdiff(erosion, dilation)
    top_hat = cv2.absdiff(image, opening)
    black_hat = cv2.absdiff(closing, image)

    titles = ['Original', 'Self_Erosion', 'Self_Dilation', 'Self_Opening', 'Self_Closing', 'Self_Gradient', 'Self_hot_hat', 'Self_black_hat']
    images = [image, erosion, dilation, opening, closing, gradient, top_hat, black_hat]
    plt.figure()
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.savefig(save_dir + "\\" + 'Morphology.png')
    plt.show()