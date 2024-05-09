
import torch
import torchvision
# print(torch.cuda.is_available())
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())

import os
import cv2
import math
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
save_dir = 'Output'

# RGB to Gray
def image_to_gray(images, rows, cols, images_gray, labels):
    cv_gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)  # RGB to gray
    if labels == 0: #average
        for r in range(rows):
            for l in range(cols):
                images_gray[r, l] = 1 / 3 * images[r, l, 0] + 1 / 3 * images[r, l, 1] + 1 / 3 * images[r, l, 2]
    else: # weightedAverage
        for r in range(rows):
            for l in range(cols):
                images_gray[r, l] = 0.11 * images[r, l, 0] + 0.59 * images[r, l, 1] + 0.3 * images[r, l, 2]

    images_gray = np.hstack((images_gray, cv_gray))


    cv2.namedWindow('image_gray', cv2.WINDOW_FREERATIO)
    # save picture
    cv2.imwrite(save_dir + "\\" + 'image_gray.jpg', images_gray)
    cv2.imshow('image_gray', images_gray.astype("uint8"))
    cv2.waitKey(0)


    images_gray_inverse = np.zeros_like(cv_gray)
    for r in range(rows):
        for l in range(cols):
            images_gray_inverse[r, l] = 255 - cv_gray[r, l]

    images_gray_inverse = np.hstack((cv_gray, images_gray_inverse))
    cv2.namedWindow('image_gray_inverse', cv2.WINDOW_FREERATIO)
    cv2.imwrite(save_dir + "\\" + 'image_gray_inverse.jpg', images_gray_inverse)
    cv2.imshow('image_gray_inverse', images_gray_inverse.astype("uint8"))
    cv2.waitKey(0)


    # Canny edge detection, a multi-stage
    edges = cv2.Canny(cv_gray, 1, 10)
    plt.subplot(121)
    plt.imshow(cv_gray, cmap="gray")
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Image")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_dir + "\\" + 'image_canny.png', dpi=300)
    plt.show()

def plt_zooplankton_image(images, images_gray, image_laplacian_kernel, image_laplacian, image_scharr_kernel1, image_sobel):
    plt.subplot(231)
    plt.imshow(images, cmap="gray")
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(232)
    plt.imshow(images_gray, cmap="gray")
    plt.title("Gray")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(233)
    plt.imshow(image_laplacian_kernel, cmap="gray")
    plt.title("Laplacian_kernel")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(234)
    plt.imshow(image_laplacian, cmap="gray")
    plt.title("Laplacian")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(235)
    plt.imshow(image_scharr_kernel1, cmap="gray")
    plt.title("Scharr")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(236)
    plt.imshow(image_sobel, cmap="gray")
    plt.title("Sobel")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_dir + "\\" + 'Edge Detection.png', dpi=300)
    plt.show()

def image_edge_detection(laplacian_kernel_label, scharr_kernel_label):
    # 图像增强/边缘检测：Laplacian，Sobel，Scharr（突出细节）
    images = cv2.imread('Images/62.jpg')
    images_gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)

    if laplacian_kernel_label == 0:
        kernel = np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype="float32")
    elif laplacian_kernel_label == 1:
        kernel = np.array((
            [1, 1, 1],
            [1, 8, 1],
            [1, 1, 1]), dtype="float32")
    else:
        kernel = np.array((
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]), dtype="float32")

    if scharr_kernel_label == 0:
        scharr_kernel = np.array((
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]), dtype="float32")
    else:
        scharr_kernel = np.array((
            [-3, -10, 3],
            [0, 0, 0],
            [3, 10, 3]), dtype="float32")






    image_laplacian_kernel = cv2.filter2D(images_gray, -1, kernel)
    image_scharr_kernel1 = cv2.filter2D(images_gray, -1, scharr_kernel)
    image_laplacian = cv2.Laplacian(images_gray, -1, ksize=1) #ksize=3--->laplacian
    image_sobel = cv2.Laplacian(images_gray, -1, ksize=3)  #ksize=3--->Sobel

    plt_zooplankton_image(images, images_gray,
                          image_laplacian_kernel, image_laplacian,
                          image_scharr_kernel1, image_sobel)


    # result_image1 = np.hstack((images_gray, image_laplacian_kernel))
    # result_image2 = np.hstack((image_laplacian, image_sobel))
    # image_laplacian_sobel = np.vstack((result_image1, result_image2))
    #
    # images_subtract = cv2.subtract(image_laplacian, image_sobel)
    # # sharpen images (original and laplacian)
    # images_add = cv2.add(image_laplacian, images_gray)
    # images_operation = np.hstack((images_subtract, images_add))
    #
    # cv2.namedWindow('images_operation', cv2.WINDOW_FREERATIO)
    # cv2.imshow('images_operation', images_operation)
    # cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
    # cv2.imshow('result', image_laplacian_sobel)
    # cv2.imwrite(save_dir + "\\" + 'image_operation.jpg', images_operation)
    # cv2.imwrite(save_dir + "\\" + 'image_laplacian_sobel.jpg', image_laplacian_sobel)
    # cv2.waitKey(0)

# 信噪比
def psnr(image_original, image_noise):
    # 图像峰值信噪比，衡量图像质量高低的重要指标；信噪比大，图像画面就干净，看不到噪声干扰（“颗粒、雪花”）
    # >40dB:图像质量极好（非常接近原始图像）
    # 30dB~40dB:图像质量是好的（失真可以察觉但可以接收）
    # 20dB~30dB:图像质量差
    # < 20dB:图像质量不可接受
    mse = np.mean((image_original / 255. - image_noise / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def snr(image_original):
    # , image_noise
    # noise_signal = np.sum((image_noise - image_original) ** 2)
    # clean_signal = np.sum((image_original) ** 2)
    # snrr = 20 * math.log10(math.sqrt(clean_signal) / math.sqrt(noise_signal))

    image_data_mean, image_data_stddev = cv2.meanStdDev(image_original) # 均值  标准差
    image_data_variance = np.square(image_data_stddev)  #方差
    SNR = (image_data_mean / image_data_stddev).mean()


    return SNR

# 增加噪声
def image_add_noise(image):
    # Gaussian Noise
    gaussianNoise = np.random.normal(0, 20, size=image.size).reshape(image.shape[0], image.shape[1], image.shape[2])
    imageGaussianNoise = (image + gaussianNoise)
    imageGaussianNoise = np.clip(imageGaussianNoise, 0, 255) / 255

    # 灰度处理
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageGaussianNoise_gray = cv2.cvtColor((image + gaussianNoise).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    PSNR = psnr(image_gray, imageGaussianNoise_gray)
    SNR = snr(image_gray, imageGaussianNoise_gray)
    print(PSNR, SNR)




    # lam 越大 噪声程度越深
    poissonNoise = np.random.poisson(lam=20, size=image.size).reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    imagePoissonNoise = np.clip((image+poissonNoise), 0, 255) / 255

    # 椒盐噪声  SNR越小 噪声越大
    x = image.reshape(1, -1)
    SNR = 0.85 #Signal noise ratio
    noise_number = x.size * (1 - SNR)
    list = random.sample(range(0, x.size), int(noise_number))
    for i in list:
        if random.random() >= 0.5:
            x[0][i] = 0
        else:
            x[0][i] = 255
    imageSaltPepperNoise = x.reshape(image.shape)


    plt.subplot(221)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow(imageGaussianNoise, cmap="gray")
    plt.title("Gaussian noise")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(223)
    plt.imshow(imagePoissonNoise, cmap="gray")
    plt.title("Poisson noise")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(imageSaltPepperNoise, cmap="gray")
    plt.title("SaltPepper noise")
    plt.xticks([])
    plt.yticks([])

    plt.show()

    # cv2.imshow('Gauss noise', imageNosie)
    # cv2.namedWindow('Gauss noise', cv2.WINDOW_FREERATIO)
    # cv2.waitKey(0)

# 去噪
def image_remove_noise(image,label):

    noise = np.random.normal(0, 50, size=image.size).reshape(image.shape[0], image.shape[1], image.shape[2])
    imageNoise = np.clip(image + noise, 0, 255) / 255

    # image1[channel][i:i + m, j:j + n].sum() / (m * n)

    # average filter
    image1 = np.transpose(imageNoise, (2, 0, 1))
    m = 3
    n = 3
    d = 4

    rec_image_average = rec_image_geometry = rec_image_harmonic_wave = rec_image_median \
        = rec_image_midpoint = rec_image_alpha = np.zeros((image1.shape[0], image1.shape[1]-m+1, image1.shape[2]-n+1))



    for channel in range(rec_image_average.shape[0]):
        for i in range(rec_image_average[channel].shape[0]):
            for j in range(rec_image_average[channel].shape[1]):
                rec_image_average[channel][i, j] = image1[channel][i:i + m, j:j + n].sum() / (m * n)  # 算术平均滤波
                rec_image_geometry[channel][i, j] = np.power(np.prod(image1[channel][i:i + m, j:j + n]), 1 / (m * n))  # 几何均值滤波
                np.seterr(divide='ignore', invalid='ignore')
                rec_image_harmonic_wave[channel][i, j] = 1 / (np.power(image1[channel][i:i + m, j:j + n], -1).sum()) * (m * n)  # 谐波平均滤波
                rec_image_median[channel][i, j] = np.median(image1[channel][i:i + m, j:j + n])  # 中值滤波
                rec_image_midpoint[channel][i, j] = (np.amax(image1[channel][i:i + m, j:j + n]) + np.amin(image1[channel][i:i + m, j:j + n])) / 2  # 中点滤波

                image2 = np.sort(np.ravel(image1[channel][i:i + m, j:j + n]))
                rec_image_alpha[channel][i, j] = (image2[int(d / 2):-int(d / 2)].sum()) * (1 / (m * n - d))  # 修正阿尔法均值滤波 alpha average

    rec_image_average = np.transpose(rec_image_average, (1, 2, 0))  #转置函数，对高维数组进行坐标轴上的变换，即转置操作
    rec_image_geometry = np.transpose(rec_image_geometry, (1, 2, 0))
    rec_image_harmonic_wave = np.transpose(rec_image_harmonic_wave, (1, 2, 0))
    rec_image_median = np.transpose(rec_image_median, (1, 2, 0))
    rec_image_midpoint = np.transpose(rec_image_midpoint, (1, 2, 0))
    rec_image_alpha = np.transpose(rec_image_alpha, (1, 2, 0))




    plt.subplot(331)
    plt.imshow(image, cmap="gray")
    plt.title("image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(332)
    plt.imshow(imageNoise, cmap="gray")
    plt.title("imageNoise")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(333)
    plt.imshow(rec_image_average, cmap="gray")
    plt.title("rec_image_average")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(334)
    plt.imshow(rec_image_geometry, cmap="gray")
    plt.title("rec_image_geometry")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(335)
    plt.imshow(rec_image_harmonic_wave, cmap="gray")
    plt.title("rec_image_harmonic_wave")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(336)
    plt.imshow(rec_image_median, cmap="gray")
    plt.title("rec_image_median")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(337)
    plt.imshow(rec_image_midpoint, cmap="gray")
    plt.title("rec_image_midpoint")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(338)
    plt.imshow(rec_image_alpha, cmap="gray")
    plt.title("rec_image_alpha")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_dir + "\\" + 'Remove Noise.png')
    plt.show()

# 直方图
def Histogram(image, label):

    # color = ('b', 'g', 'r')
    # for i, col in enumerate(color):
    #     histr = cv2.calcHist([image], [i], None, [256], [0, 256])
    #     plt.plot(histr, color = col)
    #     plt.xlim([0, 256])
    # plt.title("Matplotlib color Method")
    # plt.show()

    mask = np.zeros(image.shape[:2], np.uint8)
    mask[200:800, 700:1100] = 255
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    hist_full = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([image], [0], mask, [256], [0, 256])


    # plt.subplot(321), plt.imshow(image,'gray'), plt.title('image')
    # plt.subplot(322), plt.imshow(mask, 'gray'), plt.title('mask')
    # plt.subplot(323), plt.imshow(masked_image, 'gray'), plt.title('masked_image')
    # plt.subplot(324), plt.plot(hist_full), plt.plot(hist_mask), plt.title('hist_color')
    # plt.subplot(325), plt.hist(image.ravel(), 255, [0, 256]), plt.title('hist')
    # plt.xlim([0, 256])
    # plt.show()


    fig = plt.figure(dpi=100, constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    fig.add_subplot(gs[0, 0:1]), plt.imshow(image, 'gray'), plt.title('image')
    fig.add_subplot(gs[0, 1:2]), plt.imshow(mask, 'gray'), plt.title('mask')
    fig.add_subplot(gs[1, 0:1]), plt.imshow(masked_image, 'gray'), plt.title('masked_image')
    fig.add_subplot(gs[1, 1:2]), plt.plot(hist_full), plt.plot(hist_mask), plt.title('hist_color')
    fig.add_subplot(gs[2, 0:2]), plt.hist(image.ravel(), 255, [0, 256]), plt.title('hist')
    fig.savefig(save_dir + "\\" + 'Histogram.png', dpi=100)
    plt.show()

# 文件重命名
def rname(path):
    filelist = os.listdir(path)
    filelist.sort()
    total_num = len(filelist)
    i = 0
    for item in filelist:
        if item.endswith('.png') or item.endswith('.jpg'):
            src = os.path.join(path, item)
            dst = os.path.join(os.path.abspath(path), path.split('\\')[3] + str(i) + '.png')

            try:
                os.rename(src, dst)
                print('coverting %s to %s ... ' % (src, dst))
                i = i + 1
            except Exception as e:
                print(e)
                print('rename dir fail\r\n')
    print('total %d to rename & converted %d jpgs' % (total_num, i))




if __name__ == '__main__':
    # image = cv2.imread(r'F:\zooplanktonDataset\ZooScanSet\imgs\Aglaura/47940543.jpg')
    image = cv2.imread('Images/86.jpg')

    image = cv2.resize(image, (256, 256))
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))
    label = 0

    # SNR = snr(image) # 信噪比


    # path = r'F:\zooplanktonDataset\archive\Skeletonema'
    # rname(path)

    # Histogram(image, label=0)  # 直方图
    # image_add_noise(image)
    image_remove_noise(image, label)
    # image_edge_detection(laplacian_kernel_label=0, scharr_kernel_label=0)
    # image_to_gray(image, row, col, image_gray, label)










