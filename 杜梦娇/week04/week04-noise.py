import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import util

def add_gaussian_noise(image, mean=0, std=25):
    # 获取图像尺寸
    h, w = image.shape

    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, std, (h, w))

    # 添加噪声
    noisy_image = np.clip(image + gaussian_noise, 0, 255)  # 确保像素值在0-255范围内
    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):

    # 获取图像尺寸
    h, w = image.shape

    # 创建噪声图像
    noisy_image = image.copy()

    # 添加椒噪声
    num_pepper = int(pepper_prob * h * w)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in [h, w]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    # 添加盐噪声
    num_salt = int(salt_prob * h * w)
    salt_coords = [np.random.randint(0, i, num_salt) for i in [h, w]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    return noisy_image


# 测试代码
if __name__ == "__main__":
    # 读取图像
    img = cv2.imread('lenna.png')
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    # 添加高斯噪声
    gaussian_noisy_image = add_gaussian_noise(image, mean=0, std=25)
    # 添加椒盐噪声
    salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
    print(gaussian_noisy_image.shape)
    print(salt_and_pepper_noisy_image.shape)
    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Image with Gaussian Noise")
    plt.imshow(gaussian_noisy_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Image with Salt and Pepper Noise")
    plt.imshow(salt_and_pepper_noisy_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')
    plt.show()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #使用接口实现噪声
    gs_noise_img = util.random_noise(image, mode='gaussian')
    s_p_noise_img = util.random_noise(image, mode='s&p')
    print(gs_noise_img.shape)
    print(s_p_noise_img.shape)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Image with Gaussian Noise")
    plt.imshow(gs_noise_img, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Image with Salt and Pepper Noise")
    plt.imshow(s_p_noise_img, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')
    plt.show()


