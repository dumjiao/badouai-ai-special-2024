import numpy as np
import cv2
import matplotlib.pyplot as plt

#  高斯滤波函数
def gaussian_filter(size, sigma):
    dim = size // 2
    x, y = np.mgrid[-dim:dim+1, -dim:dim+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

#  高斯平滑函数
def gaussian_smoothing(img, filter_size, sigma):
    gaussian_kernel = gaussian_filter(filter_size, sigma)
    pad_width = filter_size // 2
    img_pad = np.pad(img, pad_width, mode='constant')
    dx, dy = img.shape
    img_smoothed = np.zeros_like(img)

    for i in range(dx):
        for j in range(dy):
            img_smoothed[i, j] = np.sum(img_pad[i:i+filter_size, j:j+filter_size] * gaussian_kernel)
    return img_smoothed

#  求梯度函数
def gradient_calculation(img):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    dx, dy = img.shape
    img_pad = np.pad(img, 1, mode='constant')
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros(img.shape)
    img_tidu = np.zeros(img.shape)
    angle = np.zeros(img.shape)

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
            angle[i, j] = np.arctan2(img_tidu_y[i, j], img_tidu_x[i, j] + 1e-7)  # 避免除零
    return img_tidu, angle

#  非极大值抑制函数
def non_maximum_suppression(img_tidu, angle):
    dx, dy = img_tidu.shape
    img_yizhi = np.zeros_like(img_tidu)

    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True
            temp = img_tidu[i-1:i+2, j-1:j+2]
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    return img_yizhi

#  双阈值检测函数
def double_threshold(img_yizhi, low, high):
    dx, dy = img_yizhi.shape
    zhan = []

    for i in range(1, dx-1):
        for j in range(1, dy-1):
            if img_yizhi[i, j] >= high:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= low:
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high) and (a[0, 0] > low):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high) and (a[0, 1] > low):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high) and (a[0, 2] > low):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high) and (a[1, 0] > low):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high) and (a[1, 2] > low):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high) and (a[2, 0] > low):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high) and (a[2, 1] > low):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high) and (a[2, 2] > low):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    return img_yizhi

# 主函数
if __name__ == '__main__':
    img = cv2.imread("lenna.png", 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 高斯平滑
    sigma = 0.5
    filter_size = 5
    img_smoothed = gaussian_smoothing(img, filter_size, sigma)
    plt.figure(1)
    plt.imshow(img_smoothed.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title("Gaussian Smoothing")

    # 2. 求梯度
    img_tidu, angle = gradient_calculation(img_smoothed)
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title("Gradient Magnitude")

    # 3. 非极大值抑制
    img_yizhi = non_maximum_suppression(img_tidu, angle)
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title("Non-Maximum Suppression")

    # 4. 双阈值检测
    low = img_tidu.mean() * 0.5
    high = low * 3
    img_result = double_threshold(img_yizhi, low, high)
    plt.figure(4)
    plt.imshow(img_result.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title("Edge Detection")

    plt.show()



# 调用接口实现canny
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 50, 300))
cv2.waitKey()
cv2.destroyAllWindows()
