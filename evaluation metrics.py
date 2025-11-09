import os
from PIL import Image
import cv2
from math import exp
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def PSNR(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1.cpu(), min=0, max=1)
    img2 = torch.clamp(img2.cpu(), min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def compute_average_psnr_ssim(folder1, folder2):
    psnr_values = []
    ssim_values = []

    for filename in os.listdir(folder1):
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        # 确保两个文件夹中存在相同文件名的图片
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # 转换为 numpy 数组并归一化到 [0, 1]
            img1_np = np.array(img1) / 255.0
            img2_np = np.array(img2) / 255.0

            # 将 numpy 数组转换为 PyTorch 张量
            img1_tensor = torch.tensor(img1_np).permute(2, 0, 1).unsqueeze(0)
            img2_tensor = torch.tensor(img2_np).permute(2, 0, 1).unsqueeze(0)

            # 计算 PSNR 和 SSIM
            psnr_value = PSNR(img1_tensor, img2_tensor)
            #psnr_value = batch_PSNR(img1_tensor, img2_tensor)

            psnr_values.append(psnr_value)

            ssim_value = SSIM(img1_tensor, img2_tensor)
            ssim_values.append(ssim_value)
        else:
            print(f"Warning: {filename} not found in both folders.")

    # 计算平均 PSNR 和 SSIM
    if len(psnr_values) == 0 or len(ssim_values) == 0:
        return None, None
    average_psnr = np.mean(psnr_values)
    average_ssim = np.mean(ssim_values)
    return average_psnr, average_ssim



def calculate_qr_code_metrics_opencv(folder1):
    """
    使用 OpenCV 的 QRCodeDetector 计算二维码的解码率和识别率
    :param folder1: 包含二维码图像的文件夹路径
    :param folder2: 包含对应 GT 数据的文件夹路径，GT 数据是二维码的预期内容
    :return: 解码率和识别率
    """
    detector = cv2.QRCodeDetector()  # OpenCV 的二维码解码器
    total_images = 0  # 总二维码图片数量
    successfully_decoded = 0  # 成功解码的二维码数量
    successfully_recognized = 0  # 成功识别内容的二维码数量

    # 遍历文件夹中的每个文件
    for file_name in os.listdir(folder1):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 只处理图像文件
            total_images += 1
            image_path = os.path.join(folder1, file_name)
            # gt_image_path = os.path.join(folder2, file_name)  # 假设 GT 数据与图像同名

            # 读取图像并尝试解码
            image = cv2.imread(image_path)
            decoded_text, _, _ = detector.detectAndDecode(image)  # 使用 OpenCV 解码二维码

            if decoded_text:  # 解码成功
                #print(decoded_text)
                successfully_decoded += 1  # 解码成功
                # 读取GT文件并比较解码结果
                # try:
                #     with open(gt_image_path, 'r',encoding='utf-8',errors='ignore') as gt_file:
                #         gt_data = gt_file.read().strip()  # 读取GT数据
                #         #print(gt_data)
                #         if decoded_text == gt_data:  # 解码数据与GT数据匹配
                #             successfully_recognized += 1  # 内容有效，识别成功
                # except FileNotFoundError:
                #     print(f"警告：未找到GT文件 {gt_image_path }")

    # 计算解码率和识别率
    decode_rate = (successfully_decoded / total_images) * 100 if total_images > 0 else 0
    recognition_rate = (successfully_recognized / total_images) * 100 if total_images > 0 else 0

    return decode_rate, recognition_rate

# 文件夹路径
folder1 =r"D:\zhuomian\QRcode\real_qrcode\target"#r"F:\pycharm\QR-main\root\output_images"#增强图像#


decode_rate, recognition_rate = calculate_qr_code_metrics_opencv(folder1)
print(f"解码率: {decode_rate:.2f}%")
print(f"识别率: {recognition_rate:.2f}%")