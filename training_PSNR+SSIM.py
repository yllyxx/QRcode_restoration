import pandas as pd
import re
import matplotlib.pyplot as plt

# 读取日志文件
file_path = r'/denoising/dr_nfunet/train.log'
with open(file_path, 'r') as file:
    log_data = file.readlines()

# 提取 PSNR 和 SSIM 数据
psnr_data = []
ssim_data = []

for line in log_data:
    match = re.search(r'epoch:(\d+), iter:\s+([\d,]+), Average (PSNR|SSIM) : ([\d.]+)', line)
    if match:
        epoch = int(match.group(1))
        iteration = int(match.group(2).replace(',', ''))
        metric_type = match.group(3)
        value = float(match.group(4))

        if metric_type == 'PSNR':
            psnr_data.append((epoch, iteration, value))
        elif metric_type == 'SSIM':
            ssim_data.append((epoch, iteration, value))

# 转换为 DataFrame 以便于绘图
psnr_df = pd.DataFrame(psnr_data, columns=['Epoch', 'Iteration', 'PSNR'])
ssim_df = pd.DataFrame(ssim_data, columns=['Epoch', 'Iteration', 'SSIM'])

# 绘制 PSNR 曲线
plt.figure(figsize=(10, 6))
plt.plot(psnr_df['Iteration'], psnr_df['PSNR'], label='PSNR')
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR')
plt.legend()
plt.grid()
plt.show()

# 绘制 SSIM 曲线
plt.figure(figsize=(10, 6))
plt.plot(ssim_df['Iteration'], ssim_df['SSIM'], label='SSIM', color='orange')
plt.xlabel('Iteration')
plt.ylabel('SSIM')
plt.title('SSIM')
plt.legend()
plt.grid()
plt.show()


# import pandas as pd
# import re
# import matplotlib.pyplot as plt
#
# # 读取日志文件
# file_path = r'D:\zhuomian\QRcode\denoising\train.log'
# with open(file_path, 'r') as file:
#     log_data = file.readlines()
#
# # 提取 PSNR 和 SSIM 数据
# psnr_data = []
# ssim_data = []
#
# for line in log_data:
#     match = re.search(r'epoch:(\d+), iter:\s+([\d,]+), Average (PSNR|SSIM) : ([\d.]+)', line)
#     if match:
#         epoch = int(match.group(1))
#         iteration = int(match.group(2).replace(',', ''))
#         metric_type = match.group(3)
#         value = float(match.group(4))
#
#         if metric_type == 'PSNR':
#             psnr_data.append((epoch, iteration, value))
#         elif metric_type == 'SSIM':
#             ssim_data.append((epoch, iteration, value))
#
# # 转换为 DataFrame 以便于绘图
# psnr_df = pd.DataFrame(psnr_data, columns=['Epoch', 'Iteration', 'PSNR'])
# ssim_df = pd.DataFrame(ssim_data, columns=['Epoch', 'Iteration', 'SSIM'])
#
# # 绘制 PSNR 曲线，以 Epoch 为横坐标
# plt.figure(figsize=(10, 6))
# plt.plot(psnr_df['Epoch'], psnr_df['PSNR'], label='PSNR')
# plt.xlabel('Epoch')
# plt.ylabel('PSNR (dB)')
# plt.title('PSNR over Epochs')
# plt.legend()
# plt.grid()
# plt.show()
#
# # 绘制 SSIM 曲线，以 Epoch 为横坐标
# plt.figure(figsize=(10, 6))
# plt.plot(ssim_df['Epoch'], ssim_df['SSIM'], label='SSIM', color='orange')
# plt.xlabel('Epoch')
# plt.ylabel('SSIM')
# plt.title('SSIM over Epochs')
# plt.legend()
# plt.grid()
# plt.show()
