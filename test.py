# import os.path
# import logging
#
# import numpy as np
# from collections import OrderedDict
#
# import torch
#
# from utils import utils_logger
# from utils import utils_model
# from utils import utils_image as util
#
#
# """
# Spyder (Python 3.7)
# PyTorch 1.6.0
# Windows 10 or Linux
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/DPIR
#         https://github.com/cszn/IRCNN
#         https://github.com/cszn/KAIR
# @article{zhang2020plug,
#   title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
#   author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
#   journal={arXiv preprint},
#   year={2020}
# }
# % If you have any question, please feel free to contact with me.
# % Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
# by Kai Zhang (01/August/2020)
#
# # --------------------------------------------
# |--model_zoo               # model_zoo
#    |--drunet_gray          # model_name, for color images
#    |--drunet_color
# |--testset                 # testsets
#    |--set12                # testset_name
#    |--bsd68
#    |--cbsd68
# |--results                 # results
#    |--set12_dn_drunet_gray # result_name = testset_name + '_' + 'dn' + model_name
#    |--set12_dn_drunet_color
# # --------------------------------------------
# """
#
#
# def main():
#
#     # ----------------------------------------
#     # Preparation
#     # ----------------------------------------
#
#     noise_level_img = 0
#     # set AWGN noise level for noisy image 36.057| 51 | 62.461 | 72.1249 | 80.631
#     noise_level_model = noise_level_img  # set noise level for model
#     model_name = 'drunet_gray'           # set denoiser model, 'drunet_gray' | 'drunet_color'
#     testset_name = 'input'               # set test set,  'waterloo_test'|'spakel' |'bsd68' | 'cbsd68' | 'set12'| 'waterloo_test' | 'ultransound_150'
#     x8 = False                           # default: False, x8 to boost performance
#     show_img = False                    # default: False
#     border = 0                           # shave boader to calculate PSNR and SSIM
#
#     if 'color' in model_name:
#         n_channels = 3                   # 3 for color image
#     else:
#         n_channels = 1                   # 1 for grayscale image
#
#     model_pool = 'model_zoo'             # fixed
#     testsets = 'testsets'                # fixed
#     results = 'results_qrcode_nafdrunet'        # fixed
#     task_current = 'dn'                 # 'dn' for denoising
#     result_name = testset_name + '_' + task_current + '55'
#
#     model_path = r'model_zoo/125000_G.pth'            #|model_zoo/naf33_160000_G(5.29).pth |  model_zoo/双域条带_145000_G.pth |  model_zoo/320000_G（双域条带）.pth
#                                                                   #|model_zoo/DRUNET-110000_G.pth  |model_zoo/shuangyu_smfa_415000.pth
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.cuda.empty_cache()
#
#     # ----------------------------------------
#     # L_path, E_path, H_path
#     # ----------------------------------------
#
#     L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
#     E_path = os.path.join(results, result_name)   # E_path, for Estimated images
#     util.mkdir(E_path)
#
#     logger_name = result_name
#     utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
#     logger = logging.getLogger(logger_name)
#
#     # ----------------------------------------
#     # load model
#     # ----------------------------------------
#
#     from models.network_unet import UNetRes as net
#     model = net(in_nc=n_channels, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
#     model.load_state_dict(torch.load(model_path), strict=True)
#     model.eval()
#     for k, v in model.named_parameters():
#         v.requires_grad = False
#     model = model.to(device)
#     logger.info('Model path: {:s}'.format(model_path))
#     number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
#     logger.info('Params number: {}'.format(number_parameters))
#
#     test_results = OrderedDict()
#     test_results['psnr'] = []
#     test_results['ssim'] = []
#
#     logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
#     logger.info(L_path)
#     L_paths = util.get_image_paths(L_path)
#
#     for idx, img in enumerate(L_paths):
#
#         # ------------------------------------
#         # (1) img_L
#         # ------------------------------------
#
#         img_name, ext = os.path.splitext(os.path.basename(img))
#         # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
#         img_H = util.imread_uint(img, n_channels=n_channels)
#         img_L = util.uint2single(img_H)
#
#         # Add noise without clipping   作为超声去噪时注释添加噪声
#         np.random.seed(seed=0)  # for reproducibility
#         # img_L += np.sqrt(img_L)*np.random.normal(0, noise_level_img/255., img_L.shape)
#         # img_L += np.random.normal(0, noise_level_img / 255., img_L.shape)
#         # noise_map = noise_estimate(img_L, 8)
#         util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None
#
#         img_L = util.single2tensor4(img_L)
#
#         # #暂时取消噪声图试试效果
#         # img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
#         img_L = img_L.to(device)
#
#         #与验证策略相同
#
#
#         # ------------------------------------
#         # (2) img_E
#         # ------------------------------------
#
#         if not x8 and img_L.size(2)//8==0 and img_L.size(3)//8==0:
#             img_E = model(img_L)
#         elif not x8 and (img_L.size(2)//8!=0 or img_L.size(3)//8!=0):
#             img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
#         elif x8:
#             img_E = utils_model.test_mode(model, img_L, mode=3)
#
#         img_E = util.tensor2uint(img_E)
#
#         # --------------------------------
#         # PSNR and SSIM
#         # --------------------------------
#
#         if n_channels == 1:
#             img_H = img_H.squeeze()
#         psnr = util.calculate_psnr(img_E, img_H, border=border)
#         ssim = util.calculate_ssim(img_E, img_H, border=border)
#         test_results['psnr'].append(psnr)
#         test_results['ssim'].append(ssim)
#         logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
#
#         # ------------------------------------
#         # save results
#         # ------------------------------------
#
#         util.imsave(img_E, os.path.join(E_path, img_name+ext))
#
#     ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#     ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#     logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))
#
#
# if __name__ == '__main__':
#
#     main()











import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util

def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_gray'           # set denoiser model
    noisy_images_path = r'D:\zhuomian\QRcode\testsets\real_qrcode'  # 指定噪声图像文件夹
    clean_images_path = r'D:\zhuomian\QRcode\testsets\real_qrcode'  # 指定干净图像文件夹
    x8 = False                           # default: False, x8 to boost performance
    show_img = False                    # default: False
    border = 0                           # shave border to calculate PSNR and SSIM

    if 'color' in model_name:
        n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image

    results = 'results_QRCode_proposed_real'  # fixed
    task_current = 'dn'                 # 'dn' for denoising
    result_name = 'evaluation_qrcode_proposed_real'

    model_path = r'model_zoo/330000_G_proposed_gray_12.20.pth'  # specify your model path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))

    # 读取噪声图像路径
    L_paths = util.get_image_paths(noisy_images_path)

    for idx, img in enumerate(L_paths):
        img_name, ext = os.path.splitext(os.path.basename(img))

        # 读取干净图像
        img_H = util.imread_uint(os.path.join(clean_images_path, img_name + ext), n_channels=n_channels)
        img_L = util.uint2single(util.imread_uint(img, n_channels=n_channels))  # 读取噪声图像

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # 生成去噪图像
        if not x8 and img_L.size(2) // 8 == 0 and img_L.size(3) // 8 == 0:
            img_E = model(img_L)
        elif not x8 and (img_L.size(2) // 8 != 0 or img_L.size(3) // 8 != 0):
            img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
        elif x8:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        # 计算 PSNR 和 SSIM
        if n_channels == 1:
            img_H = img_H.squeeze()
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name + ext, psnr, ssim))

        # 保存结果
        util.imsave(img_E, os.path.join(E_path, img_name + ext))

    # 计算并记录平均 PSNR 和 SSIM
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':
    main()


# import os.path
# import logging
# import numpy as np
# from collections import OrderedDict
# import torch
# from utils import utils_logger
# from utils import utils_model
# from utils import utils_image as util
# import cv2
#
#
# def main():
#     # ----------------------------------------
#     # Preparation
#     # ----------------------------------------
#     noise_level_img = 0
#     noise_level_model = noise_level_img
#     model_name = 'drunet_color'
#     noisy_images_path = r'D:\zhuomian\QRcode\real_qrcode\input'
#     clean_images_path = r'D:\zhuomian\QRcode\real_qrcode\target'
#     x8 = False
#     show_img = False
#     border = 0
#
#     if 'color' in model_name:
#         n_channels = 1
#     else:
#         n_channels = 1
#
#     results = 'results_real'
#     task_current = 'dn'
#     result_name = 'evaluation_qrcode_shouji_1'
#
#     model_path = r'model_zoo/shuangyu_smfa_naf.pth'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.cuda.empty_cache()
#
#     # ----------------------------------------
#     # L_path, E_path, H_path
#     # ----------------------------------------
#     E_path = os.path.join(results, result_name)
#     util.mkdir(E_path)
#
#     logger_name = result_name
#     utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
#     logger = logging.getLogger(logger_name)
#
#     # ----------------------------------------
#     # load model
#     # ----------------------------------------
#     from models.network_unet import UNetRes as net
#     model = net(in_nc=n_channels, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
#                 downsample_mode="strideconv", upsample_mode="convtranspose")
#     model.load_state_dict(torch.load(model_path), strict=True)
#     model.eval()
#     for k, v in model.named_parameters():
#         v.requires_grad = False
#     model = model.to(device)
#     logger.info('Model path: {:s}'.format(model_path))
#     number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
#     logger.info('Params number: {}'.format(number_parameters))
#
#     test_results = OrderedDict()
#     test_results['psnr'] = []
#     test_results['ssim'] = []
#
#     logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(
#         model_name, noise_level_img, noise_level_model))
#
#     # Read noisy image paths
#     L_paths = util.get_image_paths(noisy_images_path)
#
#     for idx, img in enumerate(L_paths):
#         img_name, ext = os.path.splitext(os.path.basename(img))
#
#         # Read and resize clean image
#         img_H = util.imread_uint(os.path.join(clean_images_path, img_name + ext), n_channels=n_channels)
#         img_H = cv2.resize(img_H, (512, 512), interpolation=cv2.INTER_LINEAR)
#
#         # Read and resize noisy image
#         img_L = util.imread_uint(img, n_channels=n_channels)
#         img_L = cv2.resize(img_L, (512, 512), interpolation=cv2.INTER_LINEAR)
#         img_L = util.uint2single(img_L)
#
#         img_L = util.single2tensor4(img_L)
#         img_L = img_L.to(device)
#
#         # Generate denoised image
#         if not x8 and img_L.size(2) // 8 == 0 and img_L.size(3) // 8 == 0:
#             img_E = model(img_L)
#         elif not x8 and (img_L.size(2) // 8 != 0 or img_L.size(3) // 8 != 0):
#             img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)
#         elif x8:
#             img_E = utils_model.test_mode(model, img_L, mode=3)
#
#         img_E = util.tensor2uint(img_E)
#
#         # Save results
#         util.imsave(img_E, os.path.join(E_path, img_name + ext))
#
#
# if __name__ == '__main__':
#     main()