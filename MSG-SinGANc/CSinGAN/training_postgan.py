import os
import torch
import torch.nn as nn
import torch.optim as optim

import functions
import U_Net_2D
from config import get_arguments

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_G(opt):
    netG = U_Net_2D.Generator(1, opt).to(opt.device)
    netG.apply(U_Net_2D.weights_init)
    return netG

def init_D(opt):
    netD = U_Net_2D.Discriminator(1).to(opt.device)
    netD.apply(U_Net_2D.weights_init)
    return netD

def generate_samples(netG, reals_shapes, noise_amp, condition_pyramid, n=1):

    # 创建生成的文件夹
    dir2save_parent = os.path.join(dir2save, 'random_samples')
    try:
        os.makedirs(dir2save_parent)
    except OSError:
        pass

    sample_conditioned = []

    # 循环生成
    for idx in range(n):
        noise = functions.sample_random_noise(opt.train_stages - 1, reals_shapes, opt)
        print(f'第{idx}张图片生成、修复开始...')
        with torch.no_grad():
            sample = netG(noise, reals_shapes, noise_amp, True, condition_pyramid) # post-GAN需要传递5个参数
        sample = functions.tackle_data(sample)

        functions.save_generate_image_as_txt(sample, idx, dir2save_parent)
        # 保存png或pdf
        functions.save_generate_image_as_png_or_pdf(sample, idx, dir2save)
        print(f'第{idx}张图片生成、修复结束！')
        sample_conditioned.append(sample)

    return sample_conditioned

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='with GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='generated random samples', default=10)
    parser.add_argument('--naive_img', help='naive input image (harmonization or editing)', default='')

    # 参数加载文件夹
    opt = parser.parse_args(['--model_dir', './params_generators/'])  # add the image name!
    _gpu = opt.gpu
    _naive_img = opt.naive_img
    __model_dir = opt.model_dir

    # 推断阶段，加载模型的opt参数
    opt = functions.load_config(opt=opt)
    opt.gpu = _gpu
    opt.naive_img = _naive_img
    opt.model_dir = __model_dir
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    # 创建推断文件夹
    dir2save = os.path.join(opt.model_dir, "Evaluation")
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # 加载最细stage的生成器、固定噪声列表、真实数据金字塔、噪声权重列表
    netG = torch.load('%s/G.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
    fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
    reals = torch.load('%s/reals.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))

    # 读取原始图像数据
    real = functions.read_image(opt)
    print(f'原始数据的类型为：{type(real)}和形状：{real.shape},其中-1有：{torch.sum(real == -1).item()}个, 1有：{torch.sum(real == 1).item()}个')

    # 自动寻找训练阶段数、每个尺寸的大小
    print('最小尺寸的理论范围：', ((real.shape[2]/8, real.shape[3]/7)))
    N, sizes = functions.calculate_singan_stages_and_sizes(real.shape[2], real.shape[3])
    opt.train_stages = N
    print('阶段数N：', N)

    # 对real进行下采样得到图像金字塔
    real_pyramid = functions.downsample_tensor(real, sizes)
    print('真实数据每个阶段的尺寸:')
    for i, real in enumerate(real_pyramid, 1):
        print(f'阶段数{i}：real的形状为{real.shape}')

    # 随机生成条件数据
    condition, condition_dict, location, well = functions.generate_condition_samples(real_pyramid[-1], 15, 3, 44)
    print(f'条件数据的类型为：{type(condition)}和形状：{condition.shape},其中-1有：{torch.sum(condition == -1).item()}个, 1有：{torch.sum(condition == 1).item()}个, 0(背景)有：{torch.sum(condition == 0).item()}个')

    # 展示、保存TI与条件点叠加图
    functions.plot_tensors(real_pyramid[-1], condition, file_format="png")

    # 对condition进行下采样得到图像金字塔
    condition_pyramid = functions.downsample_tensor(condition, sizes)

    print('真实条件每个阶段的尺寸:')
    for i, condition in enumerate(condition_pyramid, 1):
        print(f'第{i}阶段的条件数据的类型为：{type(condition)}和形状：{condition.shape},其中-1有：{torch.sum(condition == -1).item()}个, 1有：{torch.sum(condition == 1).item()}个, 0(背景)有：{torch.sum(condition == 0).item()}个')
    reals_shapes = [r.shape for r in real_pyramid]

    # 生成推断
    with torch.no_grad():
        realizations = generate_samples(netG, reals_shapes, noise_amp, condition_pyramid, n=200)

    # ***************************************************画图***************************************************
    for i, gen in enumerate(realizations, 1):
        print(f'第{i}个生成数据的类型为：{type(gen)}和形状：{gen.shape},其中-1有：{torch.sum(gen == -1).item()}个, 1有：{torch.sum(gen == 1).item()}个, 0(背景)有：{torch.sum(gen == -2).item()}个')


    # 再写一个函数保存每一个单张的图片（realizations, condition_dict）
    functions.display_images_with_scatter_and_save(realizations, condition_dict, 'png')
