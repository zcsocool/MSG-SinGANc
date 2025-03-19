from math import log, ceil
import torch
import torch.nn as nn
import math
import os
import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
import cv2
# from imresize import imresize
from torchvision.utils import save_image


#  读取图像附加函数（下面三个）
def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)

def move_to_gpu(t):
    if torch.cuda.is_available():
        t = t.to(torch.device('cuda'))
    return t

def np2torch(x, opt):
    x = x[:, :, :, None]
    x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

# 读取输入图像数据并返回
def read_image(opt):
    with open(opt.input_name, 'r') as f:
        data = f.readlines()
    data_size = data[0].strip().split(' ')
    data_size = list(map(eval, data_size))  # 解析数据尺寸
    x = []
    for i in range(3, len(data)):
        value = data[i].strip()  # 去除空格和换行符
        if value:  # 确保不是空行
            try:
                x.append(int(value))
            except ValueError:
                print(f"警告：跳过无法解析的值 -> {value!r}")
    x = np.array(x)
    x = x.reshape((data_size[0], data_size[1], 1))
    x = np2torch(x, opt)
    return x

# 对opt.device、opt.noise_amp_init进行初始化
def post_config(opt):
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.noise_amp_init = opt.noise_amp
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

# 推断阶段，加载模型的opt参数
def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt

# 创建TrainedModels文件夹
def generate_dir2save(opt):
    training_image_name = opt.input_name[:-4].split('/')[-1]
    dir2save = 'TrainedModels/{}/'.format(training_image_name)
    dir2save += 'test_for_concurrent_multistage_UNet_GAN'
    return dir2save

# 保存real为txt
def save_image_as_txt(image, opt, scale_num):
    image_size = list(image.size())
    image = image.detach().cpu().numpy()
    x = image_size[2]
    y = image_size[3]

    image = image.reshape((x, y))
    image = image.reshape((x * y, 1))

    file = open('%s/real_image_with_different_scale_%d.txt' % (opt.outf, scale_num), 'w')
    file.write(str(x) + ' ' + str(y) + ' ' + str(1) + '\n')
    file.write(str(1) + '\n')
    file.write('facies' + '\n')

    for i in range(len(image)):
        temp = str(float(image[i]))
        file.write(temp + '\n')

    file.close()

# 保存real为png或pdf
def save_image_as_file(image, opt, scale_num, format="png"):
    """
    保存图像为 PNG 或 PDF 文件

    参数：
        image (Tensor): 输入图像，形状为 [1, C, H, W]。
        opt (Namespace): 包含输出文件夹路径 `opt.outf` 的对象。
        scale_num (int): 当前的尺度编号，用于文件命名。
        format (str): 保存的文件格式，可以是 "png" 或 "pdf"。
    """
    # 检查文件格式
    if format not in ["png", "pdf"]:
        raise ValueError("Unsupported file format. Please choose 'png' or 'pdf'.")

    # 将 Tensor 转换为 NumPy 数组
    image = image.detach().cpu().numpy()

    # 确保输入的形状为 [1, C, H, W]
    if len(image.shape) != 4 or image.shape[0] != 1:
        raise ValueError("Input image must have shape [1, C, H, W].")

    # 去掉 batch 维度，得到形状为 [C, H, W]
    image = image[0]

    # 如果图像是单通道 (C=1)，则去掉通道维度
    if image.shape[0] == 1:
        image = image[0]

    # 创建输出文件夹
    os.makedirs(opt.outf, exist_ok=True)

    # 文件路径
    file_path = os.path.join(opt.outf, f"real_image_with_different_scale_{scale_num}.{format}")

    # 保存图像
    if format == "png":
        # 使用 Matplotlib 保存 PNG 文件
        plt.imsave(file_path, image, cmap="viridis" if image.ndim == 2 else None)
    elif format == "pdf":
        # 使用 Matplotlib 保存 PDF 文件
        plt.figure(figsize=(8, 8))
        if image.ndim == 2:
            plt.imshow(image, cmap="viridis")
        else:
            # 将通道维度移动到最后
            image = np.transpose(image, (1, 2, 0))
            plt.imshow(image)
        plt.axis("off")
        plt.savefig(file_path, format="pdf", bbox_inches="tight")
        plt.close()

    print(f"Image saved as {file_path}")

# SinGAN的阶段数、各阶段图像的尺寸自动寻优
def calculate_singan_stages_and_sizes(p, q, smin=None, scale_factor=0.75, p_N=250, q_N=250):
    """
    根据 SinGAN 模型计算阶段数和每个阶段的 TI 尺寸。

    参数:
    - p (int): 原始训练图像的高度。
    - q (int): 原始训练图像的宽度。
    - p_N (int): 最细尺度的高度。
    - q_N (int): 最细尺度的宽度。
    - smin (float, optional): 最小尺寸。默认为 None，将使用推荐范围。
    - delta (float): 缩放因子。默认值为 0.75。

    返回:
    - N (int): 阶段数。
    - sizes (list of tuples): 每个阶段的尺寸 (p_n, q_n)。
    """
    # 最大尺寸是原始图像的长宽最大值
    smax = max(p, q)

    # 如果未提供 smin，则根据推荐范围设置
    if smin is None:
        smin = ceil(min(p, q) / 8 + max(p, q) / 7) / 2  # 推荐范围的中间值

    # 根据公式计算阶段数 N
    N = ceil(log(smin / smax) / log(scale_factor) + 1)

    # 计算最小尺度对应的 epsilon
    epsilon = min((max(p_N, q_N) / smax), 1)

    # 计算缩放因子 m
    m = (smin / (epsilon * min(p, q))) ** (1 / (N - 1))

    # 初始化阶段尺寸列表
    sizes = []

    # 计算每个阶段的尺寸
    for n in range(1, N):
        pn = int(round(p * m ** ( (N-1) / log(N) * log(N-n) + 1 )))
        qn = int(round(q * m ** ( (N-1) / log(N) * log(N-n) + 1 )))
        sizes.append((pn, qn))
    sizes.append((p_N, q_N))

    print('最小尺寸smin为：{}, 最大尺寸smax为：{}, 最细尺度长宽为：{}, 原始TI的长宽为: {}'.format(smin, smax, (p_N,q_N), (p, q)))

    return N, sizes

# 根据TI与目标形状,按照"指定方法"实现下采样
def downsample_tensor(image_tensor, target_shapes, method="bilinear"):
    """
    对输入的图像张量进行下采样。

    参数:
    - image_tensor (torch.Tensor): 输入的图像张量，形状为 [1, 1, H, W]。
    - target_shapes (list): 下采样目标形状的列表，例如 [[1, 1, 26, 26], [1, 1, 31, 31], ...]。
    - method (str): 下采样的方法，包括 'maximizing', 'averaging', 'bilinear'。

    返回:
    - downsampled_list (list): 下采样结果的列表，每个元素是下采样后的张量。
    """
    downsampled_list = []
    input_height, input_width = image_tensor.shape[2], image_tensor.shape[3]

    for target_shape in target_shapes:
        # 提取目标高度和宽度
        target_height, target_width = target_shape[0], target_shape[1]

        if method == "bilinear":
            # 使用双线性插值下采样
            downsampled = nn.functional.interpolate(image_tensor, size=(target_height, target_width), mode='bilinear', align_corners=False)

        elif method == "averaging":
            # 使用平均池化下采样
            scale_h = input_height / target_height
            scale_w = input_width / target_width
            kernel_size = (int(scale_h), int(scale_w))
            stride = kernel_size
            # 先使用平均池化近似
            intermediate = nn.functional.avg_pool2d(image_tensor, kernel_size=kernel_size, stride=stride)
            # 插值到精确目标大小
            downsampled = nn.functional.interpolate(intermediate, size=(target_height, target_width), mode='bilinear', align_corners=False)

        elif method == "maximizing":
            # 使用最大池化下采样
            scale_h = input_height / target_height
            scale_w = input_width / target_width
            kernel_size = (int(scale_h), int(scale_w))
            stride = kernel_size
            # 先使用最大池化近似
            intermediate = nn.functional.max_pool2d(image_tensor, kernel_size=kernel_size, stride=stride)
            # 插值到精确目标大小
            downsampled = nn.functional.interpolate(intermediate, size=(target_height, target_width), mode='bilinear', align_corners=False)

        else:
            raise ValueError(f"Unsupported method: {method}. Use 'bilinear', 'averaging', or 'maximizing'.")

        # 检查并确保形状一致
        assert downsampled.shape[2:] == (target_height, target_width), f"Downsampled shape {downsampled.shape[2:]} does not match target shape {(target_height, target_width)}"

        downsampled_list.append(downsampled)

    return downsampled_list

# 随机生成当前尺度噪声的附加函数
def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)

# 利用双线性插值将图像上采样到指定形状
def upsample(x, size):
    x_up = nn.functional.upsample(input=x, size=size, mode='bilinear', align_corners=True)  # mode为bicubic
    return x_up

# 随机生成当前尺度噪声
def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    elif type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    else:
        raise NotImplementedError
    return noise

# 随机生成金字塔噪声
def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        noise.append(
            generate_noise(size=[opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], device=opt.device).detach())
    return noise

# 计算梯度损失
def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# 将生成的fake保存为txt
def save_generate_image_as_txt(image, iter, path):
    image_size = list(image.size())
    image = image.detach().cpu().numpy()
    x = image_size[2]
    y = image_size[3]

    image = image.reshape((x, y))
    image = image.reshape((x * y, 1))

    file = open('%s/generated_image_%d.txt' % (path, iter), 'w')
    file.write(str(x) + ' ' + str(y) + ' ' + str(1) + '\n')
    file.write(str(1) + '\n')
    file.write('facies' + '\n')

    for i in range(len(image)):
        temp = str(float(image[i]))
        file.write(temp + '\n')

    file.close()

# 将生成的fake保存为png或pdf
def save_generate_image_as_png_or_pdf(image, iter, path, save_as='png'):
    """
    将生成的图像保存为 PNG 或 PDF 格式。

    参数:
        image (torch.Tensor): 输入图像张量，形状为 (N, C, H, W)。
        iter (int): 当前迭代步数，用于文件命名。
        path (str): 保存路径。
        save_as (str): 保存格式，支持 'png' 或 'pdf'。
    """
    # 确保保存路径存在
    os.makedirs(path, exist_ok=True)

    # 检查输入张量的维度
    if image.dim() != 4:
        raise ValueError("输入图像张量的形状必须为 [N, C, H, W]。")

    # 获取批量大小
    batch_size = image.size(0)

    for idx in range(batch_size):
        # 提取单张图像
        single_image = image[idx]

        # 将张量从 (C, H, W) 转换为 NumPy 数组 (H, W, C)
        if single_image.size(0) == 1:  # 灰度图
            single_image = single_image.squeeze(0).detach().cpu().numpy()  # (H, W)
            single_image = (single_image * 255).astype('uint8')
            pil_image = Image.fromarray(single_image, mode='L')
        elif single_image.size(0) == 3:  # RGB 图像
            single_image = single_image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)
            single_image = (single_image * 255).astype('uint8')
            pil_image = Image.fromarray(single_image, mode='RGB')
        else:
            raise ValueError("通道数必须为 1（灰度图）或 3（RGB 图像）。")

        # 确定保存路径和文件名
        file_name = f'generated_image_{iter}_batch_{idx}.{save_as}'
        file_path = os.path.join(path, file_name)

        # 保存图像
        if save_as.lower() == 'png':
            pil_image.save(file_path, format='PNG')
        elif save_as.lower() == 'pdf':
            pil_image.save(file_path, format='PDF')
        else:
            raise ValueError("不支持的保存格式！仅支持 'png' 和 'pdf'。")

        print(f"图像已保存为 {file_path}")

# 数据处理：将预测得到的fake二值化为-1或1(根据正、负进行分类)，返回为tensor
def tackle_data(curr_real):
    x = curr_real.size()[2]
    y = curr_real.size()[3]
    curr_real = curr_real.detach().cpu().numpy()

    curr_real = curr_real.reshape((x, y))
    curr_real = curr_real.reshape((x * y, 1))

    for i in range(len(curr_real)):
        if curr_real[i] <= 0:
            curr_real[i] = -1
        else:
            curr_real[i] = 1

    curr_real = curr_real.reshape((x, y))
    curr_real = curr_real.reshape((1, 1, x, y))
    curr_real = torch.from_numpy(curr_real)
    return curr_real

# 保存当前stage的生成器G、判别器D、固定噪声
def save_networks(netG, netDs, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

# *********************************************************条件建模*********************************************************

# 读取指定文件夹中的txt文件，并将其转换为NumPy数组
def read_txt_file(file_path, opt):
    """
    读取txt文件，跳过前三行，并将剩余数据转为NumPy数组。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 跳过前三行
    data_lines = lines[opt.num_skip:]
    # 过滤空行并解析为浮点数
    valid_data = [float(value) for value in data_lines if value.strip()]
    # 将数据重塑为二维数组
    data = np.array(valid_data).reshape((opt.image_height, opt.image_width))
    return data

# 画出原始的图像与条件数据点
def plot_tensors(tensor1, tensor2, file_format = "png"):
    # 将tensor转换为numpy数组
    tensor1 = tensor1.squeeze().cpu().numpy()  # shape: [250, 250]
    tensor2 = tensor2.squeeze().cpu().numpy()  # shape: [250, 250]

    # 创建一个画布
    plt.figure(figsize=(6, 6))

    # 显示tensor1，-1为黑色，1为白色
    plt.imshow(tensor1, cmap='gray', vmin=-1, vmax=1)

    # 对tensor2中的元素进行处理，-1用红点表示，1用蓝点表示
    red_points = np.where(tensor2 == -1)
    blue_points = np.where(tensor2 == 1)

    # 在图上加上红点和蓝点
    plt.scatter(red_points[1], red_points[0], color='red', s=2, label='-1 (red points)')
    plt.scatter(blue_points[1], blue_points[0], color='blue', s=2, label='1 (blue points)')


    # 添加图例
    # plt.legend()

    # 关闭坐标轴
    plt.axis('off')

    # 保存图像
    file_path = f"Train_con_image.{file_format}"
    if file_format == "pdf":
        plt.savefig(file_path, format="pdf")
    else:
        plt.savefig(file_path, format="png")

    # 显示图像
    plt.show()
    # 关闭当前图像
    plt.close()
    print(f"Image has been saved as {file_path}")

# 随机生成条件点并扩展
def generate_scatter_image_expand(image_height, image_width, num_point=10, radius=2):
    height = image_height
    width = image_width
    # 创建白色背景图像
    points = np.zeros((height, width), dtype=np.float32)
    location = np.zeros((height, width), dtype=np.float32)
    well = np.zeros((height, width), dtype=np.float32)

    # 计算黑色和白色点的数量，黑色点是白色点的两倍
    num_black_points = int(num_point * 3 / 5)
    num_white_points = num_point - num_black_points  # 剩下的为白色点

    # 生成随机点位置
    black_points = np.random.randint(0, min(height - radius - 1, width - radius - 1), size=(num_black_points, 2))
    white_points = np.random.randint(0, min(height - radius - 1, width - radius - 1), size=(num_white_points, 2))

    # 绘制黑色散点 (-1), 对应结果中的红点
    for x, y in black_points:
        points[x, y] = -1
        # 生成圆形范围的坐标
        rr, cc = disk((x, y), radius, shape=points.shape)
        points[rr, cc] = -1
        coordinates = np.column_stack((rr, cc))
        black_points = np.vstack((black_points, coordinates))
        location[rr, cc] = 1

    # 绘制白色散点 (1), 对应结果中的蓝点
    for x, y in white_points:
        points[x, y] = 1
        # 生成圆形范围的坐标
        rr, cc = disk((x, y), radius, shape=points.shape)
        points[rr, cc] = 1
        well[rr, cc] = 1
        coordinates = np.column_stack((rr, cc))
        white_points = np.vstack((white_points, coordinates))
        location[rr, cc] = 1

    # 将图像转换为 tensor
    tensor_points = torch.tensor(points).unsqueeze(0).unsqueeze(0)
    tensor_location = torch.tensor(location).unsqueeze(0).unsqueeze(0)
    tensor_well = torch.tensor(well).unsqueeze(0).unsqueeze(0)
    # 返回散点图像和坐标,用于画图
    dict_points = {"black": black_points, "white": white_points}

    # 显示图像
    plt.imshow(points, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
    plt.show()

    return tensor_points, dict_points, tensor_location, tensor_well

    # 在原始训练图像的基础上，提取若干条件数据

# 在TI上生成条件点并扩展
def generate_condition_samples(image_tensor, num_samples=10, radius=0, seed=0):
    """
    从给定的图片tensor中随机选取条件样本点，并返回条件数据、位置数据和井数据。
    如果获取的点在原始图像tensor中值为-1，则字典中的键命名为black，否则命名为white。

    参数:
    - image_tensor (torch.Tensor): 输入的图片tensor，形状为 (1, C, H, W)。
    - num_samples (int): 条件样本点的数量。
    - radius (int): 扩充半径，默认为0，表示点的大小。

    返回:
    - condition (torch.Tensor): 生成的条件样本图像。
    - condition_dict (dict): 包含黑色和白色点的坐标字典。
    - location (torch.Tensor): 表示条件样本点位置的图像。
    - well (torch.Tensor): 表示井的位置的图像。
    """
    np.random.seed(seed)

    # 获取输入图像的高度和宽度
    height, width = image_tensor.shape[2], image_tensor.shape[3]

    # 创建空白背景图像和用于存储条件数据的图像
    points = np.zeros((height, width), dtype=np.float32)
    location = np.zeros((height, width), dtype=np.float32)
    well = np.zeros((height, width), dtype=np.float32)

    # 随机选择条件样本点的位置
    sample_points = np.random.randint(0, min(height - radius - 1, width - radius - 1), size=(num_samples, 2))

    # 创建字典存储样本点的坐标
    condition_dict = {"black": [], "white": []}

    # 遍历样本点并绘制条件数据
    for x, y in sample_points:
        # 获取原始tensor中该位置的值
        pixel_value = image_tensor[0, 0, x, y].item()  # 获取单个像素的值
        # 判断像素值，黑色点为-1，白色点为1
        if pixel_value == -1:
            condition_dict["black"].append((x, y))
        else:
            condition_dict["white"].append((x, y))

    # 生成随机点位置
    black_points = np.array(condition_dict["black"])
    white_points = np.array(condition_dict["white"])

    # 绘制黑色散点 (-1), 对应结果中的红点
    for x, y in black_points:
        points[x, y] = -1
        # 生成圆形范围的坐标
        rr, cc = disk((x, y), radius, shape=points.shape)
        points[rr, cc] = -1
        coordinates = np.column_stack((rr, cc))
        black_points = np.vstack((black_points, coordinates))
        location[rr, cc] = 1

    # 绘制白色散点 (1), 对应结果中的蓝点
    for x, y in white_points:
        points[x, y] = 1
        # 生成圆形范围的坐标
        rr, cc = disk((x, y), radius, shape=points.shape)
        points[rr, cc] = 1
        well[rr, cc] = 1
        coordinates = np.column_stack((rr, cc))
        white_points = np.vstack((white_points, coordinates))
        location[rr, cc] = 1

    # 将图像转换为 tensor
    tensor_points = torch.tensor(points).unsqueeze(0).unsqueeze(0)
    tensor_location = torch.tensor(location).unsqueeze(0).unsqueeze(0)
    tensor_well = torch.tensor(well).unsqueeze(0).unsqueeze(0)
    # 返回散点图像和坐标,用于画图
    dict_points = {"black": black_points, "white": white_points}

    return tensor_points, dict_points, tensor_location, tensor_well

# 保存叠加后的图像到‘Result’文件夹中
def display_images_with_scatter_and_save(image_list, condition_dict, file_format="png"):
    """
    显示图像并叠加散点，然后保存为PNG或PDF文件。

    参数:
    - image_list: 包含待显示的图像的列表
    - condition_dict: 包含散点坐标的条件字典，其中包括'black'和'white'散点
    - file_format: 保存文件的格式，支持'png'或'pdf'（默认为'png'）
    """
    # 创建保存结果的文件夹
    output_dir = 'Result'
    os.makedirs(output_dir, exist_ok=True)

    # 遍历图像列表
    for i, img in enumerate(image_list):
        img = img[0].permute(1, 2, 0)
        img = img.numpy()

        # 创建图像并绘制
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap='gray', vmin=-1.0, vmax=1.0)
        ax.axis('off')
        # ax.set_title(f"Realization {i + 1}")

        # 获取当前的散点数据
        scatter_points = condition_dict
        black_points = scatter_points["black"]
        white_points = scatter_points["white"]

        # 绘制散点
        ax.scatter(black_points[:, 1], black_points[:, 0], c='red', s=10, label='Black Points')
        ax.scatter(white_points[:, 1], white_points[:, 0], c='blue', s=10, label='White Points')

        # 设置图例
        # ax.legend()

        # 保存图像
        file_path = os.path.join(output_dir, f"realization_{i + 1}.{file_format}")
        if file_format == "pdf":
            plt.savefig(file_path, format="pdf")
        else:
            plt.savefig(file_path, format="png")

        plt.close(fig)  # 关闭图像，释放内存

    print(f"All images have been saved in the '{output_dir}' folder.")





