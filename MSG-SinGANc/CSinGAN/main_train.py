import os
import torch

from config import get_arguments
import functions
from training_generation import training


if __name__ == '__main__':

    # 创建参数对象
    parser = get_arguments()
    # 添加参数：输入图像数据、gpu
    parser.add_argument('--input_name', help='input image name for training', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation', help='train mode only generations')
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.1)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=7)

    # 输入数据实例化
    opt = parser.parse_args(['--input_name', './TI/Strebelle_250_250.txt'])

    # 判断模型是否存在！
    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

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

    # 配置模型opt的设备：cpu或gpu
    opt = functions.post_config(opt)

    # 配置模型模型运行的设备：gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    # 创建模型保存的根目录：TrainedModels\Strebelle_250_250\test_for_concurrent_multistage_UNet_GAN
    dir2save = functions.generate_dir2save(opt)

    # 判断模型是否存在！
    if os.path.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # 保存模型训练的参数文件txt！
    with open(os.path.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))

    # 打印当前的文件目录
    print("Training model ({})".format(dir2save))

    # 开始训练
    training(opt, real_pyramid)





