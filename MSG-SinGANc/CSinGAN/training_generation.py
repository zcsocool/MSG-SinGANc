import os
import torch
import torch.nn as nn
import torch.optim as optim

import functions
import U_Net_2D

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_G(opt):
    netG = U_Net_2D.Generator(1, opt).to(opt.device)
    netG.apply(U_Net_2D.weights_init)
    return netG

def init_D(opt):
    netD = U_Net_2D.Discriminator(1).to(opt.device)
    netD.apply(U_Net_2D.weights_init)
    return netD

def generate_samples(netG, opt, stage, noise_amp, reals, n = 5):

    # 生成保存5个fake的文件夹
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, stage)
    reals_shapes = [r.shape for r in reals]
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # 无梯度推断
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(stage, reals_shapes, opt)
            sample = netG(noise, reals_shapes, noise_amp)
            # 将fake转化为二值的-1和1
            sample = functions.tackle_data(sample)
            # 保存txt
            functions.save_generate_image_as_txt(sample, idx, dir2save)
            # 保存png或pdf
            functions.save_generate_image_as_png_or_pdf(sample, idx, dir2save)

def training(opt, real_pyramid):

    # 创建渐进的：固定noise列表、noise权重列表、生成器架构
    fixed_noise = []
    noise_amp = []
    generator = init_G(opt)

    # 循环训练阶段stage
    for stage in range(opt.train_stages):

        # 初始化当前阶段stage的判别器, 注意生成器初始化应该在循环外面，因为generator是渐进增长的，而不是每一个stage重新初始化！！！
        discriminator = init_D(opt)

        # 生成保留网络参数的文件夹
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, stage)
        try:
            os.makedirs(opt.outf)
        except OSError:
            print(OSError)
            pass

        # 同时保存当前尺度的real图像txt和png格式
        functions.save_image_as_txt(real_pyramid[stage], opt, stage)
        functions.save_image_as_file(real_pyramid[stage], opt, stage)

        # 如果stage>1, 网络参数初始化为stage-1的参数
        if stage > 0:
            generator.init_next_stage()
            discriminator.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, stage - 1)))

        # 提取当前stage的real图像等
        real = real_pyramid[stage].to(opt.device)
        real_shapes = [real.shape for real in real_pyramid]
        alpha = opt.alpha

        # **************************************固定噪声、噪声权重计算方式**************************************

        # 随机生成每个stage中固定不变的noise, 用于计算噪声权重与重构误差
        if stage == 0:
            z_opt = real_pyramid[0].to(opt.device)
        else:
            z_opt = functions.generate_noise(size=[opt.nc_im, real_shapes[stage][2], real_shapes[stage][3]], device=opt.device)

        fixed_noise.append(z_opt.detach())

        # 计算每个阶段的noise权重
        if stage == 0:
            noise_amp.append(1)
        else:
            noise_amp.append(0)
            # 当前阶段的重构误差只与：最粗糙阶段的real和stage之前的噪声及权重有关，与当前阶段的noise无关
            z_reconstruction = generator.forward(fixed_noise, real_shapes, noise_amp)
            criterion = nn.MSELoss()
            rec_loss = criterion(z_reconstruction, real)
            RMSE = torch.sqrt(rec_loss).detach()
            _noise_amp = opt.noise_amp_init * RMSE
            noise_amp[-1] = _noise_amp

        # **************************************渐进生成器的优化参数与方式**************************************
        # 初始化生成器G、判别器D的优化器与调度器
        optimizerD = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        # 渐进生成器中最后train_depth个block中的参数不更新
        for block in generator.body[:-opt.train_depth]:
            for param in block.parameters():
                param.requires_grad = False

        # 对渐进生成器中最后train_depth个block设置不同的学习率，学习率与训练阶段stage有关
        parameter_list = [
            {"params": block.parameters(),
             "lr": opt.lr_g * (opt.lr_scale ** (len(generator.body[-opt.train_depth:]) - 1 - idx))}
            for idx, block in enumerate(generator.body[-opt.train_depth:], 0)]

        optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

        # 当模型的迭代进行一半时，学习率乘以0.1
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.5 * opt.niter],
                                                          gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.5 * opt.niter],
                                                          gamma=opt.gamma)

        print('stage: [{}/{}]'.format(stage, opt.train_stages-1))


        # **************************************当前stage训练开始**************************************
        for iter in range(opt.niter):

            # 每轮训练的金字塔噪声
            noise = functions.sample_random_noise(stage, real_shapes, opt)
            
            # 判别器更新
            discriminator.zero_grad()
            real_out = discriminator(real)
            errD_real = -real_out.mean()

            fake = generator.forward(noise, real_shapes, noise_amp)
            fake_out = discriminator(fake.detach())
            errD_fake = fake_out.mean()
            
            gradient_penalty = functions.calc_gradient_penalty(discriminator, real, fake, opt.lambda_grad, opt.device)

            errD_total = errD_fake + errD_real + gradient_penalty
            errD_total.backward(retain_graph=True)
            optimizerD.step()
            
            # 生成器更新
            fake = generator(noise, real_shapes, noise_amp)
            fake_out = discriminator(fake)
            errG = -fake_out.mean()
            
            if alpha != 0:
                loss = nn.MSELoss()
                rec = generator.forward(fixed_noise, real_shapes, noise_amp)
                rec_loss = alpha * loss(rec, real)
            else:
                rec_loss = 0
                
            generator.zero_grad()
            errG_total = errG + rec_loss
            errG_total.backward(retain_graph=True)
            optimizerG.step()
            
            # 打印各种损失
            if iter % 100 == 0 or iter + 1 == opt.niter:
                print('errD_real_loss: %.4f' % errD_real.item())
                print('errD_fake_loss: %.4f' % errD_fake.item())
                print('gradient_penalty_loss: %.4f' % gradient_penalty.item())
                print('generator_loss: %.4f' % errG.item())
                print('reconstruction_loss: %.4f' % rec_loss.item())
                print('\n')
            # 当前stage迭代完成后， 生成5个fake样本
            if iter + 1 == opt.niter:
                generate_samples(generator, opt, stage, noise_amp, real_pyramid)

            schedulerD.step()
            schedulerG.step()

        # ******************************************模型与参数保存******************************************

        # 保存当前stage训练好的生成器G、判别器D、固定噪声
        functions.save_networks(generator, discriminator, z_opt, opt)
        # 保存最细尺度stage的生成器G、固定噪声列表、真实图像金字塔、噪声权重列表
        if stage == opt.train_stages-1:
            torch.save(fixed_noise, '%s/fixed_noise.pth' % opt.out_)
            torch.save(generator, '%s/G.pth' % opt.out_)
            torch.save(real_pyramid, '%s/reals.pth' % opt.out_)
            torch.save(noise_amp, '%s/noise_amp.pth' % opt.out_)

        del discriminator  # 释放内存
