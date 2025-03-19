import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# import imresize

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def training(x, block):
    # 下采样过程
    x = block[0](x)
    x = block[1](x)
    x = block[2](x)
    x1 = x

    x = block[3](x)
    x = block[4](x)
    x = block[5](x)
    x2 = x

    x = block[6](x)
    x = block[7](x)
    x = block[8](x)
    x3 = x

    x = block[9](x)
    x = block[10](x)
    x = block[11](x)
    x4 = x

    x = block[12](x)

    # 上采样过程
    x = block[13](x)
    x = block[14](x)
    x = block[15](x)
    x = torch.concat([x, x4], dim=1)

    x = block[16](x)
    x = block[17](x)
    x = block[18](x)
    x = torch.concat([x, x3], dim=1)

    x = block[19](x)
    x = block[20](x)
    x = block[21](x)
    x = torch.concat([x, x2], dim=1)

    x = block[22](x)
    x = block[23](x)
    x = block[24](x)
    x = torch.concat([x, x1], dim=1)

    x = block[25](x)
    x = block[26](x)

    return x

def upsample(x, size):
    x_up = nn.functional.interpolate(input=x, size=size, mode='bilinear', align_corners=True)  # mode为bicubic
    return x_up

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)

# 混合注意力模块
class MixedAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(MixedAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力模块
        x_ca = self.channel_attention(x) * x
        # 空间注意力模块
        x_sa = self.spatial_attention(x_ca) * x_ca
        return x_sa

# 多头注意力机制模块
class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads= 2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Split the channels for multi-head attention
        channels_per_head = C // self.num_heads

        # Apply 1x1 convolutions to get query, key, and value
        query = self.query_conv(x).view(batch_size, self.num_heads, channels_per_head, width * height)
        key = self.key_conv(x).view(batch_size, self.num_heads, channels_per_head, width * height)
        value = self.value_conv(x).view(batch_size, self.num_heads, channels_per_head, width * height)

        # Transpose dimensions for batch matrix multiplication
        query = query.permute(0, 1, 3, 2).contiguous().view(batch_size * self.num_heads, width * height,
                                                            channels_per_head)
        key = key.permute(0, 1, 2, 3).contiguous().view(batch_size * self.num_heads, channels_per_head, width * height)

        # Attention score computation
        attention = torch.bmm(query, key)  # Shape: [batch_size * num_heads, width * height, width * height]
        attention = self.softmax(attention)  # Normalize attention scores

        # Value projection and attention application
        value = value.permute(0, 1, 3, 2).contiguous().view(batch_size * self.num_heads, width * height,
                                                            channels_per_head)
        out = torch.bmm(attention, value)  # Shape: [batch_size * num_heads, width * height, channels_per_head]

        # Reshape output back to [batch_size, C, width, height]
        out = out.view(batch_size, self.num_heads, width * height, channels_per_head).permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, C, width, height)

        # Residual connection
        return self.gamma * out + x

# 生成器定义
class Generator(nn.Module):

    def __init__(self, in_channels, opt, dim=32):
        super(Generator, self).__init__()
        self.opt = opt
        self.body = nn.ModuleList([])

        # 尝试Unet、卷积核大小
        _Unet = nn.Sequential(

            nn.Conv2d(in_channels, dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(dim, 2 * dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * dim, 4 * dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * dim, 8 * dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU(0.2),

            # 瓶颈
            nn.Conv2d(8 * dim, 16 * dim, kernel_size=4, stride=1, padding=1),

            # 注意力机制等技术
            # MixedAttention(16 * dim),
            # MultiHeadAttention(16 * dim),

            nn.ConvTranspose2d(16 * dim, 8 * dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16 * dim, 4 * dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(8 * dim, 2 * dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(4 * dim, dim, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(2 * dim, 1, kernel_size=4, stride=1, padding=1),
            nn.Tanh(),
        )

        self.body.append(_Unet)

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp, Evalution=False, conditions=[]):

        # 每一个stage的第一个block
        x_prev_out = training(noise[0], self.body[0])

        # 每一个stage的中间blocks
        for idx, block in enumerate(self.body[1:], 1):

            if idx < len(self.body[1:]):
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]]) # 上采样结果

                x_prev_out_2 = x_prev_out_1 + noise[idx] * noise_amp[idx] # 上采样结果 + 噪声

                x_prev = training(x_prev_out_2, block) # 当前stage的“输出”

                x_prev_out = x_prev + x_prev_out_1 # 当前stage的“输出” + 上采样结果

                # 修正条件数据
                if Evalution:
                    for epoch in range(self.opt.post_niter):
                        mask = (conditions[idx].to(self.opt.device) == 1) & (conditions[idx].to(self.opt.device) != x_prev_out) # 先找con中非零的位置， 再找与con不等的输出位置
                        if torch.all(~mask):
                            break
                        else:
                            x_prev_out[mask] = conditions[idx][mask].to(self.opt.device)
                            x_prev_out_1 = x_prev_out
                            x_prev_out_2 = x_prev_out_1 + noise[idx] * noise_amp[idx]  # 替换后的输出 + 噪声
                            x_prev = training(x_prev_out_2, block)  # 替换后的输出的输出
                            x_prev_out = x_prev + x_prev_out_1
            # 每一个stage的最后一个blocks
            else:
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])

                x_prev_out_2 = x_prev_out_1 + noise[idx] * noise_amp[idx]

                x_prev_out = training(x_prev_out_2, block) + x_prev_out_1

                # 修正条件数据
                if Evalution:
                    for epoch in range(self.opt.post_niter):
                        mask = (conditions[idx].to(self.opt.device) == 1) & (conditions[idx].to(self.opt.device) != x_prev_out)  # 先找con中非零的位置， 再找与con不等的输出位置
                        if torch.all(~mask):
                            break
                        else:
                            x_prev_out[mask] = conditions[idx][mask].to(self.opt.device)
                            x_prev_out_1 = x_prev_out
                            x_prev_out_2 = x_prev_out_1 + noise[idx] * noise_amp[idx]  # 替换后的输出 + 噪声
                            x_prev = training(x_prev_out_2, block)  # 替换后的输出的输出
                            x_prev_out = x_prev + x_prev_out_1
        return x_prev_out

# 判别器定义
class Discriminator(nn.Module):

    def __init__(self, in_channels, dim=32):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(dim, 2 * dim, kernel_size=4, stride=1, padding=1),
            # nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(2 * dim, 4 * dim, kernel_size=4, stride=1, padding=1),
            # nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * dim, 8 * dim, kernel_size=4, stride=1, padding=1),
            # nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8 * dim, in_channels, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        output = self.block(x)
        return output

