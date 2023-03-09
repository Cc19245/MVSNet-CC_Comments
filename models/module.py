import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)  #; inplace=True表示就地操作，也就是对输入的值就地进行relu激活


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    #; 3D卷积，这个博客讲的很少，但是感觉还算清楚：
    # https://zhuanlan.zhihu.com/p/55567098  https://blog.csdn.net/FrontierSetter/article/details/99888787
    # https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/3D_Convolution.html
    #; 1.先看2D卷积：2D卷积假设是(C,H,W)输入，其中C是通道数，不一定就是3，因为后面高维特征通道数很高。
    #;   那么卷积的时候，对于(H,W)的图像，使用一个2D的大小为(h,w)的卷积和进行卷积，得到一个(H', W')的图像。
    #;   然后每一个C的维度上，都是用这样的一个(h,w)的小卷积核进行卷积，这样就会得到C个(H', W')的图像，然后
    #;   把这些图像按照通道C的维度进行相加，最终得到一个最终的(H', W')的图像，这样就得到了输出通道的一个值。
    #;   为了满足输出通道的个数要求，这样重复C'次，就得到C'个(H', W')的图像，再把他们堆叠起来，就得到最终输出(C',H',W')
    #; 2.3D卷积：3D卷积输入是(C,D,H,W)，其中C仍然是特征的通道数，而D变成深度(这个不是点的深度，可以是视频的帧数等)
    #;   也就是此时每个通道C的特征都是(D,H,W)形状的长方体了，是有深度的长方体，而不是2D的平面了。那么自然卷积的时候
    #;   就要使用3D的(d,h,w)大小的立方体卷积核进行卷积，这样得到一个(D', H', W')的立方体。然后为了融合每一个通道的值，
    #;   和2D卷积一样，在每一个C的维度上，都用这样的一个(d,h,w)的小卷积核进行卷积，这样就会得到C个(D', H', W')的输出立方体
    #;   把这些立方体按照通道的维度进行相加，最终得到一个最终的(D', H', W')的立方体输出，这就是输出通道的一个值。
    #;   为了满足输出通道的个数要求，这样重复C'次，得到C'个(D',H',W')的小立方体，然后把他们按照通道维度堆叠起来，
    #;   就得到最终的输出(C', D', H', W')
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    #; 注意这个src_proj和ref_proj就是4x4的形式，但是其中结果是内参和外参的乘积，做了简化放在一起了：
    #; K_3x3,  T_4x4 = [R, t; 0 1]
    #; 一个世界坐标系的坐标点Pw=[x, y, z]，转到相机坐标系下为z*[u, v, 1] = K*(R*Pw+t)
    #; 如果用其次坐标表示，如下:  x                   x
    #;     u        R  t      y      K*R  K*t     y
    #; z * v = K *         *  z  =             *  z
    #;     1        0  1      1       0    1      1
    #; 所以自后这个矩阵中存储的4x4的结果就是 [K*R K*t; 0 1]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    #! 注意：warp的过程是不计算梯度的，为什么？
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))#该投影矩阵是ref投影到世界坐标系下然后再投影到src，即ref投影到src
        rot = proj[:, :3, :3]  # [B,3,3]   这个R是把内参包含在里面的
        trans = proj[:, :3, 3:4]  # [B,3,1]  这个t也是把内参包含在里面的
        # 下面这段实现的是ref图每个像素计算其深度分别为Ndepth个depth值时转换到src相机坐标下的坐标。即[d'u',d'v',d']_src=R[u,v,d]_ref+t
        # torch.meshgrid就是传入x，y的坐标范围，然后生成二维的图像坐标的两个矩阵。注意y和x都是(h, w)维度的矩阵
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        #; 假设h=2, w=3, 那么meshgrid生成的y 和 x分别为：
        #; y:  0 0 0    x:  0 1 2
        #;     1 1 1        0 1 2

        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样
        #! 疑问：暂时记住利用meshgrid创建的就必须用contiguous来让内存分配连续吧
        y, x = y.contiguous(), x.contiguous() 
        y, x = y.view(height * width), x.view(height * width)  # 把二维的x和y都转成一维的，也就是矩阵变向量
        #; 把y和x经过view之后，变成：
        #;   y: 0 0 0 1 1 1
        #;   x: 0 1 2 0 1 2
        #; 可见y和x的对应位置就是图像的一个位置的像素坐标，即[y, x]表示的就是二维图像的（0， 1）两个维度的像素坐标。
        #;   如果此时把y和x再resize成二维图像的话，他们的表示形式就是：
        #;   [(0,0) (0,1) (0,2);
        #;    (1,0) (1,1) (1,2)]

        # print(y.shape)

        # 这里传入三个参数，就是把三个向量都堆叠起来，每个向量都是[H*W]，这样默认在第0维堆叠，得到的就是[3, H*W]
        # 注意这里stack的时候，把x放到了前面，因为图像(x,y)坐标系是x轴横、y轴竖：https://www.cnblogs.com/IllidanStormrage/articles/16284975.html
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        #; 1.上一步stack之后结果(注意到现在的举例都还没有batch的维度
        #;   [[0, 1, 2, 0, 1, 2],
        #;    [0, 0, 0, 1, 1, 1],
        #;    [1, 1, 1, 1, 1, 1]]
        #! 注意这一个步骤的物理意义：就是把ref帧上的像素坐标按照每一列组织成了[u,v,1]的其次坐标形式，列数就是H*W
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

        # [B,3,3] * [B,3,H*W], 注意这里是高纬矩阵相乘，默认保持第0维，然后把后面的维度进行矩阵乘法，
        #  比如保持B维度不变，然后后面运算[3,3] * [3, H*W] = [3, H*W]  https://blog.csdn.net/qsmx666/article/details/105783610/
        #; 1.也就是说，对于高维度的矩阵乘法，本质上都是在做数学上真正的乘法，而其他的维度都是当做batch_size来看(如果
        #;   其他维度有不同的，那么就进行广播，让他们维度相同。所以其他维度要是不同，肯定要有一个是1)
        #; 2.理解这里的物理意义：首先R是包含内参的，虽然这里的xyz是ref帧的[u,v,1]坐标，但是可以认为把R中的K先和它乘，
        #;   变成相机坐标系中的点，然后再乘以rot，就得到旋转ref帧的点到src帧中的坐标 R*P+t中 R*P的部分
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        #; 注意下面这里为什么要乘以深度，也是物理意义上来看。上一步得到 rot_xyz本质上来说都是深度为1(归一化相机平面)的那些点。
        #; 因为相机内参定义是 [u,v,1] = K*[x,y,1]。而如果我们要把点的逆深度加上的话，那么投影的结果应该是
        #;  R*(K^-1*[u,v,1] * depth) + t， 而我们上边算的rot_xyz实际上是R*K^-1*[u,v,1]，还少乘了一个深度
        # 其3个通道指的就是[d'u',d'v',d']_src
        # *前面的结果就是[B, 3, Ndepth, H*W]， depth_values大小是[B, Ndepth]，
        # 然后把它强行调整成[B, 1, Ndepth, 1]，这样才能按照元素操作进行元素级乘法
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) *\
            depth_values.view(batch, 1, num_depth,1)  # [B, 3, Ndepth, H*W]

        #下面计算的是投影到src的像素坐标[u',v'] [d'u',d'v',d']->[u',v'] 
        #; 注意这里trans.view和rot_depth_xyz相加时候的广播操作：沿最后一个1广播是对一张图片上的H*W个点的每个点加上平移
        #;   沿着倒数第2个1广播是对Ndepth张图像中的每一个点加上平移
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

        #; 这里再次注意，上面自己写的那些就是为了方便理解，到这里可以发现，实际上上面的操作前面都有一个K，也就是
        #; 不管是旋转还是平移，计算的结果前面都是加了K的，也就是都是在算像素坐标，但是最后要把他变成其次的[u',v',1]，因此
        #; 这里要除以第3个维度的值
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        
        #; 注意这里为什么要这么操作？实际上是为了后面的grid_sample函数使用的，那个函数要的是[-1,1]之间的值进行采样
        #;  因此这里要把图像的像素坐标变成[-1, 1]之间的比例值
        #! 注意：这里进行了元素索引，所以维度变成[B, Ndepth, H*W]，而不是 [B, 1, Ndepth, H*W]，因为使用元素索引自然有一个维度就没有了
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1

        # proj_x_normalized维度是[B, Ndepth, H*W]， 沿着第3个维度stack，
        #! 注意：stack会扩充维度！默认会按照dim=0扩充，这里指定dim=3，也就是堆叠之后的会在结果的dim=3维度上进行扩充
        # 这里堆叠之后的结果，最后一个维度2表示的就是图像的(x,y)比例系数
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    #根据上面计算的src的像素坐标取其对应的特征向量即得到了warp后的src的特征
    #; 双线性插值在src图像上进行采样， F.grid_sample函数：https://zhuanlan.zhihu.com/p/112030273
    #! 注意两点：
    # 1.虽然上面把像素坐标变成了比例值，但是结果并不一定就在[-1,1]之间，因为肯定有投影之后不在src图像范围内的点，那些点的
    #   比例肯定就<-1或>1，此时就直接使用0来填充结果。
    # 2.这里要把grid.view，应该就是因为函数要求输入就是这样的，可以看上面的知乎链接。也就是grid.view的参数是
    #   [B, H_o, W_o, 2]，这里因为有Ndepth张图，所以只能把Ndepth弄到H_o的这个维度上，保持W_o不变，才能完成插值采样
    #   然后完成插值采样之后下面又使用view把Ndepth这个维度给分出来了
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), 
        mode='bilinear', padding_mode='zeros')  #; 猜测这个结果应该是[B, C, Ndepth*height, W]
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)#[B, C, Ndepth, H, W]

    return warped_src_fea


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    #; 注意这个写法，使用*就是把列表/元祖中的元素全都拿出来变成标量
    depth_values = depth_values.view(*depth_values.shape, 1, 1)#[B,D,1,1]
    #depth=Σd*p
    depth = torch.sum(p * depth_values, 1)  # 沿着dim=1维度求和，就是沿着D通道求和
    return depth


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
