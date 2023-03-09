import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        # 输入通道，输出通道，kernel_size, stride, padding
        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        #输入x [B,C,H,W]
        x = self.conv1(self.conv0(x)) #[B,8,H,W]
        x = self.conv4(self.conv3(self.conv2(x)))#[B,16,H/2,W/2]
        x = self.feature(self.conv6(self.conv5(x)))#[B,32,H/4,W/4]
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        # 特征体的大小不变，每一次3D卷积的结果都会融合不同的通道C上的那些特征体，这样融合8次，输出的就是通道为8的特征体
        self.conv0 = ConvBnReLU3D(32, 8)  

        # stride=2，特征体大小/2, 通道数*2
        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)  # 特征体的大小不变，通道不变，继续融合不同通道

        # stride=2，特征体大小/2, 通道数*2
        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)  # 特征体的大小不变，通道不变，继续融合不同通道

        # stride=2，特征体大小/2, 通道数*2
        self.conv5 = ConvBnReLU3D(32, 64, stride=2)  
        self.conv6 = ConvBnReLU3D(64, 64)  # 特征体的大小不变，通道不变，继续融合不同通道

        self.conv7 = nn.Sequential(
            #; 转置卷积，把特征体的大小放大
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)  # 最终把8个通道只卷积一次，得到一个1通道的输出，变成一个代价体

    def forward(self, x):
        #输入：[B,32,D,H,W]
        conv0 = self.conv0(x) #[B,8,D,H,W]
        conv2 = self.conv2(self.conv1(conv0))#[B,16,D/2,H/2,W/2]
        conv4 = self.conv4(self.conv3(conv2))#[B,32,D/4,H/4,W/4]
        x = self.conv6(self.conv5(conv4))#[B,64,D/8,H/8,W/8]

        x = conv4 + self.conv7(x)#[B,32,D/4,H/4,W/4]
        x = conv2 + self.conv9(x)#[B,16,D/2,H/2,W/2]
        x = conv0 + self.conv11(x)#[B,8,D,H,W]
        x = self.prob(x)#[B,1,D,H,W]
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
      
        concat = F.cat((img, depth_init), dim=1)#[B,4,H,W]
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))#[B,1,H,W]
        depth_refined = depth_init + depth_residual#[B,1,H,W]
        return depth_refined


class RefineNetFixed(nn.Module):
    """_summary_
        修正的最后深度图的refine，上面的代码是先concat再归一化，顺序错了！
    """
    def __init__(self):
        super(RefineNetFixed, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # 深度图归一化到 [0,1]
        depth_init_Norm = (depth_init - depth_min.view(batch_size, 1, 1, 1)) / (
            depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)
        )
        concat = F.cat((img, depth_init_Norm), dim=1)#[B,4,H,W]
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))#[B,1,H,W]
        depth_refined = depth_init_Norm + depth_residual#[B,1,H,W]
        #深度图还原到原来的范围
        depth_refined = depth_refined * (depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1)
        
        return depth_refined

class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values):
        """_summary_   总结在前向传播中的操作（也就是整个网络的操作）：
        1.对输入图像提取特征：对输入的[B,3,H,W]大小的图像，经过FeatureNet提取特征，得到[B,32,H/4,W/4]的Feature map
          这里注意B是批量维度，论文中有利用了N张共视图，这里代码中就是使用for循环对N张图的每一个Batch都输入Feature Net提取，
          也就是说论文中花了三个Feature Net然后说他们权重共享，实际上有点误导，真实的操作就是只有一个Feature Net，然后N张
          共视图都用这个FeatureNet提取特征。这个很正常，因为后面就是要进行特征匹配，自然就要用相同的网络来提取特征
        2.对提取的Feature map进行warp：注意这里用的是深度估计中的反向warp，因为这样是differentiable的。具体来说操作就是
          要用src帧来重构ref帧，也就是从ref帧的[H/4,W/4]的每一个像素位置出发，利用相机内外参把他投影到其他的src帧上，然后
          利用双线性插值取src帧上周围4个点的像素平均值来作为重构的ref帧的像素值。但是注意这里feature map都是32维的，其实也
          很简单，就是在每一个channel上进行像素值平均就可以了，因为他们的在[H/4,W/4]的像素维度上坐标都是一样的。
          这里warp之后，N张图片(包括ref图片，它自己已经就在ref帧的视角下了，所以直接用它的Feature map即可，无需warp)都会
          warp到ref帧上。但是这里注意warp之后的维度是多少？上面利用内外参投影的时候忽略了的一件事是，从ref的像素反投影到ref
          相机系的时候，是需要点的深度的。而MVSNet用的是plane sweeping，也就是假设了192个深度值。这样对于32channel的特征图，
          每个通道利用D个深度值，都会得到warp到ref帧上的D个特征图，所以对于一张src的特征图，最后warp完成后得到的特征图就是
          [B, C, D, H/4, W/4]这个维度的。注意B是batch_size，这个在分析的时候不需要考虑，计算的时候在最前面一直都带着就行。
          C是通道数，就是32； D是设定的深度的个数，192。也就是说，对于每个通道都生成一个[D,H/4,W/4]的立方体，然后32个通道就有
          32个这样的立方体。而且注意这里立方体都是一样大的，所以论文中画的那个金字塔的图完全就是在误导人！最后N张图片，就得到了
          N个[B, C, D, H/4, W/4]的 feature volume！注意这里的N不能放到维度里面，而是for循环累加的。
        3.计算cost volume：想一下上面的feature volume代表了什么？其实就是代表了对每一个通道的特征，如果按照假设的深度进行
          warp，得到的在ref帧上的特征。如果假设的深度是正确的，那么最后得到的N张图片的warp结果的特征图就是一样的！所以说这里就要
          衡量这些特征图到底有多不一样。下面代码中用的就是Σvi^2/N-(Σvi/N)^2，也就是把N个[B, C, D, H/4, W/4]的 feature volume
          在相同位置的像素值，算平方/N，减去(/N)^2。这里类似完全平方公式，比如(a^2+b^2)/2 - ((a+b)/2)^2 = (a-b)^2/4,
          这里就出现了a-b，也就是衡量了feature volume的差异。
        4.cost volume正则化：上面的操作，把N个[B, C, D, H/4, W/4]的 feature volume，变成了一个[B, C, D, H/4, W/4]的 
          cost volume，但是这还不够，因为很明显通道维度C还存在，我们最后想要的是[B,D,H/4,W/4]的cost volume，也就是对于每一个
          假设的深度D，都有一张代价图，这样D张代价图就生成一个[D,H/4,W/4]的立方体，这个立方体的D这个维度，就代表了每一个像素使用
          当前假设的深度，得到的代价是多少(也就是和其他src帧有多么的不像)。所以这里cost volume的正则化就是对[B, C, D, H/4, W/4] 
          的cost volume进行3D卷积，融合不同通道C、同一通道下的[D,H/4,W/4]的立方体之间的代价值。对于3D卷积，其实就是把输入的最后
          3个维度看成卷积的最小单元，也就是使用一个3D的卷积核，在[D,H/4,W/4]的立方体中沿着xyz三个轴都滑动，这样卷积之后就得到了一个
          [D', H'/4, W'/4]的新的立方体，但是这并不是一次卷积的结果，因为还有通道维度，我们还要继续在其他通道上这么操作，再得到另外
          C-1个[D', H'/4, W'/4]的立方体，最后把这C个立方体加起来，才得到一次3D卷积的输出，它还是一个[D', H'/4, W'/4]的立方体。
          那么进行C'次这样的3D卷积，就得到了[C', D', H'/4, W'/4]的输出结果。再加上batch, 就是[B, C', D', H'/4, W'/4]。但是
          最后我们要把通道这个维度拿掉，就是只保留最后不同深度D造成的代价体，所以在内部进行多次3D卷积的最后一次，会让输出的维度为
          [B, 1, D', H'/4, W'/4]，当然为了保证输出和设置匹配，这里卷积的最后D'、H'、W'和D、H、W都是相等的。
        5.回归深度图：上面得到的[B, D, H/4, W/4]就是真的代价体了，最后的[D,H/4,W/4]的D维度上的每一层，反映了如果使用D这个深度，
          对src图像进行warp得到的特征图和ref图像的特征图有多不像。但是我们还要回归出一个深度图，所以这里就把代价体沿着D这个维度
          进行softmax转成概率，概率值越大表示某个像素取这个深度值的可能性越大。这里好像有点问题，按理说代价值越大应该概率越小啊？
          为什么这里反而概率越大呢？目前我感觉是这样：因为上面我们把cost volume进行为了正则化，这个过程中有很多的3D卷积，这里面
          是有学习的参数的，也就是说，在3D卷积之前，得到的确实是cost volume，它表示了代价值(因为我们算的是Σvi^2/N-(Σvi/N)^2)，
          但是这里我有3D卷积之后，我就可以利用3D卷积的参数学习，把代价值转成概率值，比如学的权重都是负数，代价值越高，反而输出越小，
          这样不就是把代价值经过3D卷积变成概率值了吗？所以说严格来说，我们第4步正则化，输入的是cost volume，而输出的是prob volume,
          即输入代价体，输出是概率体。有了概率体之后，然后就用论文中的说法，使用概率体和深度值求期望回归出深度，而不是使用经典方法
          中那种取最大概率的深度值。

        """
        # Step 0 拆分N个图像BxNxCxHxW ->N个BxCxHxW
        # torch.unbind()移除指定维后，返回一个 元组 ，包含了沿着指定维切片后的各个切片
        imgs = torch.unbind(imgs, 1)
        # proj_matrices就是每一帧相机的投影矩阵，内参和外参合在一起，就是4x4的投影矩阵
        proj_matrices = torch.unbind(proj_matrices, 1)#同上Nx[B,4,4]
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]

        print(f"H = {img_height}, W = {img_width}")   # 结果： H = 512, W = 640

        num_depth = depth_values.shape[1] # 划分的深度值范围，这里应该是192
        num_views = len(imgs)  # imgs是一个元祖，这里统计元组的长度实际就是计算一次共视计算中的图片总数N

        # Step 1. feature extraction
        # in: images; out: 32-channel feature maps
        # 第一步：特征提取 输入是Bx3xHxW 输出是[B,32,H/4,W/4]
        features = [self.feature(img) for img in imgs]  # 这个for是在for循环N个视图，每个视图都是有一个batch的
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # Step 2. differentiable homograph, build cost volume
        # 第二步：可微分单应性warp构建代价体
        #ref_feature [B,32,H/4,W/4] 在2处增加一个维度，并把这个维度增广到D(num_depth)
        #; 注意这里先计算ref的feature volume，因为它其实不用进行warp，直接就是它自己经过featurenet得到的特征体
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1) #[B,32,Ndepth,H/4,W/4]
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2 # 平方操作 [B,32,D,H/4,W/4]
        del ref_volume

        # 注意这里的for就是在for循环N-1个source图像，每一次操作，都把src图像warp到ref图像上
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            # 将src的feature warp到ref的num_depth个深度平面上
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)#[B,32,Ndepth,H/4,W/4]
            if self.training:  #; 这个training属性应该就是nn.Module父类的属性了
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                # 求和 计算Σvi 和Σvi^2
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            #; 注意这里为什么要del，因为python中全都是引用，上面homo_warping返回的对象不同，就会开辟不同的内存
            #; 建立指向返回的对象的新的引用，这样显存占用就越来越高
            del warped_volume  

        # aggregate multiple feature volumes by variance
        # 聚合：将N个feature volume ->cost volume =Σvi^2/N-(Σvi/N)^2
        # B x 32 x num_depth x (H/4) x (W/4), [B,32, Ndepth, H/4, W/4]
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2)) 

        # Step 3. cost volume regularization
        # 第三步：代价体正则化
        #; 注意经过这一步之后，把32个通道进行了融合，从而变成了一个通道
        cost_reg = self.cost_regularization(volume_variance)#[B, 1, Ndepth, H/4, W/4]
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)  #[B, Ndepth, H/4, W/4]
        prob_volume = F.softmax(cost_reg, dim=1)  #[B, Ndepth, H/4, W/4]  
        #回归出深度图 
        depth = depth_regression(prob_volume, depth_values=depth_values)  # [B, H/4, W/4]

        print("depth: ", depth.shape)   # 输出： [Batch, 128, 160]

        with torch.no_grad():
            # photometric confidence
            # 置信度求解方式是根据我们计算的最优深度值对应的那层加上上下3层共四层对应的prob_volume值相加即为要计算的置信度
            #取每个prob层上下共四层求和
            #? 1.疑问：这个地方pad怎么有6维？unsqueeze之后不才5维吗？
            #;  F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2))的结果维度：[B, 1, 195, 128, 160]， [B, 192, 128, 160]
            #!  上面疑问解答：注意这个F.pad湖面传入的参数看起来还比较反人类，pad从前向后是对低维向高纬扩充的，而不是顺序扩充的。
            #!    即前两个00扩充[B, 1, 192, 128, 160]的最后一维，也就是160不变；中间两个00扩种倒数第二个维度，即128不变。
            #!    最后的12扩充192这个维度，前+1后+2，因此变成195维度。其他参数没给，所以剩下的B和1两个维度不会进行任何改变
            #!    关于F.pad的一个非常好的讲解：https://blog.csdn.net/jorg_zhao/article/details/105295686
            #; 2.F.avg_pool3d函数：https://blog.csdn.net/qq_41512004/article/details/104700814
            #;   这里后面输入第一个参数(4,1,1)是kernel_size，这个函数的作用就是对输入的(B,C,D,H,W)的数据，只对最后三个维度
            #;   进行pool操作，然后输出是(B,C,D',H',W')，最后三个维度每一个维度的计算公式都是Out = (In+2*padding-kernel)/stride + 1
            #;   所以这里输入[B,1,195,128,160],输出(195+0-4)/1+1=192, (128+0-1)/1+1=128, 160同理。即最后是[B,1,192,128,160]
            #!   注意：这里也引出了为什么上面要对prob_volume进行扩充维度：一是为了满足这里[B,C,D,H,W]的要求，二是为了这里pool之后输出大小
            #!        和原来保持不变
            #; 3.对上面avg_pool3d的结果在进行squeeze(1)，最后得到[B,192,128,160]，然后*4，实际上就得到了沿着通道D的维度上下4层的概率和
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            # prob_volume_sum4： [B, D, H/4/, W/4]
            print("prob_volume_sum4: ", prob_volume_sum4.shape)  # 输出：[Batch, 192, 128, 160]
            
            # 计算回归的深度值在哪一层，这个得到的结果是[B, Ndepth, H/4, W/4]，每个像素的值表示这个像素回归出的深度属于哪一层
            #; 注意这个回归深度值在哪一层的时候也挺有意思的，这样算层数的索引是可以的。因为上面算期望的深度就是这么算的，自然
            #;   算期望的索引也可以这么算。然后注意这里面实际上有一个广播操作，因为传入的depth_values没有B这个维度，是在
            #;   里面进行乘法运算的时候自动在这个维度上进行了扩充的广播运算
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            
            print("depth_index: ", depth_index.shape)  # 输出： [Batch, 128, 160]

            #取对应的置信度值
            #; gather函数讲解：https://blog.csdn.net/iteapoy/article/details/106203954
            # 输入：prob_volume_sum4 = [Batch, 192, 128, 160]， depth_index.unsqueeze(1) = [Batch, 1, 128, 160]
            # 注意这里看了上面gather函数的讲解之后也不是太复杂了，然后还有知乎上的一个回答，就是首先确定gather的输出
            # 第3个index参数的大小是一样的，然后第2个参数表示维度，这个是按照第一个维度，也就是192这个维度根据后面两个维度
            # 对应位置的数字来选择相关的值。比如后面[128,160]代表[H/4,W/4]的图像，那么左上角(0,0)位置表示的值（假设是5），
            # 对于prob_volume_sum4的[128,160]维度，同样(0,0)位置，选择192这个维度的第5个维度的数值作为输出。
            # 这样最后的输出维度就是[Batch, 1, 128, 160]，然后最后在squeeze(1)，就得到了[Batch, 128, 160]
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

            print("photometric_confidence: ", photometric_confidence.shape)    # 输出： [Batch, 128, 160]

        # Step 4. depth map refinement
        # 第4步：深度图优化
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            # 这部分refine实现有问题
            #! 1.注意：这个部分讲的时候说是论文中说的，在cat之前，要把上面回归出的初始深度图归一化到[0,1]之间，再和rbg图concat.
            #!   但是这里面的实现是先concat，在归一化了，所以不对。确实应该归一化，因为rbg图也是归一化的，而你的初始深度图有可能
            #!   是各种奇怪的值，远远超过[0,1]的范围，所以归一化之后concat才有意义。
            #! 2.另外，这里对深度图归一化的操作，在一篇论文中看到说DVSO有问题(好像是内窥镜那个？)，就是说它的深度图没有归一化，
            #!   然后还说另一篇论文把这个归一化加上了，解决了这个问题。
            #; 这里的深度图的refine，有点ResNet的味道，就是再学一个残差值
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5   #; mask是数据集中的设置，有的地方是没用的背景，mask<0.5这里就不使用它
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
