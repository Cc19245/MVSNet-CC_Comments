from cv2 import imshow
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath    # 存放数据集的文件夹
        self.listfile = listfile    # 存放train/test数据集名称的文件，也就是工程中的lists/dtu文件夹下的test.txt/train.txt/val.txt
        self.mode = mode  # train/test/val模式
        #; 注意这个就是取的N张图片进行MVSNet的图片个数
        self.nviews = nviews
        self.ndepths = ndepths  # 有多少估计的深度值
        self.interval_scale = interval_scale

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        #; 使用with open xxx as f的方式，不用考虑file.close的问题
        with open(self.listfile) as f:
            # readlines，文件中的所有行都读出来
            scans = f.readlines()  
            # line.rstrip删除右边的指定字符，没有传入参数则默认删除右边的空格
            scans = [line.rstrip() for line in scans]
        
        # print(scans)
        # print("len(scans) = ", len(scans))  # 输出79

        # scans就是数据集中各个文件夹的名称列表，如['scan2', 'scan6', 'scan7']
        for scan in scans:
            #; 注意这pair.txt文件存储的内容格式：
            #; 第1行就一个数字，表示有多少张图片，比如49
            #; 第2行是0，代表下面的数据是第0张图片的匹配数据对
            #; 第3行是上面那行的图片的匹配对，比如10 10 2346.41，表示一共有10张匹配图片，匹配分数最高的图片序号是10，
            #;     得分是2346.41；然后后面还有1 2036.53,就是匹配分数第二高的图片序号是1，得分是2036.53
            #; 后面的所有行都重复第2和3行内容，一直把49张图片都重复完成
            pair_file = "Cameras/pair.txt"  # 找这个文件夹下的Cameras/pair.txt文件
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                #; 第1行，表示一共有多少图片
                num_viewpoint = int(f.readline())  
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    #; 第2行，表示下面的数据是那张图片的匹配数据，也就是当前是那张图片作为ref
                    ref_view = int(f.readline().rstrip())  
                    #; 第3行，表示当前这个ref数据匹配的那些高分的src数据，split是通过空格分割得到列表，然后从[1]开始
                    #;       分割，间隔是2，也就是只要序号，不要得分
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    #! 疑问：这里又循环0-6干什么？这样在列表中添加的都是元组
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        #; 对于训练数据，这里打印metas:27097。可以认为所有的数据集，他们观测的位置都是一样的，也就是有49个位置，
        #; 那么利用这些位置计算，可以获得每张图片都能算出来其他48张图片和当前它的匹配得分，文件中只存储前10个。
        #; 然后上面循环每一个scan子文件夹，一共是79个；每个子文件夹中都有49张图片，表示要把这49张图片作为ref估计深度
        #; 最后循环7，得到的metas长度就是79 * 49 * 7 = 27097
        #! 那最后循环7是干啥的？light_idex是什么？
        #! 解答：看下面读取数据集就知道了，顺便再去看数据集的文件。这里light_index就是光照索引，也就是数据集中，
        #!     对于同一个位置拍摄的图片，使用了7种不同的光照，所以这里要把7中不同的光照都读进来。也就是说，对于每个
        #!     scan的文件夹(比如'scan2')，虽然只有49张图片，但是有7种光照，所以实际有49*7张图片
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)   # 一共有多少数据，这样dataloader调用的时候好根据batch_size算一共有多少batch

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        #; 'scan2'， 0-6， 0-48， 任意10个索引
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            #; 每个scan文件夹下都有49*7张rgb图，其中7是因为有7中不同的光照
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            #; 每个scan文件夹下都有49张深度图
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            #; 每个scan文件夹下都有49张深度图的mask, 来去除背景的影响
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            #; 所有的scan, 49张图片的内外参都是一样的，也就是这些数据集是把相机固定在49个位置进行拍摄的
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            #; 注意这里读取的图片，就把rgb都从0-255转成0-1了
            # 1.ref和src，每张图片
            imgs.append(self.read_img(img_filename))  
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            #; 注意内参是3x3, 外参是4x4
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            # 2.ref和src，每张图片的投影矩阵
            proj_matrices.append(proj_mat)

            # 3.读取ref图片的mask和深度图，并设定求解的深度范围
            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)

        # 把所有的ref和src图片stack起来，维度变成[N, C, ]
        # print(len(imgs))  # 输出图片张数，N
        # print(imgs[0].shape)  # 每张图片的[H, W, C]
        #; 注意这里statck之后得到的维度是[N, H, W, C], 然后transpose就是把维度交换，变成[N, C, H, W]
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])  
        # print(imgs.shape)
        proj_matrices = np.stack(proj_matrices)  # 形状是[N, 4, 4], 后面的 [4,4]就是每张图片的投影矩阵

        # print(depth.shape)
        # print(depth_values.shape)
        # print(mask.shape)
       
        return {"imgs": imgs,   # [N, C, H, W]
                "proj_matrices": proj_matrices,  # [N, 4, 4]
                "depth": depth, # [128, 160]，看来就是[H/4, W/4]
                "depth_values": depth_values,  #[192, ]，也就是[D,]
                "mask": mask}   # [128, 160]，也是[H/4, W/4]


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", 
        '../lists/dtu/train.txt', 'train', 3, 128)
    item = dataset[50]  # 79 * 49 * 7 = 27097份，取索引第50份

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", 
        '../lists/dtu/val.txt', 'val', 3, 128)
    item = dataset[50]  # 取索引第50份

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", 
        '../lists/dtu/test.txt', 'test', 5, 128)
    item = dataset[50]  # 取索引第50份

    # test homography here
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask"].shape)

    ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
    src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, 5)]
    ref_proj_mat = item["proj_matrices"][0]
    src_proj_mats = [item["proj_matrices"][i] for i in range(1, 5)]
    mask = item["mask"]
    depth = item["depth"]

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx)))
    D = depth.reshape([-1])
    print("X", "D", X.shape, D.shape)

    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mats[0], X)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)
    import cv2

    warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
    warped[mask[:, :] < 0.5] = 0

    cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
    cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    cv2.imwrite('../tmp2.png', src_imgs[0][:, :, ::-1] * 255)
