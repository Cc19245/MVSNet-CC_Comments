import torch
height = 2
width = 3

y, x = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
#; 假设h=2, w=3, 那么meshgrid生成的y 和 x分别为：
#; y:  0 0 0    x:  0 1 2
#;     1 1 1        0 1 2

print('y = ', y)
print('x = ', x)

# 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样
#! 疑问：暂时记住利用meshgrid创建的就必须用contiguous来让内存分配连续吧
y, x = y.contiguous(), x.contiguous() 
print('y = ', y)
print('x = ', x)

y, x = y.view(height * width), x.view(height * width)  # 把二维的x和y都转成一维的，也就是矩阵变向量
print('y = ', y)
print('x = ', x)

xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
print('xyz = ', xyz)

batch = 4
xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
print('xyz = ', xyz)


proj_x_normalized = torch.randn(4, 5, 9)
proj_y_normalized = torch.randn(4, 5, 9)
# proj_x_normalized维度是[B, 2, Ndepth, H*W]， 沿着第3个维度stack，
proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
print(proj_xy.shape)

depth_values = torch.arange(3)
print(depth_values.shape)
a = torch.randn(4, 3, 2, 5)
depth_values = depth_values.view(*depth_values.shape, 1, 1)
print(depth_values.shape, a.shape)
# print(a + depth_values)

print(torch.cuda.is_available())