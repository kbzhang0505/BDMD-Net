import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss

# class ColorLoss(nn.Module):
#     def __init__(self):
#         super(ColorLoss, self).__init__()
#
#     def forward(self, predict, target):
#         b, c, h, w = target.shape
#         target_view = target.view(b, c, h * w).permute(0, 2, 1)
#         predict_view = predict.view(b, c, h * w).permute(0, 2, 1)
#         target_norm = torch.nn.functional.normalize(target_view, dim=-1)
#         predict_norm = torch.nn.functional.normalize(predict_view, dim=-1)
#         cose_value = target_norm * predict_norm
#         cose_value = torch.sum(cose_value, dim=-1)
#         color_loss = torch.mean(1 - cose_value)
#
#         return color_loss


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, predict, target):
        # 初始化 L1 损失的累加变量
        total_l1_loss_U = 0
        total_l1_loss_V = 0

        # 遍历每个图像
        for i in range(predict.shape[0]):
            # 提取单个图像
            img1 = predict[i]
            img2 = target[i]

            # 转换单个图像到 YUV
            yuv_image1 = rgb_to_yuv(img1)
            yuv_image2 = rgb_to_yuv(img2)

            # 提取 U 和 V 通道
            U1, V1 = yuv_image1[1], yuv_image1[2]
            U2, V2 = yuv_image2[1], yuv_image2[2]

            # 计算 L1 损失并累加
            total_l1_loss_U += F.l1_loss(U1, U2, reduction='sum')
            total_l1_loss_V += F.l1_loss(V1, V2, reduction='sum')
        color_loss = total_l1_loss_U + total_l1_loss_V
        return color_loss


def rgb_to_yuv(image):
    RGB_to_YUV = torch.tensor([[0.299, 0.587, 0.114],
                               [-0.14713, -0.28886, 0.436],
                               [0.615, -0.51499, -0.10001]])
    RGB_to_YUV = RGB_to_YUV.to(torch.device('cuda'))
    # 调整图像张量形状以适应矩阵乘法
    image = image.permute(1, 2, 0).contiguous().view(-1, 3)

    # 转换到 YUV
    yuv_image = torch.matmul(image, RGB_to_YUV.t())

    # 重新调整形状回到 (C, H, W)
    yuv_image = yuv_image.view(128, 128, 3).permute(2, 0, 1)

    return yuv_image


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        # 定义 Sobel 卷积核
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 将 Sobel 卷积核扩展到三个通道
        self.sobel_x = sobel_x.repeat(3, 1, 1, 1).to(torch.device('cuda'))
        self.sobel_y = sobel_y.repeat(3, 1, 1, 1).to(torch.device('cuda'))

        # 将卷积核注册为非可训练参数
        self.sobel_x = nn.Parameter(self.sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(self.sobel_y, requires_grad=False)

    def compute_edges(self, image_tensor):
        # 对每个通道计算梯度
        gradients_x = F.conv2d(image_tensor, self.sobel_x, padding=1, groups=3)
        gradients_y = F.conv2d(image_tensor, self.sobel_y, padding=1, groups=3)

        # 计算梯度幅度
        gradients = torch.sqrt(gradients_x ** 2 + gradients_y ** 2)

        return gradients

    def forward(self, pred, target):
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        return F.mse_loss(pred_edges, target_edges)