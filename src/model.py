from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch
from src.mobilenet_utils import DepthwiseSeparableBlock, ResidualBottleneckBlock

class MobileNetV1(nn.Module):

    def __init__(self, in_channels = 3, num_classes = 3):
        super(MobileNetV1, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.block_1 = Conv2d(in_channels=self.in_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        
        self.block32_64 = DepthwiseSeparableBlock(in_channels=32, point_out=64, dw_stride = 1)
        self.block64_128 = DepthwiseSeparableBlock(in_channels=64, point_out=128, dw_stride = 2)
        self.block128_128 = DepthwiseSeparableBlock(in_channels=128, point_out=128, dw_stride = 1)
        self.block128_256 = DepthwiseSeparableBlock(in_channels=128, point_out=256, dw_stride = 2)
        self.block256_256 = DepthwiseSeparableBlock(in_channels=256, point_out=256, dw_stride = 1)
        self.block256_521 = DepthwiseSeparableBlock(in_channels=256, point_out=512, dw_stride = 2)
        self.block512_512_1 = DepthwiseSeparableBlock(in_channels=512, point_out=512, dw_stride = 1)
        self.block512_512_2 = DepthwiseSeparableBlock(in_channels=512, point_out=512, dw_stride = 1)
        self.block512_512_3 = DepthwiseSeparableBlock(in_channels=512, point_out=512, dw_stride = 1)
        self.block512_512_4 = DepthwiseSeparableBlock(in_channels=512, point_out=512, dw_stride = 1)
        self.block512_512_5 = DepthwiseSeparableBlock(in_channels=512, point_out=512, dw_stride = 1)
        self.block512_1024 = DepthwiseSeparableBlock(in_channels=512, point_out=1024, dw_stride = 2)
        self.block1024_1024 = DepthwiseSeparableBlock(in_channels=1024, point_out=1024, dw_stride = 2)

        self.avgpool = AdaptiveAvgPool2d((1,1))
        self.fc = Linear(1024,self.num_classes)
        self.softmax = Softmax(dim = 1)
    
    def forward(self, input):

        output = self.block_1(input)
        output = self.block32_64(output)
        output = self.block64_128(output)
        output = self.block128_128(output)
        output = self.block128_256(output)
        output = self.block256_256(output)
        output = self.block256_521(output)
        output = self.block512_512_1(output)
        output = self.block512_512_2(output)
        output = self.block512_512_3(output)
        output = self.block512_512_4(output)
        output = self.block512_512_5(output)
        output = self.block512_1024(output)
        output = self.block1024_1024(output)
        
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        output = self.softmax(output)

        return output
    
class MobileNetV2(nn.Module):

    def __init__(self, in_channels = 3, num_classes = 3):
        super(MobileNetV2, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_layer_1 = Conv2d(in_channels=self.in_channels, out_channels = 32, kernel_size = 3, stride = 2)
        
        self.block1_seq_1 = ResidualBottleneckBlock(in_channels=32, expansion_out = 32, point_out=16, dw_stride = 1)

        self.block2_seq_1 = ResidualBottleneckBlock(in_channels=16, expansion_out = 16*6, point_out=24, dw_stride = 2)
        self.block2_seq_2 = ResidualBottleneckBlock(in_channels=24, expansion_out = 24*6, point_out=24, dw_stride = 1)

        self.block3_seq_1 = ResidualBottleneckBlock(in_channels=24, expansion_out = 24*6, point_out=32, dw_stride = 2)
        self.block3_seq_2 = ResidualBottleneckBlock(in_channels=32, expansion_out = 32*6, point_out=32, dw_stride = 1)
        self.block3_seq_3 = ResidualBottleneckBlock(in_channels=32, expansion_out = 32*6, point_out=32, dw_stride = 1)

        self.block4_seq_1 = ResidualBottleneckBlock(in_channels=32, expansion_out = 32*6, point_out=64, dw_stride = 2)
        self.block4_seq_2 = ResidualBottleneckBlock(in_channels=64, expansion_out = 64*6, point_out=64, dw_stride = 1)
        self.block4_seq_3 = ResidualBottleneckBlock(in_channels=64, expansion_out = 64*6, point_out=64, dw_stride = 1)
        self.block4_seq_4 = ResidualBottleneckBlock(in_channels=64, expansion_out = 64*6, point_out=64, dw_stride = 1)

        self.block5_seq_1 = ResidualBottleneckBlock(in_channels=64, expansion_out = 64*6, point_out=96, dw_stride = 1)
        self.block5_seq_2 = ResidualBottleneckBlock(in_channels=96, expansion_out = 96*6, point_out=96, dw_stride = 1)
        self.block5_seq_3 = ResidualBottleneckBlock(in_channels=96, expansion_out = 96*6, point_out=96, dw_stride = 1)

        self.block6_seq_1 = ResidualBottleneckBlock(in_channels=96, expansion_out = 96*6, point_out=160, dw_stride = 2)
        self.block6_seq_2 = ResidualBottleneckBlock(in_channels=160, expansion_out = 160*6, point_out=160, dw_stride = 1)
        self.block6_seq_3 = ResidualBottleneckBlock(in_channels=160, expansion_out = 160*6, point_out=160, dw_stride = 1)

        self.block7_seq_1 = ResidualBottleneckBlock(in_channels=160, expansion_out = 160*6, point_out=320, dw_stride = 1)

        self.conv_layer_2 = Conv2d(in_channels=320, out_channels = 1280, kernel_size = 1, stride = 1)

        self.avgpool = AdaptiveAvgPool2d((1,1))

        self.conv_layer_3 = Conv2d(in_channels=1280, out_channels = 1280, kernel_size = 1, stride = 1)
        
        self.fc = Linear(1280,self.num_classes)
        
        self.softmax = Softmax(dim = 1)
    
    def forward(self, input):

        output = self.conv_layer_1(input)
        output = self.block1_seq_1(output)
        output = self.block2_seq_1(output)
        output = self.block2_seq_2(output)
        output = self.block3_seq_1(output)
        output = self.block3_seq_2(output)
        output = self.block3_seq_3(output)
        output = self.block4_seq_1(output)
        output = self.block4_seq_2(output)
        output = self.block4_seq_3(output)
        output = self.block4_seq_4(output)
        output = self.block5_seq_1(output)
        output = self.block5_seq_2(output)
        output = self.block5_seq_3(output)
        
        output = self.block6_seq_1(output)
        output = self.block6_seq_2(output)
        output = self.block6_seq_3(output)
        output = self.block7_seq_1(output)

        output = self.conv_layer_2(output)
        
        output = self.avgpool(output)
        output = self.conv_layer_3(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        output = self.softmax(output)

        return output

class Network(nn.Module):

    def __init__(self, num_classes, in_channels = 3):
        super(Network, self).__init__()
        self.relu = ReLU()

        # conv1
        self.conv1 = Conv2d(in_channels = in_channels, out_channels=64, kernel_size=7, stride = 2)

        # batch norm layer
        self.conv1_bn = BatchNorm2d(num_features=64)

        # conv2_x
        self.maxpool1 = MaxPool2d(kernel_size=3, stride = 2)

        # avg pool, 1000 fc and softmax
        self.avg_pool = AdaptiveAvgPool2d((1,1)) # pytorch's implementation uses adaptive. Not sure what's the diff
        self.fc = Linear(64,num_classes) # 1000 dim fc
        self.softmax = Softmax(dim = 1)

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv1_bn(out)
        out = self.relu(out)

        out = self.maxpool1(out)

        out = self.avg_pool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        out = self.softmax(out)

        return out

if __name__ == "__main__":
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 224, 224).astype('float32')

    X = torch.tensor(X)

    model = MobileNetV1()
    print (model.forward(X))