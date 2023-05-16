import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential
import torch

class ResidualBottleneckBlock(nn.Module):
    
    def __init__(self, in_channels, expansion_out, point_out, dw_stride):

        super().__init__()

        self.dw_stride = dw_stride
        self.point_out = point_out

        if dw_stride == 1:
            padding = "same"

        else: 
            padding = 1


        self.expansion = Conv2d(in_channels = in_channels, out_channels = expansion_out, kernel_size = 1)
        self.depthwise = Conv2d(in_channels = expansion_out, out_channels = expansion_out, kernel_size = 3, groups = expansion_out, stride = dw_stride, padding = padding)
        self.projection = Conv2d(in_channels = expansion_out, out_channels = point_out, kernel_size = 1)

        self.batchnorm_expansion = BatchNorm2d(num_features=expansion_out)
        self.batchnorm_depthwise = BatchNorm2d(num_features=expansion_out)
        self.batchnorm_projection = BatchNorm2d(num_features=point_out)
        self.relu = ReLU()

    def forward(self, X):

        if self.dw_stride == 1:
            identity = X
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            conv_id = Conv2d(identity.shape[1], self.point_out, kernel_size=1, padding = 'same')
            batchnorm = BatchNorm2d(num_features=self.point_out)

            # need to move these to cpu or gpu manually because PyTorch only does this for those in init and not forward
            conv_id = conv_id.to(device)
            batchnorm = batchnorm.to(device)
            identity = conv_id(identity)
            identity = batchnorm(identity)

        out = self.expansion(X)
        out = self.batchnorm_expansion(out)
        out = self.relu(out)
        
        out = self.depthwise(out)
        out = self.batchnorm_depthwise(out)
        out = self.relu(out)     

        out = self.projection(out)
        out = self.batchnorm_projection(out)

        if self.dw_stride == 1:
            out += identity

        return out
    
class DepthwiseSeparableBlock(nn.Module):

    def __init__(self, in_channels, point_out, dw_stride):
        super().__init__()

        if dw_stride == 1:
            padding = "same"

        else: 
            padding = 1

        self.depthwise = Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, groups = in_channels, stride = dw_stride, padding = 1)
        self.pointwise = Conv2d(in_channels = in_channels, out_channels = point_out, kernel_size = 1)
        self.batchnorm_depthwise = BatchNorm2d(num_features=in_channels)
        self.batchnorm_pointwise = BatchNorm2d(num_features=point_out)
        self.relu = ReLU()

    def forward(self, input):
        out = self.depthwise(input)
        out = self.batchnorm_depthwise(out)
        out = self.relu(out)

        out = self.pointwise(out)
        out = self.batchnorm_pointwise(out)
        out = self.relu(out)
        
        return out


if __name__ == "__main__":
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(1, 3, 32, 32).astype('float32')
    X = torch.tensor(X)

    model = ResidualBottleneckBlock()
    print (model.forward(X))