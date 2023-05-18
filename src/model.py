from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch

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

    X = np.random.rand(5, 3, 32, 32).astype('float32')

    X = torch.tensor(X)

    model = Network(num_classes= 3)
    print (model.forward(X))