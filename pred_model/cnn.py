from torch import nn, Tensor


def conv_block(in_channels: int, out_channels: int, pool: bool=False
    ) -> nn.Module:
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels, track_running_stats = False),
              nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class cnn_small(nn.Module):
    """
    Expects images of size (28, 28, C)
    """
    def __init__(self, C: int, n_class: int, **kwargs):
        super().__init__()
        
        self.conv1 = conv_block(C, 32, pool=True)
        self.conv2 = conv_block(32, 32, pool=True)
        self.classifier = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(1568, n_class))
        
    def forward(self, xb: Tensor) -> Tensor:
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.classifier(out)
        return out