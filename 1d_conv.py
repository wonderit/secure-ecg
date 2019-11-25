import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv1d(9, 18, kernel_size=3)  # 9 input channels, 18 output channels
        self.conv2 = nn.Conv1d(18, 36, kernel_size=3)  # 18 input channels from previous Conv. layer, 36 out
        self.conv2_drop = nn.Dropout2d()  # dropout
        self.fc1 = nn.Linear(1044, 72)  # Fully-connected classifier layer
        self.fc2 = nn.Linear(72, 19)  # Fully-connected classifier layer

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        print('1', x.shape)
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        print('2', x.shape)

        # point A
        x = x.view(x.shape[0], -1)

        # point B
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    m = model()
    data = torch.randn(64, 9, 125)
    print('data', data.shape)
    out = m(data)
    print('main', out.shape)