import utilsKP
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# test network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.conv4 = nn.Conv2d(16, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#logging for tensorflow
writer = SummaryWriter()

#create a net
net = Net()

#total images
numb_images = 5011
bs =10
numb_iter_epoch = numb_images/bs

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = utilsKP.CosAnnealLR(optimizer,base_lr=1, max_lr=2, batch_size=bs, numb_images = numb_images)

# use same writer for LR
scheduler.setWriter(writer)

for i in range(30):
    print(f"batch number{i}")
    for j in range(int(numb_iter_epoch)):
        scheduler.batch_step()



    # def test_setWriter(self):
    #     self.fail()
    #
    # def test_get_lr(self):
    #     self.fail()
