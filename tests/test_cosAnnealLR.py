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
writer = utilsKP.Writer('./runs')

#create a net
net = Net()

# #total images
numb_images = 5011
bs =64

numb_images = 10
bs =2
numb_iter_epoch = numb_images/bs

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# step_size = [5,5,5,5,10,10,10]
step_size = [1,1,1]
base_lr=.001
max_lr=.1

lr = utilsKP.TriangularLR()
lr = utilsKP.CosignLR()
# lra = utilsKP.LR_anneal_linear()
lra=utilsKP.LR_anneal_linear()
scheduler = utilsKP.CyclicLR_Scheduler(optimizer,max_lr = max_lr, min_lr = base_lr,LR=lr, LR_anneal=lra, batch_size=bs,
                                       numb_images = numb_images, step_size = step_size, writer = writer)

#numb batches = sum(step_size)*2
for batch in range(sum(step_size)*2):
# for i in range(2*sum(step_size)):
    print(f"batch number {batch}")
    for j in range(scheduler.max_iter_per_epoch):
        # print(f"   step{j}:", end=' ')
        scheduler.batch_step()



def plot_LRs():
    '''
calculates triangular learning rates
'''
    import math
    import matplotlib.pyplot as plt
    iterations = 500
    maxLR = 2
    minLR = 1
    lrs = utilsKP.TriangularLR().getLRs(iterations, 2, 1)
    plt.title("triangular learning rate")
    plt.xlabel("cycle")
    plt.ylabel("lr")
    plt.plot(lrs)
    plt.show()
    lrs = utilsKP.CosignLR().getLRs(iterations, 2, 1)
    plt.title("cosign learning rate")
    plt.xlabel("cycle")
    plt.ylabel("lr")
    plt.plot(lrs)
    plt.show()
    lrs = utilsKP.TriangularLR_LRFinder().getLRs(iterations, 2, 1)
    plt.title("learningratefinder learning rate")
    plt.xlabel("cycle")
    plt.ylabel("lr")
    plt.plot(lrs)
    plt.show()


# plot_LRs()