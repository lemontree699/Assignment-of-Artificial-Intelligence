import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# 超参数
epochs = 10
batch_size = 64
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.005
donwload_mnist = False 
n_test_img = 5
momentum = 0.5
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                           
    transform=torchvision.transforms.ToTensor(),              
    download=donwload_mnist,                
)
train_load = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# 构建自编码机
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 25),
        )
        self.decoder = nn.Sequential(
            nn.Linear(25, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

# 训练AutoEncoder
f, a = plt.subplots(2, n_test_img, figsize=(5, 2))
plt.ion()  

# 选择几张图片进行显示
view_data = train_data.train_data[:n_test_img].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(epochs):
    for step, (x, b_label) in enumerate(train_load):
        b_x = x.view(-1, 28*28)   
        b_y = x.view(-1, 28*28)  

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)      
        optimizer.zero_grad()              
        loss.backward()                 
        optimizer.step()                  

        if step % 100 == 0:  # 每隔100个数据输出一次
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            _, decoded_data = autoencoder(view_data)
            for i in range(n_test_img):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

# 读取mnist数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist/', train=True, download=donwload_mnist,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist/', train=False, download=donwload_mnist,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True)

# 构建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size = 2, padding = 2)
        self.conv2 = nn.Conv2d(25, 32, kernel_size = 3, padding = 2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(288, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate,
#                       momentum=momentum)

# 训练模型
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]

def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        (data, _)= autoencoder(Variable(data.view(-1, 28*28)))
        target = Variable(target)
        optimizer.zero_grad()
        output = net(data.view(-1,1,5,5))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            (data, _)= autoencoder(Variable(data.view(-1, 28*28)))
            target = Variable(target)
            output = net(data.view(-1,1,5,5))
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nloss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
    