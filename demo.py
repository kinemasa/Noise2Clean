import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



# データを用意する
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

print(f"訓練用データセット数: {len(train_data)}")
print(f"テスト用データセット数: {len(test_data)}")
print(f'データの形状: {train_data[0][0][0].shape}')

## データの確認用に表示を行う
fig, ax = plt.subplots(1, 10, figsize=(15,5))
plt.suptitle('Original MNIST', y=0.65)
for i in range(10):
    ax[i].imshow(train_data[i][0][0].cpu().numpy(), cmap='gray')
plt.tight_layout()
plt.show()

#訓練用データのうち20%を検証用データとして利用する
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

#ノイズ追加の例
image=train_data[0][0][0]
noise = torch.randn_like(image)*0.4 #MNISTのデータと同じ形状のガウシアンノイズを作成
noisy_image=image+noise
noisy_image=image+noise
fig, ax = plt.subplots(1, 2, figsize=(8,5))
ax[0].imshow(image.cpu().numpy(), cmap='gray')
ax[1].imshow(noise.cpu().numpy(), cmap='gray' )
ax[0].set_title('Noisy image (Input)')
ax[1].set_title('Noisy image (Input)')
plt.show()


# U-Netのモデル定義
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128]):
        super().__init__()
        self.encoders = nn.ModuleList()
        for feature in features:
            self.encoders.append(
                nn.Sequential(
                    DoubleConv(in_channels, feature),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1])
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.decoders.append(
                nn.ConvTranspose2d(feature*2, feature//2, kernel_size=2, stride=2)
            )
            self.decoders.append(
                DoubleConv(feature//2, feature//2))
        self.final_conv = nn.Conv2d(features[0]//2, out_channels, kernel_size=1)

    def forward(self, x):
        inp=x.clone()
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoders), 2):
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.pad(x, [0, skip_connection.shape[3] - x.shape[3], 0, skip_connection.shape[2] - x.shape[2]])
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[i+1](self.decoders[i](concat_x))
        return self.final_conv(x)


# モデルの訓練
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = nn.MSELoss() #最小二乗誤差
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        noise = torch.randn_like(images)*0.4
        noisy_images=images+noise #ノイズデータの作成
        optimizer.zero_grad()
        outputs = model(noisy_images) #モデルの入力はノイズデータ
        loss = criterion(outputs, images) #モデルの出力とノイズの無い元の画像の誤差を計算
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return model, optimizer, epoch_loss

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noise = torch.randn_like(images)*0.4
            noisy_images=images+noise
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return model, epoch_loss

epochs = 10
best_loss = 1e10
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_losses = []
valid_losses = []
for epoch in range(epochs):
    print(f'epoch:{epoch+1}')
    model, optimizer, train_loss = train(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    print(f'Training loss {train_loss}')
    model, valid_loss = validate(model, valid_loader, criterion)
    print(f'Validation loss {valid_loss}')
    valid_losses.append(valid_loss)
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')
model.load_state_dict(torch.load('best_model.pth')) #検証用データに対する損失が最も小さいモデルを最終モデルとして採用

print('Finish Training')

# テストデータで評価
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        noise = torch.randn_like(images)*0.4
        noisy_images=images+noise
        outputs = model(noisy_images)
        fig, ax = plt.subplots(3, 5, figsize=(10,8))
        for i in range(5):
            ax[0][i].imshow(noisy_images[i][0].cpu().numpy(), cmap='gray')
            ax[0][i].set_title('Input')
            ax[1][i].imshow(outputs[i][0].cpu().numpy(), cmap='gray')
            ax[1][i].set_title('Denoised')
            ax[2][i].imshow(images[i][0].cpu().numpy(), cmap='gray')
            ax[2][i].set_title('Original')
        plt.show()
        break