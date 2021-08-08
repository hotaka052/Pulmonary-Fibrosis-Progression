import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        
        #encoder
        self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv3d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv3d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        
        self.linear1 = nn.Linear(in_features = 128, out_features = 64)
        self.linear2 = nn.Linear(in_features = 64, out_features = latent_dim)
        
        self.max_pool = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.avg_pool = nn.AvgPool3d(kernel_size = (5, 16, 16))
        
        #decoder
        self.linear3 = nn.Linear(in_features = latent_dim, out_features = 2 * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose3d(
            in_channels = 1, out_channels = 128, kernel_size = 3, stride = 2,
            dilation = 2, padding = 1, output_padding = (0, 1, 1)
        )
        self.deconv2 = nn.ConvTranspose3d(
            in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2,
            dilation = 1, padding = 1
        )
        self.deconv3 = nn.ConvTranspose3d(
            in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2,
            dilation = 1, padding = 1
        )
        self.deconv4 = nn.ConvTranspose3d(
            in_channels = 32, out_channels = 16, kernel_size = 4, stride = 2,
            dilation = 1, padding = 1
        )
        self.deconv5 = nn.Conv3d(in_channels = 16, out_channels = 1, kernel_size = 1)
        
    def encoder(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace = True)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x, inplace = True)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x, inplace = True)
        x = self.max_pool(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = x.view(-1, 128)
        x = self.linear1(x)
        x = F.relu(x, inplace = True)
        x = self.linear2(x)
        
        return x
        
    def forward(self, x):
        x = self.encoder(x)
        
        #decoder
        x = self.linear3(x)
        x = x.view(-1, 1, 2, 7, 7)
        x = self.deconv1(x)
        x = F.relu(x, inplace = True)
        x = self.deconv2(x)
        x = F.relu(x, inplace = True)
        x = self.deconv3(x)
        x = F.relu(x, inplace = True)
        x = self.deconv4(x)
        x = F.relu(x, inplace = True)
        x = self.deconv5(x)
        
        return x