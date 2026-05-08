import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    Standard U-Net for Lung Segmentation from 2D CT Slices.

    Architecture:
        - Encoder: 4 convolutional blocks, each halving spatial resolution via MaxPool2D.
        - Bottleneck: Deepest feature extraction layer (1024 channels).
        - Decoder: 4 upsampling blocks with skip connections to preserve spatial detail.
        - Output: Sigmoid-activated single-channel mask (lung tissue probability per pixel).
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder path with skip connections
        d4 = torch.cat((e4, self.up4(b)), dim=1)
        d4 = self.dec4(d4)

        d3 = torch.cat((e3, self.up3(d4)), dim=1)
        d3 = self.dec3(d3)

        d2 = torch.cat((e2, self.up2(d3)), dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat((e1, self.up1(d2)), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))


class NoduleClassifier3D(nn.Module):
    """
    3D CNN for classifying Lung Nodules (Benign vs Malignant).

    Input:  Tensor of shape (Batch, 1, 32, 32, 32) — a single-channel 3D volume.
    Output: Sigmoid probability per sample — 0.0 (Benign) to 1.0 (Malignant).

    Architecture:
        - 3x Conv3D blocks, each halving spatial resolution via MaxPool3D.
        - Flatten → 2x Fully Connected layers with ReLU and Dropout (p=0.5).
        - Sigmoid output for binary classification.
        - get_last_conv_layer() exposes the conv3 layer for Grad-CAM explainability.
    """
    def __init__(self):
        super(NoduleClassifier3D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)  # Output: (32, 16, 16, 16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2)  # Output: (64, 8, 8, 8)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2)  # Output: (128, 4, 4, 4)
        )

        self.flatten_dim = 128 * 4 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

    def get_last_conv_layer(self):
        """Returns the final convolutional layer. Used by Grad-CAM for heatmap generation."""
        return self.conv3[0]
