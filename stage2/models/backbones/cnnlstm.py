import torch
from torch import nn

class Simple3DCNNLSTM(nn.Module):
    def __init__(self, in_channels=3, num_frames=25, hidden_dim=512, lstm_layers=1):
        super(Simple3DCNNLSTM, self).__init__()
        self.num_frames = num_frames

        # 3D Convolutional layers with MaxPooling applied to spatial dimensions only
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=1, padding=1), # output channel default: 16
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1), # default: 32
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1), # default: 64
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), 

            
        )

        # Assuming the input video size is 112x112
        spatial_size_after_conv = 112 // 8  # Considering three MaxPool3d layers with stride (1, 2, 2)
        self.flat_dim = 128 * spatial_size_after_conv ** 2  # default: 64

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.flat_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim*2, 2048)  # Adjust the output dimensions to 2048


    def forward(self, x):
        b, c, d, h, w = x.shape
        x = self.conv_layers(x)
        
        # Prepare the output from conv layers for the LSTM
        x = x.view(b, d, -1)  # Flatten spatial dimensions while keeping the depth (time) dimension
        
        # LSTM layer
        x, _ = self.lstm(x)  # Use all the hidden states
        
        # Apply fully connected layer to each time step
        x = self.fc(x)  # No need to use only the last hidden state
        
        # Ensure the output shape is [batch_size, 2048, num_frames]
        x = x.transpose(1, 2)  # Transpose to get the correct shape
        
        return x


def CNNLSTM(**kwargs):
    model = Simple3DCNNLSTM(in_channels=3, num_frames=25, hidden_dim=512, lstm_layers=1)
    return model