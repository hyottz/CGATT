import torch
from torch import nn
from torch.nn import functional as F

class TemporalAttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(TemporalAttentionModule, self).__init__()
        self.in_dim = in_dim
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, temporal_dim, features_dim)
        proj_query = self.query_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        proj_value = self.value_conv(x)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return out + x

class GRUAttention(nn.Module):
    def __init__(self, in_channels=3, num_frames=25, hidden_dim=512, gru_layers=1):
        super(GRUAttention, self).__init__()

        # 3D Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        spatial_size_after_conv = 112 // 16  # Considering MaxPool3d layers
        self.flat_dim = 256 * spatial_size_after_conv ** 2 

        self.bi_gru = nn.GRU(input_size=self.flat_dim, hidden_size=hidden_dim, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.temporal_attention = TemporalAttentionModule(in_dim=hidden_dim*2)  # for Bi-GRU
        self.fc = nn.Linear(hidden_dim*2, 2048)  # Adjust according to the Bi-GRU output

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = self.conv_layers(x)
        # Adjust dimensions for GRU input
        x = x.view(b, d, -1)

        x, _ = self.bi_gru(x)
        
        x = x.transpose(1, 2)  # Change from (batch_size, temporal_dim, features_dim*2) to (batch_size, features_dim*2, temporal_dim)
        
        x = self.temporal_attention(x)
        
        x = x.transpose(1, 2)  # Change back to (batch_size, temporal_dim, features_dim*2)
        
        x = self.fc(x)
        
        # Ensure the output shape is [batch_size, 25, 2048]
        x = x.transpose(1, 2)  # Ensuring the output shape

        return x


def GRU_ATT(**kwargs):
    model = GRUAttention(in_channels=3, num_frames=25, hidden_dim=512, gru_layers=1)
    return model
