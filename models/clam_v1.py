import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio


class CLAM(nn.Module):
    def __init__(self, in_channel1 , in_channel2 , embed_dim1 , embed_dim2):
        
        super(CLAM, self).__init__()
        self.conv1 = nn.Conv1d(in_channel1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channel2, 3, kernel_size=3, padding=1)

        self.crossAttention1 = nn.MultiheadAttention(embed_dim =embed_dim1, num_heads=4)
        self.crossAttention2 = nn.MultiheadAttention(embed_dim =embed_dim2, num_heads=4)

        self.fc1 = nn.Linear(embed_dim1, 512)
        self.fc2 = nn.Linear(embed_dim2, 512)

        self.fc3 = nn.Linear(512 * 2, 1)  # Final output layer

        
    def forward(self, embed1, embed2):
        embed1 = self.conv1(embed1)
        embed1 = F.relu(embed1)
        
        embed2 = self.conv2(embed2)
        embed2 = F.relu(embed2)


        Q1, K1, V1 = embed1[:, 0, :].unsqueeze(1), embed1[:, 0, :].unsqueeze(1), embed1[:, 0, :].unsqueeze(1)
        Q2, K2, V2 = embed2[:, 0, :].unsqueeze(1), embed2[:, 0, :].unsqueeze(1), embed2[:, 0, :].unsqueeze(1)
        ou1, _ = self.crossAttention1(Q1, K1, V1)
        ou2, _ = self.crossAttention2(Q2, K2, V2)
     
        ou1 = ou1.squeeze(1)
        ou2 = ou2.squeeze(1)

        ou1 = self.fc1(ou1)
        ou2 = self.fc2(ou2)


        # Concatenate the outputs
        out = torch.cat((ou1, ou2), dim=1)

        # Apply a final linear layer

        out = self.fc3(out)
        return out
    
    def forward_training(self, embed1, embed2):
        embed1 = self.conv1(embed1)
        embed1 = F.relu(embed1)
        
        embed2 = self.conv2(embed2)
        embed2 = F.relu(embed2)


        Q1, K1, V1 = embed1[:, 0, :].unsqueeze(1), embed1[:, 0, :].unsqueeze(1), embed1[:, 0, :].unsqueeze(1)
        Q2, K2, V2 = embed2[:, 0, :].unsqueeze(1), embed2[:, 0, :].unsqueeze(1), embed2[:, 0, :].unsqueeze(1)
        ou1, _ = self.crossAttention1(Q1, K1, V1)
        ou2, _ = self.crossAttention2(Q2, K2, V2)
     
        ou1 = ou1.squeeze(1)
        ou2 = ou2.squeeze(1)

        ou1 = self.fc1(ou1)
        ou2 = self.fc2(ou2)


        # Concatenate the outputs
        out = torch.cat((ou1, ou2), dim=1)


        out = self.fc3(out)
        return out , ou1 , ou2