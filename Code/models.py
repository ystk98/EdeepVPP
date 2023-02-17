import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Models proposed in the thesis: EdeepVPP / EdeepVPPHybrid
# =============================================================================
class EdeepVPP(nn.Module):
    def __init__(self, out_size=2):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 32, kernel_size=7, padding='same')
        self.conv2 = nn.Conv1d(32, 8, kernel_size=4)
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3)
        self.pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pooling2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pooling3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=280, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=out_size)

    def forward(self, x):
        x = torch.transpose(x, 2, 1) # xの1次元と2次元を交換
        x = F.dropout(F.relu(self.conv1(x)), p=0.2)
        x = self.pooling1(x)
        x = F.dropout(F.relu(self.conv2(x)), p=0.2)
        x = self.pooling2(x)
        x = F.dropout(F.relu(self.conv3(x)), p=0.2)
        x = self.pooling3(x)
        x = self.flatten(x)
        x = F.dropout(F.relu(self.fc1(x)), p=0.2)
        y = self.fc2(x)

        return y


class EdeepVPPHybrid(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size=200, out_size=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.conv1_1 = nn.Conv1d(in_channels=5, out_channels=96, kernel_size=5)
        self.conv1_2 = nn.Conv1d(in_channels=96, out_channels=224, kernel_size=5)
        self.conv2_1 = nn.Conv1d(in_channels=224, out_channels=128, kernel_size=3)
        self.conv3_1 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=5)
        self.pooling_1 = nn.MaxPool1d(kernel_size=2)
        self.pooling_2 = nn.MaxPool1d(kernel_size=2)
        self.pooling_3 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=34, hidden_size=hidden_size, num_layers=3, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=out_size)

    def reset_state(self):
        self.lstm.reset_parameters()

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, 2, 1)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pooling_1(x)
        x = F.relu(self.conv2_1(x))
        x = self.pooling_2(x)
        x = F.relu(self.conv3_1(x))
        x = self.pooling_3(x)
        x = self.lstm(x)
        print(x[0][:, :, -1].shape)
        x = self.fc1(x[0])
        y = self.fc2(x)
        
        return y
    

# =============================================================================
# Original Models: MyEdeepVPP / ParallelEdeepVPP / OneConv
# =============================================================================
class MyEdeepVPP(nn.Module):
    """
    Note: removed Dropout of convolution layers from EdeepVPP

    """
    def __init__(self, out_size=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=7, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=8, kernel_size=4), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=280, out_features=32), 
            nn.ReLU(), 
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=32, out_features=out_size), 
        )

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y = self.fc(x)

        return y
    
class ParallelEdeepVPP(nn.Module):
    def __init__(self, out_size=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=7, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=8, kernel_size=28), 
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=5888, out_features=2048), 
            nn.ReLU(), 
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=2048, out_features=1000), 
            nn.ReLU(), 
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=1000, out_features=out_size)
        )

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        cat = torch.cat((conv1, conv2), dim=1)
        y = self.fc(cat)

        return y

class OneConv(nn.Module):
    def __init__(self, out_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=5, out_channels=128, kernel_size=9, padding='same')
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=38400, out_features=2048), 
            nn.ReLU(), 
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=2048, out_features=1024), 
            nn.ReLU(), 
            nn.Dropout(p=0.2), 
            nn.Linear(in_features=1024, out_features=out_size)
        )

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        y = self.fc(x)

        return y