import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_height=32, img_width=128, num_classes=37, hidden_size=256):
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes  # alphabet + blank
        self.hidden_size = hidden_size
        
        # =============== PARTIE CNN ===============
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x128 -> 16x64
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x64 -> 8x32
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Block 4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))  # 8x32 -> 4x32 (pool seulement en hauteur)
        
        # Block 5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Block 6
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d((2, 1))  # 4x32 -> 2x32
        
        # Final conv
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)
        # Output: 1x31x512
        
        # =============== PARTIE RNN ===============
        # Calcul de la taille d'entrée RNN
        self.rnn_input_size = 512
        
        # LSTM bidirectionnel
        self.lstm1 = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,  # bidirectionnel
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # =============== CLASSIFICATION ===============
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # =============== CNN FEATURE EXTRACTION ===============
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Block 5
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Block 6
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool6(x)
        
        # Final conv
        x = F.relu(self.conv7(x))
        
        # =============== RESHAPE POUR RNN ===============
        # x shape: [batch, channels, height, width] -> [batch, width, features]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        x = x.view(batch_size, w, c * h)  # [batch, width, features]
        
        # =============== RNN SEQUENCE MODELING ===============
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # =============== CLASSIFICATION ===============
        x = self.dropout(x)
        x = self.classifier(x)  # [batch, seq_len, num_classes]
        
        # Log softmax pour CTC loss
        x = F.log_softmax(x, dim=2)
        
        # Permutation pour CTC: [seq_len, batch, num_classes]
        x = x.permute(1, 0, 2)
        
        return x


# Création du modèle
def create_model(num_classes=37):
    return CRNN(
        img_height=32,
        img_width=128, 
        num_classes=num_classes,
        hidden_size=256
    )