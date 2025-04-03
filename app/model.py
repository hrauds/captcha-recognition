import torch
import torch.nn as nn
import torch.nn.functional as F

CHAR_SET = "2345789ABCDEFHKLMNPRTUVWXYZ"
BLANK_TOKEN = 0
NUM_CHARS = len(CHAR_SET)
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHAR_SET)}
MODEL_PATH = "captcha_model.pth"
DROPOUT_RATE = 0.3

class CaptchaModel(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super(CaptchaModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.projection = nn.Linear(64 * 6, 96)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.lstm = nn.LSTM(input_size=96, hidden_size=96, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(96 * 2, NUM_CHARS + 1)

    def forward(self, x):
        features = self.cnn(x)
        features = self.dropout(features)
        batch_size, channels, height, width = features.size()
        sequence = features.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)
        sequence = self.projection(sequence)
        sequence, _ = self.lstm(sequence)
        logits = self.classifier(sequence)
        return F.log_softmax(logits, dim=2)

def load_model():
    """Load model and the trained weights."""
    model = CaptchaModel()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model
