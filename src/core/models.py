from . import *

class Model_C(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.word_embedding = nn.Sequential(
            nn.Embedding(65536, 200)
        )
        
        self.lstm = nn.Sequential(            
            nn.LSTM(200, 256, 1, batch_first=True, bidirectional=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        outputs = self.word_embedding(inputs)
        outputs = self.lstm(outputs)[0][:, -1, :]
        outputs = self.classifier(outputs)
        
        return outputs