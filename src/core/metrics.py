from . import *

class Metric_Loss:
    def __init__(self):
        self.sum = 0.0
        self.num = 0
        
    @torch.no_grad()
    def step(self, loss):
        loss = loss.detach().item()
        
        self.sum += loss
        self.num += 1
        
    def calc_mean(self):
        mean = self.sum / self.num
        
        return mean
        
    def reset(self):
        self.sum = 0.0
        self.num = 0
        
class Metric_Accuracy:
    def __init__(self):
        self.sum = 0
        self.num = 0
        
    @torch.no_grad()
    def step(self, outputs, targets):
        outputs = outputs.detach().reshape(-1).tolist()
        targets = targets.detach().reshape(-1).tolist()
        
        for output, target in zip(outputs, targets):
            if (target == 1 and output > 0.5) or (target == 0 and output < 0.5):
                self.sum += 1
            
        self.num += len(outputs)
            
    def calc_accuracy(self):
        accuracy = self.sum / self.num
        
        return accuracy
    
    def reset(self):
        self.sum = 0
        self.num = 0