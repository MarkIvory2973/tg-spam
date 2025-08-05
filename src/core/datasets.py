from . import *

num_workers = os.cpu_count()

class Dataset_TGSpam(Dataset):
    def __init__(self, root, train):
        super().__init__()
        
        self.dataset = []
        
        if train:
            path = os.path.join(root, "train.csv")
        else:
            path = os.path.join(root, "eval.csv")
        with open(path) as file:
            lines = file.readlines()[1:]
            
        for line in lines:
            input, target = line[2:-1], line[0]
            input, target = self.tokenize(input), int(target)
            
            self.dataset.append([input, target])
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        
        inputs = self.add_padding(inputs)
        
        inputs, targets = torch.tensor(inputs), torch.tensor(targets, dtype=torch.float)
        
        return inputs, targets
            
    def tokenize(self, input):
        output = []
        for char in input:
            if self.in_vocab(char):
                char = ord(char)
            else:
                char = 1
            
            output.append(char)
            
        return output
        
    def in_vocab(self, char):
        if char.isascii() and (char.isalpha() or char.isdigit()):
            return True
        elif "\u4e00" <= char <= "\u9fff":
            return True
        elif unicodedata.category(char).startswith("P"):
            return True
        else:
            return False
        
    def add_padding(self, inputs):
        inputs_lens_max = 0
        for input in inputs:
            input_len = len(input)
            
            if input_len > inputs_lens_max:
                inputs_lens_max = input_len
                
        outputs = []
        for input in inputs:
            output = input + (inputs_lens_max - len(input)) * [0]
            
            outputs.append(output)
            
        return outputs
    
def load(root, train, batch_size):
    dataset = Dataset_TGSpam(root, train)
    dataset = DataLoader(
        dataset,
        batch_size,
        train,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    
    return dataset