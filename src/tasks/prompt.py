from . import *

def tokenize(input):
    output = []
    for char in input:
        if in_vocab(char):
            char = ord(char)
        else:
            char = 1
        
        output.append(char)
        
    return output

def in_vocab(char):
    if char.isascii() and (char.isalpha() or char.isdigit()):
        return True
    elif '\u4e00' <= char <= '\u9fff':
        return True
    elif unicodedata.category(char).startswith('P'):
        return True
    else:
        return False

@torch.no_grad()
def run(root, epoch, input):
    ckpt = torch.load(os.path.join(root, f"ckpt/tg-spam/ckpt.{epoch}.pth"), device)
    
    model_c = models.Model_C().to(device)
    model_c.load_state_dict(ckpt["models"]["c"])
    
    ckpt = None
    
    model_c.eval()
    
    input = torch.tensor(tokenize(input)).unsqueeze(0)
    
    output = model_c(input)
    
    print(f"Spam probability is {output.cpu().item()*100:.1f}%")
