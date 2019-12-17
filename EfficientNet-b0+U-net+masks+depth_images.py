use_depth = True
from helpers import *
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = MyUNet(8).to(device)

def inference(image, depth):
