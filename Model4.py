from helpers import *
from efficientnet_pytorch import EfficientNet

checkpoint = 'Model4.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyUNet(8, use_depth=True).to(device)
model.load_state_dict(torch.load(checkpoint))

def inference(image, depth):
  # the inference function of this model
  
  img = torch.tensor(image[None]).to(device)
  depth = torch.tensor(depth[None]).to(device)
  output = model(img, depth).data.cpu().numpy()
  coords_pred = extract_coords(output[0])
  
  return coords_pred
