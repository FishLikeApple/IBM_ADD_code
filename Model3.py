from helpers import *
from efficientnet_pytorch import EfficientNet

checkpoint = 'Model3.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyUNet(8).to(device)
model.load_state_dict(torch.load(checkpoint))

def inference(image):
  # the inference function of this model
  
  img = torch.tensor(image[None]).to(device)
  output = model(img).data.cpu().numpy()
  coords_pred = extract_coords(output[0])
  
  return coords_pred
