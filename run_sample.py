from helpers import *
import Model1, Model2, Model3, Model4
import requests 

# download and unzip models
url = 'https://www.kaggle.com/dataset/965835a48e203aa402d44dc3278357247749ba78497bdb7fd06dfa72ec9d3a98/download' 
r = requests.get(url) 
with open("models-of-ibm-add.zip", "wb") as code:
      code.write(r.content)
unzip_single('models-of-ibm-add.zip', 'models/')

# get input of all the models
image, depth = get_image_and_depth('test_sample.jpg', 'monodepth2/')

# get predictions of all the models
model1_prediction = Model1.inference(image)
model2_prediction = Model2.inference(image, depth)
model3_prediction = Model3.inference(image)
model4_prediction = Model4.inference(image, depth)

# merge all the predictions
#prediction = 
