from helpers import *
import requests 

# unzip and load models
header = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0\
          .3538.25 Safari/537.36 Core/1.70.3732.400 QQBrowser/10.5.3819.400'}
url = 'https://drive.google.com/uc?export=download&confirm=pvHZ&id=1tXez97ZG4ElFTRGVS8ZbTFv8nCtMllqQ'
r = requests.post(url=url, headers=header, allow_redirects=False)
print(r.headers)
r = requests.get(r.headers['location']) 
with open("models.zip", "wb") as code:
      code.write(r.content)
unzip_single('models.zip', 'models/')
import Model1, Model2, Model3, Model4

# get input of all the models
image, depth = get_image_and_depth('IBM_ADD_code/test_sample.jpg', 'monodepth2/')

# get predictions of all the models
model1_prediction = Model1.inference(image)
model2_prediction = Model2.inference(image, depth)
model3_prediction = Model3.inference(image)
model4_prediction = Model4.inference(image, depth)

# merge all the predictions
#prediction = 
