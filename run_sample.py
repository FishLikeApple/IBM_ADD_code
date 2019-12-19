from helpers import *
import requests 

# download, unzip and load models
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/ 78.0.3904.108 Safari/537.36'}
url = 'https://www.kaggle.com/dataset/965835a48e203aa402d44dc3278357247749ba78497bdb7fd06dfa72ec9d3a98/download'
r = requests.post(url=url, allow_redirects=False)
print(r.headers)
r = requests.get(r.headers['location']) 
with open("models-of-ibm-add.zip", "wb") as code:
      code.write(r.content)
unzip_single('models-of-ibm-add.zip', 'models/')
import Model1, Model2, Model3, Model4

# get input of all the models
image, depth = get_image_and_depth('test_sample.jpg', 'monodepth2/')

# get predictions of all the models
model1_prediction = Model1.inference(image)
model2_prediction = Model2.inference(image, depth)
model3_prediction = Model3.inference(image)
model4_prediction = Model4.inference(image, depth)

# merge all the predictions
#prediction = 
