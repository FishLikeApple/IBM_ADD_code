from helpers import *
import requests 

# unzip and load models
if(platform.system()=='Windows'):
    os.system('type IBM_ADD_code/models/models.z* > models.zip')
else:
    os.system('cat IBM_ADD_code/models/models.z* > models.zip')
    
unzip_single('models.zip', 'models/')
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
