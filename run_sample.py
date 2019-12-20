from helpers import *
import requests 

# unzip and load models
header = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
          'accept-encoding': 'gzip, deflate, br',
          'accept-language': 'zh-CN,zh;q=0.9',
          'cookie': 'download_warning_13058876669334088843_1tXez97ZG4ElFTRGVS8ZbTFv8nCtMllqQ=pvHZ; CONSENT=YES\
          +US.zh-CN+201908; SEARCH_SAMESITE=CgQIg44B; S=billing-ui-v3=2oZy5qyzrnHktYxz0UgTnRJ_C1SmEcG2:billing\
          -ui-v3-efe=2oZy5qyzrnHktYxz0UgTnRJ_C1SmEcG2; ANID=AHWqTUmvgk4SJQ7Ha1Oe2O-acwDdPbJu7HxKpgOQKAXUVtfqJf\
          gr_hyri0HkrgW4; NID=194=O8KRLMSs2wBPYYLtyMLbr3dUx8WEy8ibzRDPdvDIvUAn1s3a2uBpXz2v1MeMW5Ig9CI3ninHqxsX\
          Q4HPyQcToU4GPDtV-U8BxdGyCVbO6DzN1YkV6uacom3Dd6KofFojBINZInyoDfvIhQmCfxEa-iIfvJFVBf3RHHHgfDVI3aBniGQh\
          Mbv6j7TyQQlYWl5WrQ57VmQeBXIg1UhcPJGQhVPCIsGWtiRV_5Iq1_-Qau2MF3dZgW4ZAWc; 1P_JAR=2019-12-20-5',
          'upgrade-insecure-requests': '1',
          'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.\
          0.3538.25 Safari/537.36 Core/1.70.3722.400 QQBrowser/10.5.3751.400'}
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
