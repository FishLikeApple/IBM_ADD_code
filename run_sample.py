from helpers import *
import requests 

# download and unzip models
url = 'https://storage.googleapis.com/kaggle-data-sets/448321/848002/bundle/archive.zip?GoogleAccessId=web\
      -data@kaggle-161607.iam.gserviceaccount.com&Expires=1577009087&Signature=VLIDNctwKUAU%2BRb3Hn80wT4Pe%2Bf7s\
      6mWn5tAwE%2Bo4QyP6SSSe9J%2BnTxF9U9KMexV8RYcvIsvq4AGbHX4vuMjEbT3S%2FelMLhS296GyDaIlnPmFS9VMImMb7TwWYVG527VN\
      axmgX0MRHifKJXKg%2BpFDQSzt6A8Z%2BvN1jckYEvkJ1g1vBqXxQJx4FLQS75uD9n8UlxFb9k8t%2BP3e9aq8054XQPa1vbXZtZ512h%2\
      Fm6DNR7Byx8Ouu1wfiQJFAKbUA7Vffg77MOsXezCG%2FVq3eNmcFE5qdrCxIMOzghBZ243BW9mNkgqL%2BVlaa%2B5EDCq0A3ZJiXHPSQG\
      fvYJuqKJMPp6%2FFg%3D%3D&response-content-disposition=attachment%3B+filename%3D965835a48e203aa402d44dc32783\
      57247749ba78497bdb7fd06dfa72ec9d3a98.zip' 
r = requests.get(url) 
with open("models-of-ibm-add.zip", "wb") as code:
      code.write(r.content)
unzip_single('models-of-ibm-add.zip', 'models/')


