from helpers import *
import requests 

# download, unzip and load models
header = {'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
          'accept-encoding': 'gzip, deflate, br',
          'accept-language': 'zh-CN,zh;q=0.9',
          'cookie': 'GCLB=CPr7yuKmsrXi7wE; _ga=GA1.2.894895260.1574243575; ka_sessionid=b56807965a369e951aa92d\
          721ae270309b52bfbe; CSRF-TOKEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL_Zzcku-1wTLjTFRkGcqkR88-WFW4mkM4eTT1aZoC15\
          y7o2nX2ewjSwoJ7sDC-USdpYyaZy9ksqPDs_7tExO3Kn8laNC8Z1CSvZYuvKCXHXMUmMfW4gRPrppamgcmDGtEU; _gid=GA1.2\
          .1709354907.1575516997; .ASPXAUTH=F322635B4C2F17C6946B0718DFF5537757A8A64549521FFD5086B37278BD9CEAC\
          2ACD79C0A4C927529A1AEFF2959D40FFE77859A008DC1ACF8F4D0A8254047C9BF8352771A3182638C7A7ACEB5BF8C5674A7D\
          5BD; CLIENT-TOKEN=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpc3MiOiJrYWdnbGUiLCJhdWQiOiJjbGllbnQiLCJzd\
          WIiOiJhMTg1MDk2MTc4NSIsIm5idCI6IjIwMTktMTItMTlUMDU6NTM6MDIuOTYwMDM2MVoiLCJpYXQiOiIyMDE5LTEyLTE5VDA1O\
          jUzOjAyLjk2MDAzNjFaIiwianRpIjoiN2JiZWM2Y2MtNzBhZC00MTg0LTk1YTgtODFhZWU0YmE0NzRhIiwiZXhwIjoiMjAyMC0wMS\
          0xOVQwNTo1MzowMi45NjAwMzYxWiIsInVpZCI6MjE2OTk0MiwiZmYiOlsiRmxleGlibGVHcHUiLCJLZXJuZWxzSW50ZXJuZXQiLCJ\
          EYXRhRXhwbG9yZXJWMiIsIkRhdGFTb3VyY2VTZWxlY3RvclYyIiwiS2VybmVsc1ZpZXdlcklubmVyVGFibGVPZkNvbnRlbnRzIiw\
          iRm9ydW1XYXRjaERlcHJlY2F0ZWQiLCJVdGlsaXR5U2NyaXB0cyIsIk5ld0tlcm5lbFdlbGNvbWUiLCJNZGVJbWFnZVVwbG9hZGVy\
          IiwiRGlzYWJsZUN1c3RvbVBhY2thZ2VzIiwiUGluT3JpZ2luYWxEb2NrZXJWZXJzaW9uIiwiUGhvbmVWZXJpZnlGb3JHcHUiLCJDb\
          G91ZFNlcnZpY2VzS2VybmVsSW50ZWciLCJVc2VyU2VjcmV0c0tlcm5lbEludGVnIiwiTG9naW5NaWdyYXRlRGVwcmVjYXRlZFNzby\
          JdfQ.; intercom-session-koj6gxx6=OHpIbUlPb05EVGRHVEJJUStzNVlpeS85RW5SRkVSRW9GM3Z0OTNDOUFHV1ZheWFKc3Z4R\
          Th3NURST3BkOGM1cy0taHNBeGovdjBoRWZXTC9jWTgzSlZqUT09--852305c984b463a10f8473c3a16064000a5c4466; XSRF-TO\
          KEN=CfDJ8LdUzqlsSWBPr4Ce3rb9VL8UmD9Z0qAV3Ll9GFh1lYjbfx7WZZ7AkOVjxLdnogpB0GMeAgQ4Pw3jP9p92FTi-LFUf8G57j\
          J6ioYczuI821C2pRVURBaEeWicMtwXZKlK26Y7wqLHzoHFSzG6j07aRiH4hzM7N8_6EFzrVcrIXsDC5zZCs_OucOVsH-11iGKlvA; \
          .AspNetCore.Mvc.CookieTempDataProvider=CfDJ8LdUzqlsSWBPr4Ce3rb9VL9EHrnLzm6JCzi4BJ414dMfjdkirT2wwocamrv\
          F_-EHXLrox-0o_MlElDF0i-5Z1YvTNRxJqdkiM4OgHV6InBu83PJzVGHSakAgHCZS0EslgsLbOoxB54VptlzdC8TydKc',
          'upgrade-insecure-requests': '1',
          'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0\
          .3538.25 Safari/537.36 Core/1.70.3732.400 QQBrowser/10.5.3819.400'}
url = 'https://www.kaggle.com/dataset/965835a48e203aa402d44dc3278357247749ba78497bdb7fd06dfa72ec9d3a98/download'
r = requests.get(url=url, headers=header, allow_redirects=False)
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
