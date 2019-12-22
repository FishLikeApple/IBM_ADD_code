from helpers_and_configurations import *
from output_merging import output_merging
from rules import *
import argparse
import requests
from bs4 import BeautifulSoup

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)

    return parser.parse_args()

def run():
          
    # get args
    args = parse_args()

    # unzip and load models
    header = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.\
            0.3538.25 Safari/537.36 Core/1.70.3722.400 QQBrowser/10.5.3751.400'}
    url = 'https://drive.google.com/uc?export=download&confirm=pvHZ&id=1tXez97ZG4ElFTRGVS8ZbTFv8nCtMllqQ'
    r = requests.get(url=url, headers=header)
    header['cookie'] = r.headers['set-cookie']
    soup = BeautifulSoup(r.text, "lxml")
    r = requests.get('https://drive.google.com'+soup.select('#uc-download-link')[0]['href'], headers=header) 
    with open("models.zip", "wb") as code:
        code.write(r.content)
    unzip_single('models.zip', 'models/')
    import Model1, Model2, Model3, Model4

    # get input of all the models
    image, depth = get_image_and_depth(args.input, 'monodepth2/')
    image_to_show = imread(args.input)

    # get predictions of all the models
    model1_prediction = Model1.inference(image, image_to_show.shape)
    model2_prediction = Model2.inference(image, depth, image_to_show.shape)
    model3_prediction = Model3.inference(image, image_to_show.shape)
    model4_prediction = Model4.inference(image, depth, image_to_show.shape)

    # merge all the predictions
    prediction1_3 = output_merging(model1_prediction, model3_prediction)
    prediction2_4 = output_merging(model2_prediction, model4_prediction)
    final_prediction = output_merging(prediction1_3, prediction2_4)
    
    # apply rules
    warning_coords = rule1(final_prediction)
    warning_coords = rule2(final_prediction, warning_coords)
    warning_coords = rule3(final_prediction, warning_coords)
    warning_coords = rule4(final_prediction, warning_coords)
    
    # output the result
    cv2.imwrite(args.output, visualize(image_to_show, warning_coords)[:, :, ::-1])

if __name__ == '__main__':
    run()
