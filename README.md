# IBM_ADD_code

# Environment
Python3, git, pip

# How To Use
git clone https://github.com/FishLikeApple/IBM_ADD_code
git clone https://github.com/nianticlabs/monodepth2
pip install -r IBM_ADD_code/requirements.txt
pip install -r monodepth2/requirements.txt
kaggle datasets download -d a1850961785/models-of-ibm-add
python IBM_ADD_code/run_sample.py --input IBM_ADD_code/test_sample.jpg --output warning_image.jpg

%warning_image.jpg is the output image. You can also use another image as the input.%
