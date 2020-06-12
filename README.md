# Information-Extraction-From-Unstructured-Data-Using-Deep-Learning


Need detail information about our project
1.	Software used

Python

2.	Database used :- MSCOCO 

3.	Accuracy measured methods/formulas

We have used Bleu and Cider metric to evaluate the image captioning results. You can get more insight on same in below paper (Please find the attachment in mail with following name)  Re-evaluating Automatic Metrics for Image Captioning.pdf

4.	Detailed steps of project functioning:

4.1	which server(college) we have used:-
 EL GPU Server ()
4.2	all installation steps in that server:-

1.	 User can log into the server using SSH.
command to connect to the server:
 ssh username@server_ip
2.	Python scripts can be submitted for execution using commands "pycompute27" or "pycompute35" which use Python2.7 and Python3.5 respectively.
3.	 Help for these commands are located in ~/Documents/help_compute.txt. The help file mentions MATLAB but the command syntax is same for Python
4.	 User should create their own virtual environment in their home directory and install required packages via pip.

4.4	need all steps to run our project from client environment:-

Creating work environment: 
It is recommended to create a virtual environment in python to run project 
to create virtual environment in python follow below steps:
Step1: install package
$ sudo apt install python3-venv
Step2: Create environment
$ python3 -m venv my-project-env
Step3: Activate environment
$ source my-project-env/bin/activate
Now You can install necessary packages/libraries in environment using simple pip command, Required libraries are mentioned below:
1.	Tensorflow 
2.	NumPy  
3.	OpenCV 
4.	Natural Language Toolkit (NLTK) 
5.	Pandas 
6.	Matplotlib 
7.	tqdm 

To install above package use pip command and make sure you are in same environment created in above steps:
$ pip3 install package_name
example: $ pip3 install tensorflow



1.	Preparation:
Download the COCO train2014 and val2014. Put the COCO train2014 images in the folder train/images, and put the file captions_train2014.json in the folder train. Similarly, put the COCO val2014 images in the folder val/images, and put the file captions_val2014.json in the folder val. Furthermore, download the pretrained  ResNet50 net and put in main project directory

Link to dataset: http://cocodataset.org/#download
Link to Resnet50 : https://app.box.com/s/17vthb1zl0zeh340m4gaw0luuf2vscne







2.	Train model:
command:

python main.py --phase=train \
    --load_cnn \
    --cnn_model_file='./restnet50_no_fc.npy'\
    --train_cnn

Resume training from checkpoint:
command:

python main.py --phase=train \
    --load \
    --model_file='./models/model_number.npy'\
    --train_cnn

3.	Evaluation: To get score of model
command:

python main.py --phase=eval \
    --model_file='./models/model_number.npy' \
    --beam_size=3

4.	Results:
 You can use the trained model to generate captions for images. Put such images in the folder test/images, and run below command:

python main.py --phase=test \
    --model_file='./models/model_number.npy' \
    --beam_size=3

The generated captions will be saved in the folder test/results








5.	future scope 
5.1	if you found any challenges/improvement in our project

We got average results in the topic modeling section, as LDA cannot give optimal results with short text.

5.2	if you suggest or thought any solutions of that challenges

To solve above problem we can use LSA algorithm and we also can use large corpus data to train algorithm on LDA itself.

6.	Program code files (send me in a compressed folder)
6.1	need all code/files 
6.2	steps to run/install  that code/files
Same as mentioned in 4.4 section above

6.3	Mentioned if any library to install and its procedure:- 
1.	Tensorflow 
2.	NumPy 
3.	OpenCV 
4.	Natural Language Toolkit (NLTK)   
5.	Pandas
6.	Matplotlib 
7.	Tqdm
Steps to install above libraries are mentioned in section 4.4

