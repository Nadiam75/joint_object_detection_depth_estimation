# joint_object_detection_depth_estimation
## Deep Learning Final Project     
#### Direct access to the [Project Report](https://github.com/Nadiam75/joint_object_detection_depth_estimation/blob/main/Project_Report.pdf)

#### Created by: Zahra Meskar , Mohsen Shirkarami 

In this project we combined two neural networks to perform object recognition and depths estimation tasks simultaneously.

Official repository:            (https://github.com/Nadiam75/joint_object_detection_depth_estimation)

Pretrained Depth Estimation:    (https://github.innominds.com/karoly-hars/DE_resnet_unet_hyb)

Pretrained Yolo_V5:             (https://github.com/ultralytics/yolov5)

Pretrained Yolo_V2:             (https://pjreddie.com/darknet/yolov2)

Dataset Used to Train the Network (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html


![model](pictures/indoor.png)

## Usage

### 1) Requirements

- Python
- Pytorch
- Opencv-Python
- Matplotlib
- h5py
- PIL
- scipy
- tensorflow
- torchvision



### 2) Folders & Files


  [Project_Report.pdf](https://github.com/Nadiam75/joint_object_detection_depth_estimation/blob/main/Project_Report.pdf)
  
  Contains detailes on the implementation of the three structures implemented, our telegram bot and the webApp. 
  
  
  DL_Project.ipynb      
  
  joint object detection and depth estimation using pretrained YOLO_V5 and Pretrained Depth Estimation 
  
  DL_YOLOv2_ResnetUnetHybrid.ipynb     
  
  joint object detection and depth estimation using pretrained YOLO_V2  and Pretrained Depth Estimation
  
  trained_depth_yolo_v5        
  
  joint object detection and depth estimation using pretrained YOLO_V2  and trainin Depth Estimation on NYU dataset
  
  

### 3) Telegram Bot

Telgeram bot available at: @DL_Sharif_Project_bot

Our telegram bot is capabale of detecting objects and estimating their corresponding depths closer or farther from 2, 3 or 4 meters, this threshold can be changed according to the users needs!

![disp](pictures/7.jpg)

### 4) Web application

In order to install the dependencies run the following commands in shell

``` shell
pip install -r requirements.txt
```


To Start the Server Run the following command:
``` shell
python manage.py runserver
```
![disp](pictures/6.jpg)

### 3) Visualize result



Here are some of our training results on TEST DATASET (depth estimation model has been trained for 30 epochs on NYU):

<!-- // ![disp](pictures/indoor.png)
 -->

![disp](pictures/1.jpg)

<!-- ![disp](pictures/2.jpg)
 -->
![disp](pictures/3.jpg)

![disp](pictures/4.jpg)

![disp](pictures/5.jpg)


### 4) Video

A short video containing details on how to use the web application has also been uploaded.


![disp](pictures/Depth_Estimation_WEBAPP.gif)


## Contact

Email: nadia.meskar@yahoo.com, m.shirkarami@gmail.com

Welcome for any discussions! 

