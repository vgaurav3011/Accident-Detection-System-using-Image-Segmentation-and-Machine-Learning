# Accident Detection System using Image Segmentation and Machine Learning
<p align="justify">
Road accidents in India are a major cause of decreasing life expectancy with road accidents contributing to over 148,000 deaths out of 467,000 deaths in 2016. Indian Economy has a hit of 3 percent of GDP growth due to road accidents as per the United Nations with an estimated loss of $58,000 in terms of value every year. The metropolitan cities such as Chennai, Mumbai and New Delhi have been increasingly highlighted for lack of road safety and rash driving cases. The recent trends show that there has been an increase in the global number of road accidents even in developed countries. However,  under-developed and developing countries suffer a more significant impact due to life and economic losses. These accidents occur due to violation of traffic safety rules, careless rash driving, driver drowsiness and lack of good quality roads. The problem becomes more adverse for highways and hilly areas where accidents are unavoidable. Road accidents are characterized by high death rates due to delay in arrival of help and inefficient systems of mitigation to alert the concerned authorities. Road accidents on the highways are typically caused by natural reasons such as extreme weather conditions such as fog and consecutive collision of vehicles are common on Indian highways due to lack of visibility. The states of Maharashtra, Tamil Nadu and Uttar Pradesh account for the highest number of road accidents in India. The problem can be handled by making use of computer vision and low-cost sensor networks. The current solutions involved heavy dependency on sensor networks and area coverage. This can be substantially replaced by making use of object detection and image segmentation for accident classification. The system identifies the accident-prone areas which are the target stakeholders for the deployment and sets it apart from other implementations since it provides a feasibility factor associated with it. Furthermore, the system provides enhanced mitigation alert to the concerned authorities which helps in preventing any consecutive collisions that could possibly lead to greater loss of lives.
</p>
<br>The proposed system is as follows:
<p align="center">
  
 <img width="500" height="500" src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/block_diagram.jpg">
 </p>
<p align="justify">
  The first phase is to identify the accident prone areas across India with the data given from data.gov.in by Government of India which is pretty straighforward and linear by nature thus making linear regression suitable to predict the number of possible accidents that can occur in any state and identified Maharashtra, Tamil Nadu and Uttar Pradesh as the highest frequency of road accidents prone states.
 </p>
  <img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/download.png">
<p align="justify">
  The second phase is to perform identification of the cars in the image using image segmentation where we experimented with Fast RCNN, YOLO, canny edge detection, used watershed segmentation as preprocessing to improve the output of RCNN and finally made use of the Facebook masked RCNN with pre trained coco-weights that is used to obtain the final segmented output.
</p>
<img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/sample.jpg">
The YOLO algorithm misclassified in some cases and thus it was infeasible to use for deployment. The following is an output on video recorded in Bangalore Highway Airport Road.
<img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/yolo.png">
<p align="center">
<img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/edge_detection.png">
<img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/watershed.png">
</p>
<p align="center">
  The Fast RCNN output was improved with watershed segmentation.
  <img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/fast_rcnn.png">
 </p>
 <p align="center">
 The final output with masked RCNN is as follows:
 </p>
 <img src="https://github.com/vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning/blob/master/mask_rcnn_output.png">
 <p align="justify">
  The final phase included severity prediction of the accident based on various parameters provided ranging from weather conditions, lighting on road and coordinates provided. Random Forest Classifier was successfull compared to counterparts for this approach.
  This system was published as a research paper in International Journal of Engineering Research and Technology (IJERT) in Volume 8, Issue 10, October 2019. <br>
  Link to the paper: https://www.ijert.org/research/accident-detection-severity-prediction-identification-of-accident-prone-areas-in-india-and-feasibility-study-using-improved-image-segmentation-machine-learning-and-sensors-IJERTV8IS100164.pdf
  </p>
