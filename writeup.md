##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.jpg
[image2]: ./output_images/hog_example.jpg

[image4a]: ./output_images/windows_example1.jpg
[image4b]: ./output_images/windows_example2.jpg
[image4c]: ./output_images/windows_example3.jpg
[image4d]: ./output_images/windows_example4.jpg

[image5]: ./output_images/bboxes_and_heat.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/output_bboxes.jpg
[video1]: .output_images/vehicles_detected.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features()` in lines #55 through #68 of the file called `vehicle_detection.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]  

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out that extracting HOG features on YCrCb format produced close to 100% accurate test results. Other parameters like `orientation`, `pixels_per_cell` and `cells_per_block` were also found by hit and trial on the test images. The values selected produced decent bounding boxes. 

I also extracted color features, but in practice, they didn't affect the results considerably. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The function `train_classifier()` from line #113 through #152 performs the task of training and SVM classifier using HOG and color features. It extracts the features for the two classes of cars and not-cars, splits the dataset into training and test set (80%-20%), and trains the classifer using Python's `sklearn.svm` library.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window search is implemented in the function `find_cars()` in the nested for-loop starting at line #228, which basically slides a sliding window over an image of HOG features and crops out a patch with HOG features, instead of calculating HOG features again and again for each patch. The code currently uses only a fixed size window of size 64x64 pixels scaled down to 1.5 times, with 75% overlap, which showed decent performance on test images.  A higher scale would lead to smaller window sizes, which would miss near vehicles as they appear bigger, so need a larger window to capture features. A lower scale value would mean bigger window size, and inaccurate capturing of features from farther vehicles. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on a fixed scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image4a]
![alt text][image4a]  
![alt text][image4a]
![alt text][image4a]  

To be mroe efficient, the pipeline only searches in the lower half of the frame, where vehicles can exist, and crops out anything above the horizon. Also, as mentioned before, it calculates the HOG features of the whole frame once and then crops out patches corresponding to the sliding window, instead of calculating features for each window. 

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/vehicles_detected.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The heatmap method basically filters out the results both spatially and temporally. For all the pixels with overlapping bounding boxes within a frame, the heatmap is incremented. Also, within consecutive `num_frames_to_avg` number of frames, heatmap is updated for overlapping boxes. This way, false positives are eliminated, as features will need to be detected in a certain number of consecutive frames to be regarded as a vehicle. Similarly, within a frame, overlaping bounding boxes should share common features to be regarded as a vehicle.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Currently the pipeline uses a fixed size window. To be more robust, it should be variable: larger near the bottom of the frame and smaller near the horizon.  
Another issue is that when two vehicles are closer together, it detects them as one big vehicle, as the sliding window gets features from both the vehicles, and it is not smart enough to know which features correspond to which vehicle.  
As with all low pass filters, increasing the number of frames over which heatmap is created will get rid of false positives and stabilize the bounding boxes, but it also introduces a delay. It will also fail in situations where the relative speed between our vehicle and the other vehicle is high, due to which there won't be sufficient number of overlapping windows within frames. 

