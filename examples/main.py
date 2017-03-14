# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 00:21:56 2017

@author: Masoud
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from collections import deque
from moviepy.editor import VideoFileClip
import pickle




import myUtils


# Read training and testing data
cars = glob.glob('../data/vehicles/**/*.png', recursive=True)
notcars = glob.glob('../data/non-vehicles/**/*.png', recursive=True)


#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]


color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 720] # Min and max in y to search in slide_window()
ystart,  ystop = y_start_stop
scale = 1.5
heat_threshold = 10  # For single image, it should be low value ~1, for video frames, it can be high value ~5
num_frames_to_avg = 10


car_features = myUtils.extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = myUtils.extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)



X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# to save memory
del car_features, notcar_features


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Using SVM for training 
svc = LinearSVC()

t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

dist_pickle = {"svc":svc, "X_scaler": X_scaler}
pickle.dump(dist_pickle, open("model_svc_YCrCb_NoHist.p", "wb"))
    
    

# Test on a sample image
############################################################

image = mpimg.imread('../test_images/test7.png')
#img = image
img = image.astype(np.float32)/255

ystart = 400
ystop = 720
scale = 2.6


out_img, bbox_list = myUtils.find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, color_space, spatial_feat, hist_feat, hog_feat)
              

num_frames_to_avg = 1
bbox_deque = deque([], num_frames_to_avg)
bbox_img_deque = deque([], num_frames_to_avg)
                         
bbox_deque.append(bbox_list)
bbox_img_deque.append(out_img)
    
heat_avg = np.zeros_like(image[:,:,0]).astype(np.float)
 
i = 0  
for (bbox_deque_i, bbox_img_deque_i) in zip(bbox_deque, bbox_img_deque):        
    
    # Add heat to each box in box list
    heat_avg = myUtils.add_heat(heat_avg, bbox_deque_i)    
    """
    fig1 = plt.figure(1)
    plt.subplot(num_frames_to_avg, 2, 2*i+1)
    plt.imshow(bbox_img_deque_i)
    plt.title('Bounding boxes')
    plt.subplot(num_frames_to_avg, 2, 2*i+2)
    heatmap = np.clip(heat_avg, 0, 255)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    """
    i += 1
        
# Apply threshold to help remove false positives
heat_avg = myUtils.apply_threshold(heat_avg, heat_threshold)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat_avg, 0, 255)
plt.imshow(heatmap, cmap='hot')
# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = myUtils.draw_labeled_bboxes(np.copy(image), labels)

fig2 = plt.figure(2)
plt.imshow(labels[0], cmap='gray')

fig2 = plt.figure(3)
plt.imshow(draw_img)
    
     
plt.imshow(out_img)




scale = 1.4
heat_threshold = 10  # For single image, it should be low value ~1, for video frames, it can be high value ~5

num_frames_to_avg = 10
bbox_deque = deque([], num_frames_to_avg)
bbox_img_deque = deque([], num_frames_to_avg)
                   
def detectCars(image):
    out_img, bbox_list = myUtils.find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, color_space, spatial_feat, hist_feat, hog_feat)

    bbox_deque.append(bbox_list)
    bbox_img_deque.append(out_img)
    
    heat_avg = np.zeros_like(image[:,:,0]).astype(np.float)

    i = 0  
    for (bbox_deque_i, bbox_img_deque_i) in zip(bbox_deque, bbox_img_deque):                
        heat_avg = myUtils.add_heat(heat_avg, bbox_deque_i)        
        i += 1
        
    # Apply threshold to help remove false positives
    heat_avg = myUtils.apply_threshold(heat_avg, heat_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat_avg, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)     
    draw_img = myUtils.draw_labeled_bboxes(np.copy(image), labels)
    
    
    return draw_img
"""
dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler=dist_pickle["X_scaler"]
"""
video_output = '../output_images/project_video_result_4_14_10_10.mp4'
clip1 = VideoFileClip("../project_video.mp4")
input_clip = clip1.fl_image(detectCars)
input_clip.write_videofile(video_output, audio=False)   

