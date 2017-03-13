import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#import matplotlib
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import pickle
from moviepy.editor import VideoFileClip
from collections import deque
#matplotlib.rcParams.update({'font.size': 12})

def convert_color(image, cspace='RGB'):
    if cspace != 'RGB':
        if cspace == 'HSV':
            conv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            conv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            conv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            conv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            conv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: conv_image = np.copy(image)

    return conv_image
    
    
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel() 
    return features
                        
                        
# Define a function to compute color histogram features 
# TODO: NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):  
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    
          
# Define a function to return HOG features and visualization
# Works only on a single channel
def get_hog_features(ch, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(ch, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=False)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(ch, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, spatial_size=(32, 32),
                        hist_bins=32, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        feature_image = convert_color(image, cspace)    
        file_features = []
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)                             
            
            # Append the new feature vector to the features list
            file_features.append(hog_features)
    
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
    
        #Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
    
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def train_classifier():
    t=time.time()
    car_features = extract_features(cars, colorspace, orient, spatial_size, hist_bins, pix_per_cell, cell_per_block, hog_channel)
    notcar_features = extract_features(notcars, colorspace, orient, spatial_size, hist_bins, pix_per_cell, cell_per_block, hog_channel)

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract All features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()

    svc.fit(X_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    dist_pickle = {"svc":svc, "X_scaler": X_scaler}
    pickle.dump(dist_pickle, open("svc_pickle.p", "wb"))

    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace='YCrCb'):
    
    draw_img = np.copy(img)
    #TODO :Confirm this
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace)
     
    #conv_image = convert_color(img, cspace)
    #ctrans_tosearch = conv_image[ystart:ystop,:,:]

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    bbox_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat_ch = {}
            hog_feat_ch["1"] = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat_ch["2"] = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat_ch["3"] = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            
            if hog_channel == "ALL":
                hog_features = np.hstack((hog_feat_ch["1"], hog_feat_ch["2"], hog_feat_ch["3"]))
            else:
                hog_features = hog_feat_ch[str(hog_channel+1)]

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction

            if spatial_feat and hist_feat and hog_feat:
                test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))    
            elif spatial_feat and hist_feat:
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features)).reshape(1, -1))    
            else:
                test_features = X_scaler.transform(np.array(hog_features).reshape(1, -1))
            
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img, bbox_list


def process_frame(image):
    out_img, bbox_list = find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace)

    bbox_deque.append(bbox_list)
    bbox_img_deque.append(out_img)
    
    heat_avg = np.zeros_like(image[:,:,0]).astype(np.float)

    i = 0  
    for (bbox_deque_i, bbox_img_deque_i) in zip(bbox_deque, bbox_img_deque):
            
        #heat_frame = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        #heat_frame = add_heat(heat_frame, bbox_deque_i)
        heat_avg = add_heat(heat_avg, bbox_deque_i)
        
        #fig1 = plt.figure(1)
        #plt.subplot(num_frames_to_avg, 2, 2*i+1)
        #plt.imshow(bbox_img_deque_i)
        #plt.title('Bounding boxes')
        #plt.subplot(num_frames_to_avg, 2, 2*i+2)
        #heatmap = np.clip(heat_frame, 0, 255)
        #plt.imshow(heatmap, cmap='hot')
        #plt.title('Heat Map')
        i += 1
        
    # Apply threshold to help remove false positives
    heat_avg = apply_threshold(heat_avg, heat_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat_avg, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)     
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    #fig2 = plt.figure(2)
    #plt.imshow(labels[0], cmap='gray')
    #fig2 = plt.figure(3)
    #plt.imshow(draw_img)

    return draw_img


def test_static():
     
    # Generate a random index to look at a car/not-car image
    ind_car = np.random.randint(0, len(cars))
    ind_not_car = np.random.randint(0, len(notcars))
    # Read in the image
    car_image = mpimg.imread(cars[ind_car])
    not_car_image = mpimg.imread(notcars[ind_not_car])
                           
    # Call our function with vis=True to see an image output
    car_feat_image = convert_color(car_image, colorspace)
    nchan = car_feat_image.shape[2]
    car_hog_image = np.empty_like(car_feat_image)
    for ch in range(nchan):
        _, car_hog_image[:,:,ch] = get_hog_features(car_feat_image[:,:,ch], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    
    
    not_car_feat_image = convert_color(not_car_image, colorspace)
    not_car_hog_image = np.empty_like(not_car_feat_image)
    nchan = not_car_feat_image.shape[2]
    for ch in range(nchan):
        _, not_car_hog_image[:,:,ch] = get_hog_features(not_car_feat_image[:,:,ch], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    
    # Plot the examples
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(car_image)
    plt.title('Car image')
    plt.subplot(1,2,2)
    plt.imshow(not_car_image)
    plt.title('Not car image')

    fig = plt.figure()
    nchan = car_hog_image.shape[2]
    for ch in range(nchan):
        plt.subplot(nchan, 4, 4*(ch)+1)
        plt.imshow(car_feat_image[:,:,ch], cmap='gray')
        plt.title('Car CH-' + str(ch) + ' in ' + colorspace)
        plt.subplot(nchan, 4, 4*(ch)+2)
        plt.imshow(car_hog_image[:,:,ch], cmap='gray')
        plt.title('HOG of ' + 'Car CH-' + str(ch) + ' in ' + colorspace)

        plt.subplot(nchan, 4, 4*(ch)+3)
        plt.imshow(not_car_feat_image[:,:,ch], cmap='gray')
        plt.title('not-Car CH-' + str(ch) + ' in ' + colorspace)
        plt.subplot(nchan, 4, 4*(ch)+4)
        plt.imshow(not_car_hog_image[:,:,ch], cmap='gray')
        plt.title('HOG of ' + 'not-Car CH-' + str(ch) + ' in ' + colorspace)

    image = mpimg.imread('test_images/test3.jpg')
    
    out_img, bbox_list = find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)        
        
    fig = plt.figure()
    plt.imshow(out_img)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()    
    plt.show() 
 

def test_video():
  video_output = 'output_images/vehicles_detected.mp4'
  clip1 = VideoFileClip("../project_video.mp4")
  input_clip = clip1.fl_image(process_frame)
  input_clip.write_videofile(video_output, audio=False)


# Read in our vehicles and non-vehicles
cars = glob.glob('vehicles/vehicles/**/*.png', recursive=True)
notcars = glob.glob('non-vehicles/non-vehicles/**/*.png', recursive=True)

### TODO: Reduce the sample size because HOG features are slow to compute
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
hog_feat = True
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
y_start = 400 # Min and max in y to search in slide_window()
y_stop = 720
scale = 1.5
heat_threshold = 10  # For single image, it should be low value ~1, for video frames, it can be high value ~5
num_frames_to_avg = 10

#train_classifier()  # <------- Comment or uncomment as required

dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
svc = dist_pickle["svc"]
X_scaler=dist_pickle["X_scaler"]

bbox_deque = deque([], num_frames_to_avg)
bbox_img_deque = deque([], num_frames_to_avg)
      
#test_static()
test_video()
#plt.show() 
