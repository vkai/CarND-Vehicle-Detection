**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/car.png
[notcar]: ./output_images/notcar.png
[car_hog]: ./output_images/car_hog.png
[notcar_hog]: ./output_images/notcar_hog.png
[detection1]: ./output_images/detection1.png
[detection2]: ./output_images/detection2.png
[detection3]: ./output_images/detection3.png
[detection4]: ./output_images/detection4.png
[heatmap]: ./output_images/heatmap.png
[scale_1.5]: ./output_images/scale_1.5.png
[scale_1]: ./output_images/scale_1.png

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 51 through 99 of the file called `lesson_functions.py`.

I started by reading in all the `vehicle` and `non-vehicle` images. Examples of the images are shown below.

![car][car] ![notcar][notcar]

First I converted each image from `RGB` space to `YCrCb` color space. I then explored different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the output looks like. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![car_hog][car_hog] ![notcar_hog][notcar_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` to provide the best predictions with reasonable performance. With `pixels_per_cell=(8, 8)` I achieved some better predictions, but my performance suffered too much to be useful.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the `Train Classifier` section of `pipeline.ipynb`. I used spatial, color, and HOG features to train my model and predict images. This resulted in a feature vector length of 1788 when training. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is illustrated in the `Sliding Windows` section of `pipeline.ipynb`. With the test images, I experiemented with various scales of windows to search. I found that a scale of 1 (64, 64) and 1.5 (96, 96) were enough to perform well in detecting the vehicles within a modest distance. I overlapped the windows by 0.5 in both x and y directions, finding this to perform sufficiently.

![scale_1][scale_1] ![scale_1.5][scale_1.5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To improve performance, I limited my search to a narrow patch of the image where the road would appear, `y=(400,600)`. I was also able to only use two scales to limit the number of windows searched. Because extracting HOG features was performance intensive, I extracted the HOG features for the image patch once and sub-sampled the HOG features for each window. This improved performance from simply performing a HOG extraction on each window.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![detection1][detection1] ![detection2][detection2] ![detection3][detection3] ![detection4][detection4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a YouTube [link to my video result](https://youtu.be/tX1sDJcR0ew). The original video is also available as `video.mp4` in this repository.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I kept track of the previous 30 frames of positive detections. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. This reduced the number of false positives significantly. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected, based on the minimum and maximum values of the individual detected positions.

Here's an example result showing the heatmap from a test image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![heatmap][heatmap]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My model is currently only trained on well lit images of cars and would likely struggle on roads with many shadows or other unfavorable lighting conditions. I would likely need to supplement the feature vector with another color space like HLS.

There is also one point in the video where the view of the white vehicle is at a strong angle. Most of the side view images in the training set are not at such a strong angle. I believe the training set needs to be supplemented with images at this angle in order for the classifier to detect these situations.

My pipeline also takes a while to process and is nowhere near real time. On my entry level Macbook Pro the project video took 8 minutes to complete. I could possibly improve performance by skipping every other frame for extracting HOG features because a vehicle would not move in and out of view that quickly. More specialized hardware would also significantly improve feature extraction performance.