## Writeup

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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_heat_labels.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output_videos/project_video4_linear_q8_th5_focus_full.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 6th code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` and `YUV` color spaces and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters chose ones which gave me the best performance/accuracy balance of SVM classifier produced using them. At first, I tried to achieve the best accuracy possible, but after I waited for 1 hour to render my 50 seconds video on my old Macbook 2010, I started to search for some optimizations.

But the biggest finding was that SVM with RBF kernel was the slowest part of the pipeline and substituting it with LinearSVM gave me 3 fold speed increase.

HOG parameters was not as influential on the speed as SVM parameters. Of course when I used HOG features from only one channel, it also increased the speed, but accuracy dropped from 0.98 to 0.95 and I got a lot more false positives. And I decided to use all channels for HOG.

I tried set pixels per cell to 8, which gave almost no difference in accuracy, but had very big impact on speed.

2 cells per block gives the most accurate predictions compared to values 1 and 3. Performance is in the middle of them.

Also the most optimal orientations was empirically proven to be 11. Gives the most accurate model with not much performance loss.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the code cells 8-10 under the title "Train a Support Vector Machine" I trained the LinearSVM classifier using only HOG features in YUV color space. Initially I used SVM with RBF kernel, but for reasons described above, I moved to LinearSVM.

I used GridSearchCV to find the most optimal `C` parameter value of the LinearSVM model. The most accurate result was 0.984 with C=1.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the code cells under the title "Sliding window search" there are some functions for sliding window search, based on the code from the lectures.

I decided to search almost all sliding windows from line 400 (which is almost coincides with horison) and below. Because found that roofs of another cars are almost always on the same level regardless of their scale in the image. So, I used 2 small scale (1.0) windows, 5 medium windows (1.3, 1.4, 1.5), and 4 large windows (2.0, 2.5, 3.5). They was tuned manually to give the best performance possible on the test images.

My final search pattern:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 7 scales using YUV 3-channel HOG features. Here are some example images:

![alt text][image4]

I used heatmap with a threshold to filter most of the false positives. In video mode there is additional heatmap, which filters spurious bounding boxes from false positives.

![alt text][image5]

Also I extracted some frames with false negatives and tuned sliding windows positions on them, like pictures test7.jpg and test8.jpg in the `output_images/` folder.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video4_linear_q8_th5_focus_full.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the section `Final video pipeline` of my IPython notebook, there is class `Detector()` which has function `process_frame()` which implements all of the pipeline from feature extraction through heatmap filtering.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  I store bounding boxes from the previous 8 frames of the video and perform additional heatmap integration and thresholding so that a car certainly detected only when it was detected on 5 frames out of the last 8.

I also added focused search to improve detection quality. It looks for the bounding boxes from the last frame and performs additional sliding window search in that regions with similar scale.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I followed the suggested pipeline from the lecture. I took the dataset of cars and non-cars images, shuffled and split it in train and test datatsets. Then I converted images to YUV color space and extracted HOG features from each channel. After that I used sliding window technique to detect cars on the video. Then I implemented time smoothing using heatmaps.

Everything worked like it should, but performance of HOG and SVM on my old hardware is not so good. And it is especially annoying because to test a simple idea sometimes you need to wait dozen of minutes.

The classical machine learning and computer vision approaches requires a lot of hand fine tuning and I think that neural networks would perform a lot better and faster in such scenario.

The pipeline will likely to fail in situations of low contrast of a car with a background or when there is some unusual car, which was not in the training dataset.
