# Vehicle Detection Project

| Note    | |
|:-----------|:-------------|
| **Source Code**  | https://github.com/aurangzaib/CarND-Vehicle-Detection  |
| **Car training data**  | https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip      |
| **Not-car training data**  | https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip      |
| **How to run**  | `cd implementation && python main.py`      |

The steps of the project are the following:

- Perform a Histogram of Oriented Gradients (`HOG`), Color Transform and Spatial Bining to extract features on a labeled training set of images.

- Randomize and normalize the features and train a `SVM` classifier.

- Implement a sliding-window technique with `HOG` sub-sampling and use the trained classifier to search for vehicles in images by predicting the labels for each feature.

- Create a `heat map` of recurring detections.

- Remove `false positives` and Update the bounding boxes for vehicles detected.

---

### 1-	Feature Extraction

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/feature_extraction.py`  |
| Method  | `FeatureExtraction.bin_spatial`      |
| Method  | `FeatureExtraction.color_hist`      |
| Method  | `FeatureExtraction.get_hog_features`      |

Here are a few samples of vehicle and non-vehicle training datasets:

![alt text](./documentation/vehicles.png)

![alt text](./documentation/non-vehicles.png)

The algorithm is as follows:

-	Reading in all the `vehicle` and `non-vehicle` images.
-	For `Spatial Bining`, we resize the image to `32x32` and use numpy `ravel` for each color channel to get features vector.
-	For `Color Histogram`, we use numpy `histogram` for each channel and concatenate the result.
-	For `HOG` features, skimage `hog` is used with predefined following parameters.

The combination of parameters are found after running a battery of tests on small datasets. Parameters are selected for best balance of time and prediction accuracy.

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/classifier.py`  |
| Method  | `Classifier.evaluate_classifier_parameters`      |
| Car training data  | https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip      |
| Not-car training data  | https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip      |

```python
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

clf = SVC(kernel='rbf', C=10)

t_start = int(time.time())

clf.fit(x_train, y_train)
t_end = int(time.time())

t_start_test = int(time.time())
y_predicted = clf.predict(x_test)
t_end_test = int(time.time())
score = accuracy_score(y_test, y_predicted)

print("train: {}s, test: {}s, accuracy: {}%".format(t_end - t_start,
                                                    t_end_test - t_start_test,
                                                    score * 100))
```

Following are the results of the optimal parameters search:

| Orient    |Channel  |Colorspace  | Accuracy (%)  |Train time (s) |Predict time (s)  |
|:-----------|:-------------|:-------------|:-------------|:-------------|:-------------|
| 08  | R  |RGB  |98.06  |10  |2  |
| 10  | R  |RGB  |98.7  |11  |3  |
| 10  | ALL  |RGB  |99.1  |22  |6  |
| 10  | Y  |YCrCb  |98.49  |10  |4  |
| 10  | ALL  |YCrCb  |100  |16  |4  |
| 10  | ALL  |LUV  |99.56  |16  |4  |
| 10  | ALL  |HLS  |99.78  |17  |4  |
| 10  | ALL  |HSV  |99.84  |16  |4  |
| 12  | ALL  |RGB  |99.5  |25  |7  |

Following parameters combination is selected:

| HOG parameters    |Value  |
|:-----------|:-------------|
| Orientation  | 10  |
| Pixel per cell  | 8      |
| Cell per block  | 2      |
| Color space  | YCrCb      |

```python
def bin_spatial(img, size=(32, 32)):
    color1 = cv.resize(img[:, :, 0], size).ravel()
    color2 = cv.resize(img[:, :, 1], size).ravel()
    color3 = cv.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))
```
``` python
def color_hist(img, nbins=32):  # bins_range=(0, 256)
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)[0]
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)[0]
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)[0]
    
    hist_features = np.concatenate((channel1_hist, 
                                    channel2_hist, 
                                    channel3_hist))
    return hist_features
```

```python
def get_hog_features(img, feature_vec=False, folder="", filename=None):
    features = hog(img,
                   orientations=config["orient"],
                   pixels_per_cell=(config["pix_per_cell"],
                                    config["pix_per_cell"]),
                   cells_per_block=(config["cell_per_block"],
                                    config["cell_per_block"]),
                   transform_sqrt=True,
                   visualise=False,
                   feature_vector=feature_vec)
    return features
```
Here is an example of HOG features of training data samples:

1: `Original Image`. 2:  `Channel 1 HOG features` 3: `Channel 2` 4: `Channel 3`

![alt text](./documentation/hog-features-1.png)

![alt text](./documentation/hog-features-2.png)

![alt text](./documentation/hog-features-3.png)

![alt text](./documentation/hog-features-4.png)


### 2-	Training SVM Classifier:

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/classifier.py`  |
| Method  | `Classifier.normalize_features`      |
| Method  | `Classifier.get_trained_classifier`      |

The algorithm is as follows:

-	Randomize dataset using numpy `shuffle`.
-	Normalize features using sklearn `StandardScaler`.
Using Support Vector Machine (`SVM`) classifier to fit on the training features and labels.
-	Save the trained classifier as `pickle` file for reuse.

Parameter for `SVM` classifier found using `GridSearchCV` are as follows:

| SVM parameters    |Value  |
|:-----------|:-------------|
| Kernel  | rbf  |
| C  | 0.001      |

```python
not_cars = glob.glob(config["training_not_cars"])
cars = glob.glob(config["training_cars"])

# files for cars and not cars
not_cars_files = [img_file for img_file in not_cars]
cars_files = [img_file for img_file in cars]

# features for cars and not cars
car_features = FeatureExtraction.extract_features(cars_files)
not_cars_features = FeatureExtraction.extract_features(not_cars_files)

# append the feature vertically -- i.e. grow in rows with rows constant
features = np.vstack((car_features, not_cars_features)).astype(np.float64)

# normalize the features
features, scaler = Classifier.normalize_features(features)

# labels
labels = np.hstack((np.ones(len(cars_files)), np.zeros(len(not_cars_files))))

# shuffle dataset
features, labels = shuffle(features, labels)

# initialize SVM with optimized params using GridSearchCV
clf = SVC( kernel='rbf', C=0.001)

# train the classifier
clf.fit(features, labels)

print("classifier trained.")

pickle.dump({"clf": clf, "x_scaler": scaler}, open(config["classifier"], 'wb'))
return clf, scaler
```
### 3-	Sliding Window Search with HOG subsampling:

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/window_search.py`  |
| Method  | `WindowSearch.get_window_params`      |
| Method  | `WindowSearch.get_frame_hog`      |
| Method  | `WindowSearch.get_box`      |
| Method  | `WindowSearch.get_bounding_boxes`      |

The algorithm is as follows:

-	Get `HOG` features for each full image:
	-	Get Region of Interest (`ROI`). Description for selecting `ROI` is provided below.
    -	Find number of search steps using window size and number of windows.
    -	Get `HOG` features of `Y`, `Cr` and `Cb` channels individually.
    
-	Loop over the windows in `x` and `y` direction:
    -	Get HOG features for the full image only once.
	-	Get subsample of `HOG` features for each window.
    -	Get Spatial and Color Histogram features of the subsample.
    -	Use HOG, Spatial and Color features to predict the labels using pretrained SVM classifier.
    -	Get the coordinates of bounding boxes if the classifier predicts the label as a car.
    
-	Determining the ROI:
	-	ROI is carefully chosen after experimentation to achieve a balance between prediction accuracy and time taken window searching.
    -	The larger the ROI, the more area is search by sliding window algorithm but the time taken will be higher.
    - Smaller ROI has shorter searching time but at the expense of missing potential regions to detect the vehicle. 
    
| ROI parameters    |Value  |
|:-----------|:-------------|
| Left Side  | (0, 400), (370, 600)  |
| Top Side  | (400, 800), (380, 560)  |
| Right Side  | (800, 1270), (370, 600)  |

![alt text](./documentation/region-1.png)
![alt text](./documentation/region-2.png)
![alt text](./documentation/region-3.png)


-	Determining window parameters:
	-	Window size is set to be **64** as image size in training dataset is **64x64**.
	-	Number of blocks in a window are defined.
        ```
        Blocks per window = (Window size) / (Pixels per cell - Cells per block + 1)
        ```
	-	Instead of using overlap, cells per step is defined using `number of blocks` and `blocks in a window`.
        ```
        Number of steps = (Blocks - Blocks per window) / (Cells per step)
        ```
        
    
| Window Search parameters    |Value  |
|:-----------|:-------------|
| Window size  | 64  |
| Scale  | 1.7  |
| Subsample Image Size  | (64, 64)      |
| Number of Blocks   |Right: (28,15)      |
|    | Top: (28,12)      |
|    | Left: (33,15)      |
| Number of Steps   | Right: (10,4)      |
| | Top: (10,2)      |
| | Left: (13,4)      |
| Blocks per window  | 7      |

![alt text](./documentation/multi-window-1.png)
![alt text](./documentation/multi-window-2.png)

### 4-	Find the Heatmaps and remove false positives:

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/helper.py`  |
| Method  | `Helper.add_heat`      |
| Method  | `Helper.get_heatmap`      |
| Method  | `Helper.remove_false_positives`      |

The algorithm is as follows:

- Increment heat value (+1) for all pixels within windows where a positive detection is predicted by your classifier.
```python
for box in bbox_list:
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
return heatmap
```
-	Apply thresholding on the heatmap.
```python
# Apply threshold to help remove false positives
heat_binary = heatmap[heatmap <= threshold] = 0
heatmap_binary = np.clip(heat_binary, 0, 1)
```
-	Multi-frame accumulated heatmap can be used to optimize the pipeline for subsequent video frames:
	-	Create history vector with max length restricted using `deque`:
    	```python
    	from collection import deque
    	history = deque(maxlen=8)
        ```
	-	Maintain history of heatmaps:
    	```python
    	history.append(heatmap)
        ```
 	-	Use average heatmap for thresholding and for finding labels instead of only current heatmap:
       	```python
   	    new_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = np.mean(history, axis=0) if len(history) > 0 else new_heat
        ```

| Heatmap parameters    |Value  |
|:-----------|:-------------|
| Threshold  | 2      |

![alt text](./documentation/heat-map-1.png)
![alt text](./documentation/heat-map-2.png)
![alt text](./documentation/heat-map-3.png)
![alt text](./documentation/heat-map-4.png)

### 5-	Update bounding boxes:

| Source Code Reference    |  |
|:-----------|:-------------|
| File  | `implementation/helper.py`  |
| Method  | `Helper.remove_false_positives`      |
| Method  | `Helper.draw_updated_boxes`      |

To update the previously found duplicates and false postive bounding boxes:
-	Using sklearn `measurements` to remove false positive.
-	Iterate through all car labels:
	-	Get `x` and `y` pixel positions.
    -	Define top left and bottom right coordinates of a rectangle.
    -	Draw bounding box using opencv `rectangle`.
    
```python
# get the average of heatmaps from history
new_heat = np.zeros_like(img[:, :, 0]).astype(np.float)
heat = np.mean(history, axis=0) if len(history) > 0 else new_heat
heat = Helper.add_heat(heat, bounding_boxes)

# Get binary heat map
heatmap = Helper.get_heatmap(heat)

# Find final boxes from heatmap using label function
labels = label(heatmap)

# update heatmap history
history.append(heatmap)

# show box where label is 1
detected_cars = Helper.draw_updated_boxes(np.copy(img), labels)
```

![alt text](./documentation/pipeline-1.png)
![alt text](./documentation/pipeline-2.png)
![alt text](./documentation/pipeline-3.png)


### 6-	Combining results with Lane Detection:


The results of vehicle detection are combined with lane detection from previous project.

![alt text](./documentation/pipeline-with-lanes-1.png)
![alt text](./documentation/pipeline-with-lanes-2.png)
![alt text](./documentation/pipeline-with-lanes-3.png)
![alt text](./documentation/pipeline-with-lanes-4.png)
![alt text](./documentation/pipeline-with-lanes-5.png)
![alt text](./documentation/pipeline-with-lanes-6.png)

Here is the video of the complete pipeline:

[![Advanced Vehicle Detection](http://img.youtube.com/vi/Ff96rLUurrc/0.jpg)](http://www.youtube.com/watch?v=Ff96rLUurrc)


Discussion
------------------

##### Possible Improvements:
-	Using Convolutional Neural Networks (CNN) can be much faster compared to Support Vector Machine (SVM), also to reduce the dependency on window sliding algorithm.
- Region of Interest (ROI) can be further improved by generating a trapezoid dynamically instead of hardcoded rectangle coordinates.

##### Potential failure points and problems with current pipeline:
-	Varying light conditions and trees shadow.
-	Pipeline will most definitely fail in snow conditions.
-	Pipeline has issues with overlapping cars.
- The vehicle detection is not fast enough to be realtime, specially sliding window search algorithm even after improvements in Region of Interest (ROI).
- Classifier may not predict Trucks, motorbikes etc. as it is trained only on cars' datasets.
- The pipeline will have issues with high elevations due to fixed Region of Interest (ROI).

