<p style="font-size:300%; color:#04B45F; text-align:center;line-height : 80px; margin : 0; padding : 0;">
Mapping potato late blight from UAV-based multispectral imagery</p>

Jorge Luis Rodríguez

email: jorodriguezga@unal.edu.co

## Introduction

This work proposes a method to process high resolution multispectral images for detection and mapping late blight in potato crops.
It is based on three main aspects: (i) radiometric and geometric correction of the raw images; (ii) morphological operations for background removing;
(iii) model trainig and supervised classification stage using a random forest classifier; and (iv) implementation of trained model in a new dataset.

# Data and methods

The study area is a 1920 sq.m. potato field located in Subachoque, Colombia (see figure). This field is part of an experimental plot designed to evaluate the potato *Diacol capiro* variety response to different nutrient treatments. As the weather conditions favoured the appearance of late blight disease in several plots, the project had the chance to monitor the crop disease development. The experimental plot was  inspected every seven days during the entire crop life span (i.e. 120 days from planting to maturity). 

Mosaic generated extend over 3.2 Ha, which include the experimental plot and a bigger area where a different variety of potato was located, being this variety
a yellow potato, this means, the tuber root colour it produces is yellow. In contrast, the potato from the experimental plot was from one variety of white potato.
The experimental plot area was clipped by using a reference polygon (red polygon).


## Study area:

<img src="Images/Figure1.png" alt="Imagen" width = "700" height="350" style="border: black 2px solid;" >
    
High resolution multispectral images were acquired at 40 m altitude above the ground surface at 11:00 am local time (GMT-5). Each 
multispectral image acquired by the MicaSense camera had five bands as described in the next table.
In this work camera bands 4 and 5 have been reset according to the following order: Blue (B);
Green (G); Red (R); Red edge (RE) and Near infrared (NIR).

<table class="tg">
  <tr>
    <th class="tg-0pky"># Banda</th>
    <th class="tg-0lax">Nombre</th>
    <th class="tg-0lax">Centro de Banda (nm)</th>
    <th class="tg-0lax">Ancho Banda FWHM (nm)</th>
  </tr>
  <tr>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">Azul</td>
    <td class="tg-0lax">475</td>
    <td class="tg-0lax">20</td>
  </tr>
  <tr>
    <td class="tg-0lax">2</td>
    <td class="tg-0lax">Verde</td>
    <td class="tg-0lax">560</td>
    <td class="tg-0lax">20</td>
  </tr>
  <tr>
    <td class="tg-0lax">3</td>
    <td class="tg-0lax">Rojo</td>
    <td class="tg-0lax">668</td>
    <td class="tg-0lax">10</td>
  </tr>
  <tr>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">Infrarrojo cercano<br></td>
    <td class="tg-0lax">840</td>
    <td class="tg-0lax">40</td>
  </tr>
  <tr>
    <td class="tg-0lax">5</td>
    <td class="tg-0lax">Borde Rojo</td>
    <td class="tg-0lax">717</td>
    <td class="tg-0lax">10</td>
  </tr>
</table>


Experimental crop area consisted of an array of 18 blocks of 12 m $\times$ 8 m inside of a field of 77 m $\times$ 24 m. Each block had 9 rows arranged along the long side
of the field with an area of 216 sq.m., the space between rows was 1m. Each row contained approximately 30 seed tubers of the same variety. 



<img src="Images/experimental_plot_a.png" alt="Imagen" width = "700" height="350" style="border: black 2px solid;" >
(a) May 12, 2018
<img src="Images/experimental_plot_b.png" alt="Imagen" width = "700" height="350" style="border: black 2px solid;" >
(b) May 26, 2018

The layout of the experimental crop. The yellow lines indicate the division of the field into 18 experimental blocks. (a) May 12, 2018; (b) May 26, 2018.


## Background removing

### Multispectral image load




```python
%matplotlib inline
import time
import os, shutil
from matplotlib import pyplot as plt
from matplotlib import colors
from IPython.display import Image
import numpy as np
import cv2
import math
import time
from osgeo import gdal
from skimage import img_as_ubyte
import scipy
import pandas as pd
from skimage.filters import threshold_otsu
import skimage.io as io
  
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.metrics import plot_roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
```


```python
ds = gdal.Open('Raster/Correccion_reflectancia/MR_CE_20180512_Subachoque.tif')
```


```python
# asignacion cada banda 
AZUL = ds.GetRasterBand(1).ReadAsArray()
VERDE = ds.GetRasterBand(2).ReadAsArray()
ROJO = ds.GetRasterBand(3).ReadAsArray()
NIR = ds.GetRasterBand(4).ReadAsArray()
REDEDGE = ds.GetRasterBand(5).ReadAsArray()

srs = ds.GetProjectionRef()
geo_transform = ds.GetGeoTransform()

plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(321) ,plt.imshow(AZUL, cmap='gray'),plt.title('Banda azul')
plt.subplot(322) ,plt.imshow(VERDE, cmap='gray'),plt.title('Banda verde')
plt.subplot(323) ,plt.imshow(ROJO, cmap='gray'),plt.title('Banda roja')
plt.subplot(324) ,plt.imshow(NIR, cmap='gray'),plt.title('Banda infrarroja')
plt.subplot(325) ,plt.imshow(REDEDGE, cmap='gray'),plt.title('Banda borde rojo')
plt.show()
```


![png](Images/output_7_0.png)


### Thresholding

A thresholding technique reduces a grey-level image into an image where objects and background are represented with two levels: a binary image (Glasbey, 1993). Because of the difference in reflectance between the soil and the potato plants, both near infrared and red edge bands allowed separation of the potato plants from the soil.

The first step was to analyse multispectral bands of the orthomosaics by plotting their histograms.
The next figure shows the histogram for each band in the multispectral
orthomosaic for the two categories under study as well as ground reference images of the two categories. Green, Rededge and Near infrared bands have right-skewed bimodal histograms in the first category. Bands with high weed presence had a right-skewed histograms without a clear valley. Thresholding method used to separate vegetation was OTSU's thresholding method (Otsu, 1979).


```python
hist_b1 = np.histogram(AZUL[AZUL!=0].ravel(),100,[0,1])[0]
hist_b2 = np.histogram(VERDE[VERDE!=0].ravel(),100,[0,1])[0]
hist_b3 = np.histogram(ROJO[ROJO!=0].ravel(),100,[0,1])[0]
hist_b4 = np.histogram(REDEDGE[REDEDGE!=0].ravel(),100,[0,1])[0]
hist_b5 = np.histogram(NIR[NIR!=0].ravel(),100,[0,1])[0]

hist_range = np.arange(0,1,0.01)
plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(231), plt.bar(hist_range,hist_b1,0.01), plt.title("Blue")
plt.subplot(232), plt.bar(hist_range,hist_b2,0.01),plt.title('Green')
plt.subplot(233), plt.bar(hist_range,hist_b3,0.01),plt.title('Red')
plt.subplot(234), plt.bar(hist_range,hist_b4,0.01),plt.title('Rededge')
plt.subplot(235), plt.bar(hist_range,hist_b5,0.01),plt.title('Near infrared')
```




    (<matplotlib.axes._subplots.AxesSubplot at 0x7fae900306d8>,
     <Container object of 100 artists>,
     <matplotlib.text.Text at 0x7fae72ad9e48>)




![png](Images/output_9_1.png)



```python
image=NIR
hist_image = image.ravel()
hist = np.histogram(image[image!=0].ravel(),100,[0,1])[0]
thresh = threshold_otsu(image)
binary = image > thresh

plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(131) ,plt.imshow(NIR, cmap='gray'),plt.title('Near infrared')
plt.subplot(132) ,plt.imshow(binary,cmap='gray'),plt.title('Binary image')
plt.show()

plt.show()
```


![png](Images/output_10_0.png)


### Masking image

To delete background from original multispectral image we multiply each band for the binary image resulting from thresholding step using the next equation: 


\begin{equation} \label{Eq3}
 C_{i}(x,y) = A_{i}(x,y)B(x,y),
\end{equation}

where $C_{i}(x,y)$ is the resulting pixel at position $(x,y)$ for $i$ band without background, $A_{i}(x,y)$ is the original pixel at position $(x,y)$ for $i$ band
of the multispectral image and $B(x,y)$ is the binary image pixel at position $(x,y)$ created in the thresholding process.


```python
mask = np.array(binary, dtype='uint8')
filtered_mask = cv2.medianBlur(mask,3)

plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(131) ,plt.imshow(NIR, cmap='gray'),plt.title('Banda NIR')
plt.subplot(132) ,plt.imshow(mask,cmap='gray'),plt.title('Mascara')
plt.subplot(133) ,plt.imshow(filtered_mask,cmap='gray'),plt.title('Mascara con filtro de mediana')
plt.show()

plt.show()
```


![png](Images/output_12_0.png)


Here we can see how looks any band from the multispectral image after masking process. It can be seen how the ground surface does not appear any more.


```python
azul_seg = AZUL*filtered_mask
verde_seg = VERDE*filtered_mask
rojo_seg = ROJO*filtered_mask
rededge_seg = REDEDGE*filtered_mask
nir_seg = NIR*filtered_mask

plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(321) ,plt.imshow(azul_seg, cmap='gray'),plt.title('Blue')
plt.subplot(322) ,plt.imshow(verde_seg, cmap='gray'),plt.title('Green')
plt.subplot(323) ,plt.imshow(rojo_seg, cmap='gray'),plt.title('Red')
plt.subplot(324) ,plt.imshow(rededge_seg, cmap='gray'),plt.title('Red edge')
plt.subplot(325) ,plt.imshow(nir_seg, cmap='gray'),plt.title('Near infrared')
plt.show()
```


![png](Images/output_14_0.png)


We can now create a false colour composition to improve the visualization of the background removing step. This was performed for visualisation purposes only:


```python
# Contrast stretch function for enhance the bands
# Only for visualization purposes
def enhance_band(band, min_in= 0.1, max_in= 0.2):
    min_setup = 0.0
    max_setup = 1.0
    enhanced = (band-min_in)*(((max_setup-min_setup)/(max_in-min_in))+min_setup)
    return enhanced
```


```python
NIR_IN = enhance_band(NIR, min_in= 0.0, max_in= (NIR.max()-0.12))
ROJO_IN = enhance_band(ROJO, min_in= 0.0, max_in= (ROJO.max()-0.06))
VERDE_IN = enhance_band(VERDE, min_in= 0.0, max_in= VERDE.max())

rgb = np.stack([NIR_IN,ROJO_IN,VERDE_IN], axis=2) #axis =2 so its shape is (M,N,3) otherwise doesn't work with plt. (M, N, 3): an image with RGB values (0-1 float or 0-255 int)
mask = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR) # So we can apply the mask 
plantsImage = rgb*mask

plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(121) ,plt.imshow(rgb),plt.title("Original image")
plt.subplot(122) ,plt.imshow(plantsImage),plt.title("Masked image")
plt.show()

plt.figure(2, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(121) ,plt.imshow(rgb[1450:1700,1100:1300]),plt.title("Original image")
plt.subplot(122) ,plt.imshow(plantsImage[1450:1700,1100:1300]),plt.title("Masked image")
plt.show()
```


![png](Images/output_17_0.png)



![png](Images/output_17_1.png)


### Segmented bands to files

Now we proceed to save each of the segmented bands:


```python
bands_out = np.stack([azul_seg,
                      verde_seg,
                      rojo_seg,
                      nir_seg,
                      rededge_seg],
                     axis=2)

band_names = {'M_2018-05-12_Corte_Seg_B1.tif',
              'M_2018-05-12_Corte_Seg_B2.tif',
             'M_2018-05-12_Corte_Seg_B3.tif',
             'M_2018-05-12_Corte_Seg_B4.tif',
             'M_2018-05-12_Corte_Seg_B5.tif'
             }
dir_out = 'Raster/Segmented/'
i=0
for band_name in band_names:
    filename_output = dir_out + band_name
    x_size = ds.RasterXSize  # Raster xsize
    y_size = ds.RasterYSize  # Raster ysize
    driver = gdal.GetDriverByName('GTiff')
    arch = driver.Create(filename_output,x_size,y_size,1,gdal.GDT_Float32)
    arch.SetGeoTransform(geo_transform)
    arch.SetProjection(srs)
    arch.GetRasterBand(1).WriteArray(bands_out[:,:,i].astype(np.float32))
    del(arch)
    i +=1
    print("Band "+band_name+" exported")
```

    Band M_2018-05-12_Corte_Seg_B5.tif exported
    Band M_2018-05-12_Corte_Seg_B1.tif exported
    Band M_2018-05-12_Corte_Seg_B4.tif exported
    Band M_2018-05-12_Corte_Seg_B3.tif exported
    Band M_2018-05-12_Corte_Seg_B2.tif exported


## Supervised classification of dataset A and models training

### Loading segmented bands


```python
rootdir = "Raster/Segmented"
# path to your training data
path_pix = "Entrenamiento/"
# path to your model
path_model = "Data/Models/"
# path to your classification results
path_class = "Data/Class/"

samples = path_pix + "Entrenamiento_raster.tif"

ds_seg1 = gdal.Open('Raster/Segmented/M_2018-05-12_Corte_Seg_B1.tif')
ds_seg2 = gdal.Open('Raster/Segmented/M_2018-05-12_Corte_Seg_B2.tif')
ds_seg3 = gdal.Open('Raster/Segmented/M_2018-05-12_Corte_Seg_B3.tif')
ds_seg4 = gdal.Open('Raster/Segmented/M_2018-05-12_Corte_Seg_B4.tif')
ds_seg5 = gdal.Open('Raster/Segmented/M_2018-05-12_Corte_Seg_B5.tif')

Segmented_image = np.stack([ds_seg1.GetRasterBand(1).ReadAsArray(),
                           ds_seg2.GetRasterBand(1).ReadAsArray(),
                           ds_seg3.GetRasterBand(1).ReadAsArray(),
                           ds_seg4.GetRasterBand(1).ReadAsArray(),
                           ds_seg5.GetRasterBand(1).ReadAsArray()], axis=2)
```

Training and classification of the dataset is performed by the next function:


```python
def plot_time(dt):
    fig, ax = plt.subplots()

    methods = ('RF', 'GBC', 'SVC', 'LSVC', 'KNN')
    y_pos = np.arange(len(methods))
    performance = dt

    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Seconds')
    ax.set_title('Performance by method')

    plt.show()

# declare a new function
def training(img_ds, samples, path_pix, path_model, path_class):

    img = img_ds.copy()

    # Load trainig classes 
    roi_ds = io.imread(samples)   
    roi = np.array(roi_ds, dtype='int8')  
    
    # read in your labels
    labels = np.unique(roi[roi > 0]) 
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

    # compose your X,Y data (dataset - training data)     
    X = img[roi > 0, :] 
    Y = roi[roi > 0] 
    
    # Split dataset into trainig an test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    # assign class weights (class 1 has the weight 3, etc.)
    #weights = {1:3, 2:2, 3:2, 4:2}
    weights = None
    #weights = "balanced_subsample"
    
    ################### Random Forest Classifier ##########################
    # build your Random Forest Classifier 
    # for more information: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    ti_rf = time.time()
    rf = RandomForestClassifier(class_weight = weights, n_estimators = 500, criterion = 'gini', max_depth = 4, 
                                min_samples_split = 2, min_samples_leaf = 2, max_features = 'auto', 
                                bootstrap = True, oob_score = True, n_jobs = 2, random_state = None, verbose = True)  
    
    
    # now fit your training data with the original dataset
    rf = rf.fit(X_train,Y_train)

    # Save your Random Forest Model     
    model = path_model + "model_RF.pkl"
    joblib.dump(rf, model)   

    # call your random forest model
    rf = path_model + "model_RF.pkl"          
    clf = joblib.load(rf)    

    # Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack) 
    new_shape = (img.shape[0] * img.shape[1], img.shape[2]) 
    img_as_array = img[:, :, :23].reshape(new_shape)   

    class_prediction = clf.predict(img_as_array) 
    class_prediction = class_prediction.reshape(img[:, :, 0].shape)     
    tf_rf = time.time()
    dt_rf = tf_rf - ti_rf
    
    #########################Gradient Boost Classifier#############################
    # alternatively you may try out a Gradient Boosting Classifier 
    # It is much less RAM consuming and considers weak training data      
    ti_gbc = time.time()
    GBC = GradientBoostingClassifier(n_estimators = 500, min_samples_leaf = 1, min_samples_split = 4, max_depth = 4,    
                                    max_features = 'auto', learning_rate = 0.8, subsample = 1, random_state = None,         
                                    warm_start = True)
    
    # now fit your training data with the original dataset
    GBC = GBC.fit(X_train,Y_train)

    # Save your Gradient Boosting Model     
    model_GBC = path_model + "model_GBC.pkl"
    joblib.dump(GBC, model_GBC)

    GBC = path_model + "model_GBC.pkl"          
    clf_GBC = joblib.load(GBC)    

    # Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack) 
    new_shape = (img.shape[0] * img.shape[1], img.shape[2]) 
    img_as_array = img[:, :, :23].reshape(new_shape)   

    class_prediction_GBC = clf_GBC.predict(img_as_array) 
    class_prediction_GBC = class_prediction_GBC.reshape(img[:, :, 0].shape)
    tf_gbc = time.time()
    dt_gbc = tf_gbc - ti_gbc
        
    #########################Support Vector Classifier#############################    
    ti_svc = time.time()
    svc = SVC(random_state = None, gamma = 'auto')
    
    # now fit your training data with the original dataset
    svc = svc.fit(X_train,Y_train)

    # export your SVC model     
    model_svc = path_model + "model_svc.pkl"
    joblib.dump(svc, model_svc)

    # call your SVC model
    svc = path_model + "model_svc.pkl"          
    clf_svc = joblib.load(svc)    

    # Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack) 
    new_shape = (img.shape[0] * img.shape[1], img.shape[2]) 
    img_as_array = img[:, :, :23].reshape(new_shape)   

    class_prediction_svc = clf_svc.predict(img_as_array) 
    class_prediction_svc = class_prediction_svc.reshape(img[:, :, 0].shape)  
    tf_svc = time.time()
    dt_svc = tf_svc - ti_svc
    #########################Linear Support Vector Classifier#############################   
    ti_lsvc = time.time()
    lsvc = LinearSVC()
    
    # now fit your training data with the original dataset
    lsvc = lsvc.fit(X_train,Y_train)

    # export your Random Forest / Gradient Boosting Model     
    model_lsvc = path_model + "model_lsvc.pkl"
    joblib.dump(lsvc, model_lsvc)

    # call your Linear SVC model
    lsvc = path_model + "model_lsvc.pkl"          
    clf_lsvc = joblib.load(lsvc)    

    # Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack) 
    new_shape = (img.shape[0] * img.shape[1], img.shape[2]) 
    img_as_array = img[:, :, :23].reshape(new_shape)   

    class_prediction_lsvc = clf_lsvc.predict(img_as_array) 
    class_prediction_lsvc = class_prediction_lsvc.reshape(img[:, :, 0].shape)  

    tf_lsvc = time.time()
    dt_lsvc = tf_lsvc - ti_lsvc
    ######################### KNN Classifier#############################  
    ti_knn = time.time()
    knn = KNeighborsClassifier()
    
    # now fit your training data with the original dataset
    knn = knn.fit(X_train,Y_train)

    # export your KNN Model     
    model_knn = path_model + "model_knn.pkl"
    joblib.dump(knn, model_knn)

    # call your KNN model
    knn = path_model + "model_knn.pkl"          
    clf_knn = joblib.load(knn)    

    # Classification of array and save as image (23 refers to the number of multitemporal NDVI bands in the stack) 
    new_shape = (img.shape[0] * img.shape[1], img.shape[2]) 
    img_as_array = img[:, :, :23].reshape(new_shape)   

    class_prediction_knn = clf_knn.predict(img_as_array) 
    class_prediction_knn = class_prediction_knn.reshape(img[:, :, 0].shape) 
    
    tf_knn = time.time()
    dt_knn = tf_knn - ti_knn
    ########## Plot ROC curves ####################
    ax = plt.gca()

    rfc_disp = plot_roc_curve(clf,X_test,Y_test, ax=ax)
    GBC_disp = plot_roc_curve(clf_GBC,X_test,Y_test, ax=rfc_disp.ax_)
    svc_disp = plot_roc_curve(clf_svc,X_test,Y_test, ax=GBC_disp.ax_, alpha=0.8)
    lsvc_disp = plot_roc_curve(clf_lsvc,X_test,Y_test, ax=svc_disp.ax_, alpha=0.8)
    knn_disp = plot_roc_curve(clf_knn,X_test,Y_test, ax=lsvc_disp.ax_, alpha=0.8)
    knn_disp.figure_.suptitle("ROC curve comparison")
    plt.show()
    
    
    ax1 = plt.gca()
    ax1.set(xlim=[0,0.7], ylim=[0.4,1])

    rfc_disp = plot_roc_curve(clf,X_test,Y_test, ax=ax1)
    GBC_disp = plot_roc_curve(clf_GBC,X_test,Y_test, ax=rfc_disp.ax_)
    svc_disp = plot_roc_curve(clf_svc,X_test,Y_test, ax=GBC_disp.ax_, alpha=0.8)
    lsvc_disp = plot_roc_curve(clf_lsvc,X_test,Y_test, ax=svc_disp.ax_, alpha=0.8)
    knn_disp = plot_roc_curve(clf_knn,X_test,Y_test, ax=lsvc_disp.ax_, alpha=0.8)
    knn_disp.figure_.suptitle("ROC curve comparison - ZOOM")
    plt.show()
    
    plot_time([dt_rf, dt_gbc, dt_svc, dt_lsvc,dt_knn])
    
    
    return class_prediction, class_prediction_GBC, class_prediction_svc, class_prediction_lsvc, class_prediction_knn
```


```python
class_prediction, class_prediction_GBC, class_prediction_svc, class_prediction_lsvc, class_prediction_knn = training(Segmented_image, samples, path_pix, path_model, path_class)

```

    The training data include 2 classes: [1 2]


    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.7s
    [Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    1.5s
    [Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    1.7s finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    3.9s
    [Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:   16.3s
    [Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:   36.9s
    [Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:   41.3s finished
    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.1s finished



![png](Images/output_24_2.png)


    [Parallel(n_jobs=2)]: Using backend ThreadingBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:    0.0s
    [Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=2)]: Done 500 out of 500 | elapsed:    0.1s finished



![png](Images/output_24_4.png)



![png](Images/output_24_5.png)



The scikit-klearn methods do not support NaN data, so we can't replace pixels with zero values by using np.nan, so when classifying, pixels with a zero value are included in one of the classes, in this case, in the late blight class. To clean the results, we use the mask created previously.


```python
from matplotlib.colors import from_levels_and_colors
cmap, norm = from_levels_and_colors([0,0.5,1,2],['black','red','yellow'])

plt.figure(3, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(231) ,plt.imshow(class_prediction, cmap=cmap),plt.title('RFC')
plt.subplot(232) ,plt.imshow(class_prediction_GBC,cmap=cmap),plt.title('GBC')
plt.subplot(233) ,plt.imshow(class_prediction_svc,cmap=cmap),plt.title('SVC')
plt.subplot(234) ,plt.imshow(class_prediction_lsvc,cmap=cmap),plt.title('LSVC')
plt.subplot(235) ,plt.imshow(class_prediction_knn,cmap=cmap),plt.title('KNN')
plt.show()
```


![png](Images/output_26_0.png)



```python
cmap, norm = from_levels_and_colors([0,0.5,1,2],['white','#EA8A00','green'])

rf_classification = class_prediction*filtered_mask
gbc_classification = class_prediction_GBC*filtered_mask
svc_classification = class_prediction_svc*filtered_mask
lsvc_classification = class_prediction_lsvc*filtered_mask
knn_classification = class_prediction_knn*filtered_mask

plt.figure(3, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(231) ,plt.imshow(rf_classification, cmap=cmap),plt.title('RFC')
plt.subplot(232) ,plt.imshow(gbc_classification,cmap=cmap),plt.title('GBC')
plt.subplot(233) ,plt.imshow(svc_classification,cmap=cmap),plt.title('SVC')
plt.subplot(234) ,plt.imshow(lsvc_classification,cmap=cmap),plt.title('LSVC')
plt.subplot(235) ,plt.imshow(knn_classification,cmap=cmap),plt.title('KNN')
plt.show()
```


![png](Images/output_27_0.png)



```python
labels_class = np.unique(class_prediction[class_prediction > 0]) 
print('The classified data include {n} classes: {classes}'.format(n=labels_class.size, classes=labels_class))
```

    The classified data include 2 classes: [1 2]


# Accuracy assessment of results

## Call ground reference


```python
ds_ground_truth = gdal.Open('Validacion/Raster/Ground_truth_2018-05-12.tif')

# asignacion cada banda 
ground_truth_array_raw = ds_ground_truth.GetRasterBand(1).ReadAsArray()
ground_truth_array = ground_truth_array_raw*filtered_mask
srs = ds_ground_truth.GetProjectionRef()
geo_transform = ds_ground_truth.GetGeoTransform()

cmap, norm = from_levels_and_colors([0,0.5,1,2],['black','#EA8A00','green'])

plt.figure(1, dpi=300)
plt.subplots_adjust(left=0.0, right=3.0, bottom=0.0, top=3.0)
plt.subplot(111) ,plt.imshow(ground_truth_array, cmap=cmap),plt.title('Ground truth')
plt.show()
```


![png](Images/output_30_0.png)


### Random forest:


```python
y_true = ground_truth_array.ravel()
target_names = ['0', 'Late blight', 'Healthy']

y_pred_rf = rf_classification.ravel()


print(classification_report(y_true, y_pred_rf, target_names=target_names))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.96      0.98   4190374
     Late blight       0.33      0.89      0.48    268850
         Healthy       0.89      0.43      0.58    532752
    
        accuracy                           0.90   4991976
       macro avg       0.74      0.76      0.68   4991976
    weighted avg       0.95      0.90      0.91   4991976
    


### Gradient Boost Classifier


```python
y_pred_gbc = gbc_classification.ravel()

print(classification_report(y_true, y_pred_gbc, target_names=target_names))

```

                  precision    recall  f1-score   support
    
               0       1.00      0.96      0.98   4190374
     Late blight       0.33      0.80      0.47    268850
         Healthy       0.78      0.48      0.60    532752
    
        accuracy                           0.90   4991976
       macro avg       0.70      0.75      0.68   4991976
    weighted avg       0.94      0.90      0.91   4991976
    


### Support Vector Classifier


```python
y_pred_svc = svc_classification.ravel()

print(classification_report(y_true, y_pred_svc, target_names=target_names))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.96      0.98   4190374
     Late blight       0.33      0.86      0.48    268850
         Healthy       0.87      0.47      0.61    532752
    
        accuracy                           0.90   4991976
       macro avg       0.73      0.76      0.69   4991976
    weighted avg       0.95      0.90      0.91   4991976
    


### Linear Support Vector Classifier


```python
y_pred_lsvc = lsvc_classification.ravel()

print(classification_report(y_true, y_pred_lsvc, target_names=target_names))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.96      0.98   4190374
     Late blight       0.33      0.86      0.48    268850
         Healthy       0.87      0.47      0.61    532752
    
        accuracy                           0.90   4991976
       macro avg       0.73      0.76      0.69   4991976
    weighted avg       0.95      0.90      0.91   4991976
    


### K Neighbors Classifier


```python
y_pred_knn = knn_classification.ravel()

print(classification_report(y_true, y_pred_knn, target_names=target_names))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.96      0.98   4190374
     Late blight       0.33      0.87      0.48    268850
         Healthy       0.87      0.43      0.58    532752
    
        accuracy                           0.90   4991976
       macro avg       0.73      0.75      0.68   4991976
    weighted avg       0.95      0.90      0.91   4991976
    


# References

Glasbey, C.A., 1993.  An Analysis of Histogram-Based Thresholding Algorithms.

Otsu, N., 1979. A Threshold Selection Method from Gray-Level Histograms. IEEE transactions on systems, man, and cybernetics 9, 62–66.arXiv:1011.1669v3.

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011


```python


```