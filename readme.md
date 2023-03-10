# Preprocessing and Modeling Approach to Predict Lung Cancer using CT Scans

This article presents an approach to predict lung cancer using CT scans. The process involves preprocessing and modeling steps. In the preprocessing step, the CT scan images are resampled, interpolated, and separated into air regions and lung regions using morphological techniques. The modeling step involves the use of a 3D version of the ResNet model with modifications to address the problem of vanishing gradients. The model was trained and tested using a 5-fold cross-validation approach, achieving high accuracy. However, the model exhibits high variance, which can be addressed by increasing the number of layers and output channels.

## Preprocessing
To make the images uniform, resampling and third-degree polynomial interpolation for the third dimension are performed on each image. The images are in the Hounsfield Unit (HU) scale, and the values of different tissues are described in a table. CT scan images obtained from different manufacturers have different scaling values. These values can be obtained in the HU scale using the slope and intercept parameters determined by the manufacturer. Morphological techniques are then used to separate the air regions from lung regions in the image. The following steps are performed:

Apply a filter with a threshold value of greater than HU 320 to separate non-air regions in a binary image.
Perform Connected component analysis to determine region labels.
Obtain the air region label from the first pixel in the corner of the three-dimensional image.
Change the air regions surrounding the patient to a value of 0 in the binary image.
Identify the largest connected region in each axial slice, including tissue and air surrounding the patient, and set the remaining regions to 0.
Identify and separate the largest region with a value of 0.
After these steps, the internal regions of the lungs are separated, and the remaining regions are removed from the image.

## Model
The ResNet model with 3D convolutional layers is used for prediction. The model uses residual blocks, batch normalization, and ReLu activation function to avoid the problem of vanishing gradients. Each ResBlock3d layer consists of two conv3d layers, and their final outputs are added to the input of the block. The network has a very high variance, which may be addressed by increasing the number of residual layers and output channels. Non-linear classification using the Class Activation Map (CAM) algorithm is used, eliminating the need for fully connected layers.

## Results
The proposed model is implemented using the PyTorch library and trained on an NVIDIA GeForce RTX 2060 GPU. A 5-fold cross-validation approach was used to investigate overfitting, and the model achieved an average accuracy of 86.7%. The error and accuracy graphs of the predictions during training and evaluation of the test data are shown in Figure 6. The final performance report of the model on the test data is shown in Figure 7.

## Class Activation Map
The CAM algorithm is used to determine which parts of the CT scan the network pays more attention to in determining the class of the sample. The algorithm retrieves the weights of the fc layer for each class of the model and multiplies them in the output of the average pool layer when a sample enters the model. The resulting vector is used to calculate the weighted average of the convolutional layers. It creates an image that shows which parts of the image were more weighted in determining the class of the sample. The results of the CAM algorithm are visible for five random samples in Table 5.

## Future Work
To improve model performance and reduce variance, newer architectures such as DensNet can be used, increasing the depth of the network and the number of channels. Attention layers between residual blocks and removing the dictated architecture for running the CAM algorithm may also improve the model accuracy.

# References

Dataset:
https://wiki.cancerimagingarchive.net/display/Public/LungCT-Diagnosis

HU Values Table:
https://en.wikipedia.org/wiki/Hounsfield_scale

CT Scan Images Preprocessing:
https://www.kaggle.com/akh64bit/full-preprocessing-tutorial

Implementing CAM in PyTorch:
https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923

# Contact me

- Email: mohamad-mehdi@live.com
- LinkedIn: https://www.linkedin.com/in/m3di/