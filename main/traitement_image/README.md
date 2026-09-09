# Dataset generation

It was collected using the proposed cart a image dataset of the way the cart is supposed to pass using the simulated road. In ```dataset_video/``` there is the original videos obtained using a webcam in a cart.

- - -
The ```labels.csv``` dataset correlate the processed files in ```dataset_post_traitement/``` and it specific label. The road was divided in the following labels:

* Straight;
* Crossroad;
* Left Curve (left_curve);
* Right Curve (right_curve);

- - -

The folder ```emb_implementations/``` organizes the files to process the images in the carts' embedded system. 

## Extraction and processing

The library dependences to run the python files present in this repository are

```
pip install python-opencv numpy
```

### Image extration

The script present in ```videotoframe.py``` receives as argument the produced video path and generate a folder with all the images recorded in the video.

### Image processing 

The script ```traitement_all_images.py``` access the images in ```dataset_pre_traitement/``` using glob and generate the post processing images at ```dataset_post_traitement/```. The proposed image processing pipeline proposes an contour extraction from the original images to avoid possible outliers from objects outside of the road. 
