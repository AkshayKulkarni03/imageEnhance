# Image Enhancer
Basic image enhancement using python(> 3.7) using [OpenCV](https://pypi.org/project/opencv-python/) and 
[Numpy](https://numpy.org/) and [Deskew](https://github.com/sbrunner/deskew)
## Gray scale, skew detection and correction and Face Detection
Perform the image enhancements with basic functions like
~~~
 Markup : - Gray scaling - default gray scaling filters from opencv
          - Deskewing - detect skew angle and correct image alignment
          - Face Detection - If input flag is set then face detection and masking will be done 
~~~ 

## Cli usage
###### Expected is full path of the image
Perform the Image Enhancement with deskew and without face detection:
```
enhanceImage input.png False
```

Perform the Image Enhancement with face detection and without deskew:
```
enhanceImage input.png False
```