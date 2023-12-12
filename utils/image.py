import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.circle import Circle
from utils.preprocessing_exceptions import *
import os
import random
from utils.file_utils import extract_user_sample_ids


class Image:
    """
    Class representing an image of an iris, with most common functionalities
    needed for data processing.
    """

    PUPIL_HOUGH_PARAMS = {
        "dp": 1.0,
        "minDist": 200,
        "param1": 10,
        "param2": 15,
        "minRadius": 20,
        "maxRadius": 150
    }

    IRIS_HOUGH_PARAMS = {
        "dp": 1.5,
        "minDist": 200,
        "param1": 10,
        "param2": 30,
        "minRadius": 100,
        "maxRadius": 300
    }

    def __init__(self, img: np.ndarray = None, image_path: str = None):
        self.img = img
        if image_path is not None:
            self.read(image_path) 
        else:
            if img is not None:
                self._update_shape()
            else:
                self.height = None
                self.width = None
                self.num_channels = None

        self.shape = self.img.shape
        self.pupil: Circle = None
        self.iris: Circle = None

    def _update_shape(self):
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        if len(self.img.shape) == 3:
            self.num_channels = self.img.shape[2]
        else:
            self.num_channels = 1

    def read(self, input_image_path: str):
        self.img = cv2.imread(filename=input_image_path)
        assert self.img is not None, f"Read empty image at {input_image_path}"
        self._update_shape()

    def save(self, output_image_path: str):
        assert self.img is not None, "Trying to write an empty binarized_image"
        cv2.imwrite(filename=output_image_path, img=self.img)

    def show(self, title=None, figsize= (5,2), fontsize=10, cmap="gray"):
        plt.figure(figsize=figsize, dpi=80)
        plt.imshow(self.img, cmap=cmap)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.show()

    def show_group(self, title= None, imgarray= None):
      fig, axs= plt.subplots(2,4, figsize=(15, 15))
      
      titles = ["Original Black and White Image", "Removed reflection",
      "Iris image", "Iris only", "Pupil image", "Pupil only", "Optimized pupil", 
      "Circled iris and pupil"]

      plt.subplots_adjust(left=None, bottom=None, right=None, top=0.5, wspace=None, hspace=None)
      for i in range(2):
        for j in range(4):
          axs[i, j].imshow(imgarray[4*i+j].img, cmap= "gray")
          axs[i, j].set_title(titles[4*i+j], fontsize = 8)
      plt.show()
    
    def show_refl(self, title= None, imgarray= None):
      fig, axs= plt.subplots(ncols = 4, figsize=(15, 15))
      titles = ["Original Black and White Image", "Binary Thresholding",
      "Dilation", "Inpainting: Removed Reflection"]

      plt.subplots_adjust(left=None, bottom=None, right=None, top=0.5, wspace=None, hspace=None)
      for j in range(4):
          axs[j].imshow(imgarray[j].img, cmap= "gray")
          axs[j].set_title(titles[j], fontsize = 8)
      plt.show()

    #to divide the figure and background with the given treshold
    def binarize(self, threshold_factor: float):
        p = np.sum(self.img) / (self.height * self.width)
        p = p / threshold_factor

        # Use binary thresholding
        _, thresholded = cv2.threshold(self.img, p, np.max(self.img),
                                       cv2.THRESH_BINARY)

        return Image(thresholded)

    #to erode away the boundaries of foreground object
    def erode(self, kernel, iterations=1):
        return Image(cv2.erode(self.img, kernel, iterations=iterations))

    #to remove noise from the normalized iris
    def close(self, kernel):
        return Image(cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel))

    #to remove noise from background
    def open(self, kernel):
        return Image(cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel))

    #to transform the image in Black and White
    def to_bw(self):
        return Image(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))

    def draw_circle(self, circle: Circle, color=(0, 0, 0), thickness=5):
        return Image(cv2.circle(self.img, (int(circle.center_x), int(circle.center_y)),
                                int(circle.radius), color, thickness=thickness))
    
    #histogram equalization --> normal contrast
    def enhance_contrast(self):
        return Image(cv2.equalizeHist(self.img))

    #adaptive histogram equalization
    def apply_clahe(self, **kwargs):
        return Image(cv2.createCLAHE(**kwargs).apply(self.img))

    #canny edge detector
    def canny(self, **kwargs):
      return Image(cv2.Canny(self.img, 150, 200))
    
    def plot_histogram(self):
        plt.hist(self.img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') 
        plt.show()
    
    #The Median blur operation is similar to the other averaging methods. 
    #Here, the central element of the image is replaced by the median of all the pixels in the kernel area. 
    #This operation processes the edges while removing the noise.
    def blur(self):
      return Image(cv2.medianBlur(self.img, 25))

    @staticmethod
    def _find_circle_nearest_point(binarized_image,
                                   point: np.ndarray,
                                   **kwargs) -> Circle:
        """
        Use Hough Transform to find circles in the image and return the one
        with a center closest to a given point.

        :param binarized_image: Binary BW image
        :param point:           Point of interest
        :param kwargs:          Arguments to be passed to cv2.HoughCircles
        """
        #circles array with coords and radius
        circles = cv2.HoughCircles(binarized_image.img,
                                   method=cv2.HOUGH_GRADIENT, **kwargs)

        if circles is None:
            raise CirclesNotFoundException("No circles found")

 
        dist_from_centre = np.linalg.norm(circles[0, :, :2] - point, axis=1) 
        circle_params = circles[0, np.argmin(dist_from_centre)]
        return Circle(*circle_params), circles


    def find_iris_and_pupil(self):
        """
        Localize iris and pupil on the image. Updates self.iris and self.pupil
        parameters.
        """
        img_array = []
        refl_array = []
  
        eye_bw = self.to_bw()
       
        img_array.append(eye_bw)
        
        #remove reflection
        refl_array.append(eye_bw)
        ret, mask = cv2.threshold(eye_bw.img, 150, 255, cv2.THRESH_BINARY) 
        refl_array.append(Image(mask))
        kernel = np.ones((2, 2), np.uint8)                          
        dilation = cv2.dilate(mask, kernel, iterations=1)
        refl_array.append(Image(dilation))           
        without_reflection = Image(cv2.inpaint(eye_bw.img, dilation, 5, cv2.INPAINT_TELEA))
        img_array.append(without_reflection)
        refl_array.append(without_reflection)
        
      
        # Increase contrast for easier eye_image detection
        iris_img = without_reflection.enhance_contrast() \
        .apply_clahe(clipLimit=2.0, tileGridSize=(10, 10)) \
        .binarize(1.5)
        img_array.append(iris_img)
        
        iris_only = iris_img.close(np.ones((5, 5), np.uint8)) \
            .erode(np.ones((5, 5), np.uint8)) \
            .open(np.ones((5, 5), np.uint8))

        img_array.append(iris_only)
        
        # Find eye_image - its center should be close to the center
        # of the image
        image_center = np.array([eye_bw.width // 2, eye_bw.height // 2]) #np array containing the coordinates of the center
        #to find the circle of the iris --> coord + radius
        iris, circles = self._find_circle_nearest_point(iris_only, image_center,
                                               **Image.IRIS_HOUGH_PARAMS)

        # Use CLAHE to enhance pupil visibility in dark eyes/photos
        pupil_img = eye_bw \
            .apply_clahe(clipLimit=3.0, tileGridSize=(20, 20)) \
            .binarize(2.0) 
        img_array.append(pupil_img)
        

        pupil_only = pupil_img.close(np.ones((6, 6), np.uint8)) 
        #we add an optimization technique to find the pupil as in colored and transparent pictures most of the time it coincides with the iris
        pupil_only = pupil_only.blur()\
        .close(np.ones((12, 12), np.uint8))
        img_array.append(pupil_only)
        # Find pupil - its center should be close to eye_image' center
        pupil, circles = self._find_circle_nearest_point(pupil_only,
                                              iris.to_numpy()[:-1],
                                                **Image.PUPIL_HOUGH_PARAMS)

        # Ensure that the pupil found is within the eye_image
        
        if not pupil.is_within(iris):
          raise PupilOutsideIrisException(
                "Pupil found outside of the eye_image")

        self.iris = iris
        self.pupil = pupil
      
        img_array.append(self.draw_circle(self.pupil).draw_circle(self.iris)) 
       


    def circle_iris_and_pupil(self):
        """
        Draw circles around localized iris and pupil
        """
        if self.pupil is None or self.iris is None:
            self.find_iris_and_pupil()
        
        return self.draw_circle(self.pupil).draw_circle(self.iris)
    



