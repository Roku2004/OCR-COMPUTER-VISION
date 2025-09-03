import cv2
import numpy as np
import onnxruntime
import pyclipper
from shapely.geometry import Polygon


class DetectorRuleBase:
    def __init__(
        self,
        min_size_image
    ):
        self.min_size_image = min_size_image

    def resize_image(self, img, minsize=None):
        """
        Resizes the input image to a multiple of 32 while maintaining the aspect ratio.

        Parameters:
        - img: Input image of shape (height, width, channels).
        - minsize: Minimum size of the image after resizing, defaults to None.

        Returns:
        - image: Resized image of shape (height, width, channels).
        """
        # Get the height and width of the input image
        h, w = img.shape[:2]

        # Calculate the scaling factor to ensure that the image is a multiple of 32
        # while maintaining the aspect ratio
        scale = min(minsize / min(w, h), 1)

        # Resize the image to the calculated scale while ensuring that the width
        # and height are both multiples of 32
        w = int(w * scale / 32) * 32
        h = int(h * scale / 32) * 32
        image = cv2.resize(img, (w, h))

        return image

    def pre_process(self, img):
        """
        Pre-process the input image.

        Resizes the image to a multiple of 32 while maintaining the aspect ratio.
        Normalizes the image if the 'db_plus_plus' flag is False, otherwise subtracts
        the RGB mean.

        Parameters:
        - img: Input image of shape (height, width, channels).

        Returns:
        - img: Pre-processed image of shape (1, channels, height, width).
        """
        # Resize the image and convert it to float32
        img = self.resize_image(img, self.min_size_image)
        img_cp = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]
        
        # Apply thresholding to preprocess the image
        # thresh =  thresh = cv2.threshold(gray, 10, 150, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 25)
        # Apply dilation to merge adjacent text contours
        cv2.rectangle(thresh, (0, 0), (width - 1, height - 1), 0, 25)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,21))
        cv2.erode(thresh, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,3))
        dilate = cv2.dilate(thresh, kernel, iterations=1)
        return dilate, img_cp

    def run(self, img):
        """
        Run the detector on the input image.

        Args:
            img (str or np.ndarray): Input image. If it's a string, it's assumed to be a path to the image,
            and the image is read using OpenCV. If it's a numpy array, it's assumed to be the image itself.

        Returns:
            The detection results obtained from the post-processing step.
            The format of the results depends on the value of `result_type`.
            If `result_type` is "rectangle", the results are bounding boxes.
            If `result_type` is "polygon", the results are polygons.
        """
        # Convert the image to RGB format if it's not already in that format
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif not isinstance(img, np.ndarray):
            print("Data format doesn't support")
            return

        # Preprocess the image
        h_org, w_org, _ = img.shape
        img, img_cp = self.pre_process(img)
        # cv2.imwrite("test.png", img)
        h_pre, w_pre = img.shape[:2]
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if aspect_ratio > 2 and h > 15 and w > 25:
                cv2.rectangle(img_cp, (x, y), (x + w, y + h), (0, 255, 0), 3)
                result.append([
                    [x/w_pre*w_org, y/h_pre*h_org],
                    [(x+w)/w_pre*w_org, (y)/h_pre*h_org],
                    [(x+w)/w_pre*w_org, (y+h)/h_pre*h_org],
                    [x/w_pre*w_org, (y+h)/h_pre*h_org]]
                )
        # Run the post-processing step to obtain the final results
        # cv2.imwrite("test_cp.png", img_cp)
        print(len(result))
        return result



if __name__ == "__main__":
    boxes, score = detector.detect("C:/Users/Admin/Desktop/test_table/test3.png")
    print(boxes, score)
