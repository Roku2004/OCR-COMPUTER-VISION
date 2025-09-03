import time

import cv2
import numpy as np
import onnxruntime
from box_post_processing import crop_image
import time
from definitions import TextOCR


class RecognizerCRNN:
    def __init__(
        self,
        path_model,
        vocabulary,
        img_height,
        img_width, 
        rgb,
        batch_size,
        gpu,
        **kwargs
    ):
        if gpu > -1:
            self.ort_session = onnxruntime.InferenceSession(
                path_model, providers=["CUDAExecutionProvider"]
            )
            # self.ort_session.set_providers(['CUDAExecutionProvider'])
        else:
            self.ort_session = onnxruntime.InferenceSession(
                path_model, providers=["CPUExecutionProvider"]
            )
            # self.ort_session.set_providers(['CPUExecutionProvider'])
        self.vocabulary = vocabulary
        self.height = img_height
        self.width = img_width
        self.rgb = rgb
        if self.rgb:
            self.rgb_mean = np.array([127.5, 127.5, 127.5])
        else:
            self.rgb_mean = np.array([127.5])
        self.batch_size = batch_size
        self.character = ["blank"] + list(vocabulary)

    def preprocess(self, image):
        image = (image.astype(np.float32) - self.rgb_mean) / 255
        image = cv2.resize(image, (self.width, self.height))
        image = np.expand_dims(image, 0)
        return image
    
    def run(self, image, bboxes, img_width, batch_size):
        self.width = img_width
        self.batch_size = batch_size
        text_boxes = []
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_h, image_w = gray_image.shape[:2]
        for i in range(0, len(bboxes), self.batch_size):
            inputs_ = []
            texts_ = []
            for j in range(i, min(i+self.batch_size, len(bboxes))):
                bbox = bboxes[j]
                xmin = int(np.min(bbox[:, 0]) * image_w)
                xmax = int(np.max(bbox[:, 0]) * image_w)
                ymin = int(np.min(bbox[:, 1]) * image_h)
                ymax = int(np.max(bbox[:, 1]) * image_h)
                bbox[:, 0] = bbox[:,0]*image_w
                bbox[:, 1] = bbox[:,1]*image_h
                crop_image_ = crop_image(gray_image, bbox)
            # crop_image = gray_image[ymin:ymax, xmin:xmax]
                if xmax - xmin <= 0 or ymax - ymin <= 0:
                    continue
                text_box = TextOCR(xmin, ymin, xmax, ymax, "", 0)
                texts_.append(text_box)
                input_data = self.preprocess(crop_image_)
                inputs_.append(input_data)
            if len(inputs_) == 0:
                continue
            ort_inputs = {self.ort_session.get_inputs()[0].name: inputs_}
            outputs = self.ort_session.run(None, ort_inputs)
            texts, confidences = self.postprocess(outputs[0])
            for ik, ts in enumerate(texts):
                texts_[ik].text = texts[ik]
                texts_[ik].conf_reg = confidences[ik]
            text_boxes += texts_
        return text_boxes

    def decode(self, text_index, confidences):
        texts = []
        list_char = []
        list_confidences = []
        for i in range(len(text_index)):
            # removing repeated characters and blank.
            if text_index[i] != 0 and (not (i > 0 and text_index[i - 1] == text_index[i])):
                list_char.append(self.character[text_index[i]])
                list_confidences.append(confidences[i, text_index[i]])
        text = ''.join(list_char)
        return text, list_confidences

    def postprocess(self, inputs):
        batch = inputs.shape[0]
        list_text = []
        list_confidences = []
        for i in range(batch):
            text_index = inputs[i, :, :]
            confidences = self.softmax(text_index, axis=1)
            text_index = np.argmax(confidences, axis=1)
            text, conf = self.decode(text_index, confidences)
            list_text.append(text)
            list_confidences.append(conf)
        return list_text, list_confidences

    def softmax(self, X, theta=1.0, axis=None):
        """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
            first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
        # make X at least 2d
        y = np.atleast_2d(X)
        # find axis
        if axis is None:
            axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
        # multiply y against the theta parameter,
        y = y * float(theta)
        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)
        # exponentiate y
        y = np.exp(y)
        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
        # finally: divide elementwise
        p = y / ax_sum
        # flatten if X was 1D
        if len(X.shape) == 1:
            p = p.flatten()
        return p


if __name__ == "__main__":
    from cccd_window_app.config_ocr import ConfigRecognizerCRNN

    config_db = ConfigRecognizerCRNN()
    recognizer = RecognizerCRNN(config_db)
    while True:
        pass
    img = cv2.imread(
        "/data/prdai/chungnx/datatextreg/IC13/Challenge2_Test_Task3_Images/word_3.png"
    )
    text, conf = recognizer.run([img, img, img])
    print(text, conf)
