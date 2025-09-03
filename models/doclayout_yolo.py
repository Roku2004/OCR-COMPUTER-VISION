import numpy as np
import onnxruntime
from functools import partial
import time
from definitions import DocLayout
import cv2


class DocLayoutYolo:
    def __init__(
        self,
        path_model,
        class_name,
        gpu,
        input_shape=(1024, 1024),
        verbose=False,
        **kwargs
    ):
        self.path_model = path_model
        self.class_names = class_name
        self.input_name = "images"
        self.output_name = "output0"
        self.input_shape = input_shape
        self.timeout = 10
        if gpu > -1:
            self.ort_session = onnxruntime.InferenceSession(
                path_model, providers=["CUDAExecutionProvider"]
            )
        # self.ort_session.set_providers(['CUDAExecutionProvider'])
        else:
            # self.ort_session.set_providers(['CPUExecutionProvider'])
            self.ort_session = onnxruntime.InferenceSession(
                path_model, providers=["CPUExecutionProvider"]
            )

    def preprocess(self, img):
        """
        Preprocess image for inference
        Args:
            img: OpenCV image in BGR format
        Returns:
            Preprocessed image array
        """
        # Store original image size for postprocessing
        original_shape = img.shape[:2]

        # Resize
        img = cv2.resize(img, self.input_shape)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img, original_shape

    def run(self, images):
        """
        Perform inference on an image
        Args:
            img: OpenCV image in BGR format
        Returns:
            Model output array
        """
        # Prepare input
        list_outputs = []
        count_rs = []
        for i in range(len(images)):
            list_outputs.append([])
        for i, image in enumerate(images):
            input_data, org_shape = self.preprocess(image)
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
            output = self.ort_session.run(None, ort_inputs)[0]
            tmp_rs = self.postprocess(output, org_shape)
            list_outputs[i] = tmp_rs
        return list_outputs
            
    def postprocess(self, output, org_shape, conf_threshold=0.25, iou_threshold=0.45):
        output = output.squeeze()
        boxes = output[:, :-2]
        confidences = output[:, -2]
        class_ids = output[:, -1].astype(int)

        mask = confidences > conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Rescale boxes to original image dimensions
        img_height, img_width = org_shape
        input_height, input_width = self.input_shape
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([img_width, img_height, img_width, img_height])
        final_rs = []
        for box, score, class_id in zip(boxes, confidences, class_ids):
            xmin, ymin, xmax, ymax = box.astype(int)
            label = self.class_names[class_id]
            tmp_rs = DocLayout(xmin, ymin, xmax, ymax, label, score)
            final_rs.append(tmp_rs)
        return final_rs

    def visualize(self, img, list_output, class_names=None):
        """
        Visualize detection results on image
        Args:
            img: Original OpenCV image
            boxes: Array of bounding boxes
            scores: Array of confidence scores
            class_ids: Array of class IDs
            class_names: Optional list of class names
        Returns:
            Image with visualized detections
        """
        img_vis = img.copy()

        for output in list_output:
            x1, y1, x2, y2 = output.xmin, output.ymin, output.xmax, output.ymax
            class_n = output.class_name
            score = output.conf
            # Draw box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{class_n}: {score:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_vis, (x1, y1-label_h-5), (x1+label_w, y1), (0, 255, 0), -1)
            cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        return img_vis


if __name__ == "__main__":
    import config as cfg
    dl_engine = DocLayoutYoloTriton(**cfg.config_doclayout_triton)
    img = cv2.imread("/home/d2seng/chungnx/PDF-Extract-Kit/assets/demo/ocr/page0.jpg")
    rs = dl_engine.run([img])
    imgv = dl_engine.visualize(img, rs[0], cfg.config_doclayout_triton["class_name"])
    cv2.imwrite("vis.jpg", imgv)
    
