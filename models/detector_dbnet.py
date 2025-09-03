import cv2
import numpy as np
import onnxruntime
import pyclipper
from shapely.geometry import Polygon


class DetectorDB:
    def __init__(
        self,
        path_model,
        binary_threshold,
        polygon_threshold,
        un_clip_ratio,
        max_candidates,
        result_type,
        min_size_image,
        gpu,
        db_plus_plus=True,
        forced_resize=False,
        **kwargs
    ):
        self.min_contour_size = 3
        self.min_size_image = min_size_image
        self.binary_threshold = binary_threshold
        self.polygon_threshold = polygon_threshold
        self.max_candidates = max_candidates
        self.unclip_ratio = un_clip_ratio
        self.db_plus_plus = db_plus_plus
        self.forced_resize = forced_resize

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
        img = self.resize_image(img, self.min_size_image).astype(np.float32)

        if not self.db_plus_plus:
            # Normalize the image if 'db_plus_plus' flag is False
            # Subtract the RGB mean
            img = img / 255.0
            img -= [0.485, 0.456, 0.406]
            img /= [0.229, 0.224, 0.225]
        else:
            # Subtract the RGB mean
            rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793]).astype(
                np.float32
            )
            img -= rgb_mean
            img /= 255.0

        # Transpose the image to the format (channels, height, width)
        # and add an extra dimension for the batch size
        # img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def run(
        self,
        images,
        min_size=720,
        polygon_threshold=0.5,
        binary_threshold=0.5,
        batch_size=4
    ):
        self.min_size_image = min_size 
        self.polygon_threshold = polygon_threshold
        self.binary_threshold = binary_threshold
        bboxes = []
        bbox_confs = []
        result_count = []
        for i in range(len(images)):
            bboxes.append([])
            bbox_confs.append([])
        for i in range(0, len(images), batch_size):
            inputs_ = []
            indexes = []
            for j in range(i, min(i+batch_size, len(bboxes))):
                input_data = self.pre_process(images[j])
                real_index = j
                if len(inputs_) == 0:
                    inputs_.append([input_data])
                    indexes.append([real_index])
                else:
                    is_new_batch = True
                    for ii, inp in enumerate(inputs_):
                        if inp[-1].shape == input_data.shape:
                            inputs_[ii].append(input_data)
                            indexes[ii].append(real_index)
                            is_new_batch = False
                            break
                    if is_new_batch:
                        inputs_.append([input_data])
                        indexes.append([real_index])
            for ii, inp in enumerate(inputs_):
                ort_inputs = {self.ort_session.get_inputs()[0].name: inp}
                output = self.ort_session.run(None, ort_inputs)
                predict = output[0]
                for iii, ik in enumerate(indexes[ii]):
                    bboxes_, bbox_confs_ = self.post_process(predict[iii])
                    bboxes[ik] = bboxes_
                    bbox_confs[ik] = bbox_confs_
        return bboxes, bbox_confs

    def post_process(self, pred):
        mask = pred[0].squeeze()
        _, bitmap = cv2.threshold(
            mask, self.binary_threshold, 255, cv2.THRESH_BINARY)
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        bboxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_min_boxes(contour)
            if sside < self.min_contour_size:
                continue
            points = np.array(points)
            score = self.box_score(mask, contour)
            if self.polygon_threshold > score:
                continue
            polygon = Polygon(points)
            distance = polygon.area / polygon.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            points = np.array(offset.Execute(distance * 1.5)).reshape((-1, 1, 2))

            box, min_side = self.get_min_boxes(points)
            if min_side < self.min_contour_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(box[:, 0] / width, 0, 1)
            box[:, 1] = np.clip(box[:, 1] / height, 0, 1)
            bboxes.append(box)
            scores.append(score)
        bboxes = np.array(bboxes).astype(np.float32)
        scores = np.array(scores).astype(np.float32)
        return bboxes, scores

    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_min_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def get_mini_boxes2(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).astype(np.float32), min(w, h)

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        if xmin == xmax or ymin == ymax:
            return 0
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def box_score(self, bitmap, contour):
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        x1 = np.clip(np.min(contour[:, 0]), 0, w - 1)
        y1 = np.clip(np.min(contour[:, 1]), 0, h - 1)
        x2 = np.clip(np.max(contour[:, 0]), 0, w - 1)
        y2 = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - x1
        contour[:, 1] = contour[:, 1] - y1
        contour = contour.reshape((1, -1, 2)).astype("int32")

        cv2.fillPoly(mask, contour, color=(1, 1))
        return cv2.mean(bitmap[y1:y2 + 1, x1:x2 + 1], mask)[0]


if __name__ == "__main__":
    from config_app import ConfigDetectorDB

    config_db = ConfigDetectorDB()
    detector = DetectorDB(config_db)
    boxes, score = detector.detect("C:/Users/Admin/Desktop/test_table/test3.png")
    print(boxes, score)
