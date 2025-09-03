import numpy as np
from datetime import datetime
from box_post_processing import words_to_lines, sort_polygon
from image_pre_processing import (
    auto_rotation,
    resize_image,
    rotate_image_angle
)
from box_post_processing import (
    merge_box_vertical,
    merge_box_horizontal,
    doclayout_mapping
)
from visualize import visualize
from models.detector_dbnet import DetectorDB
from models.recognizer_crnn import RecognizerCRNN
from models.doclayout_yolo import DocLayoutYolo
import asyncio
import cv2
import time
import os
import json


class OcrEngine:
    def __init__(self, config_db, config_crnn, config_doclayout):
        self.db_processor = DetectorDB(**config_db)
        self.crnn_processor = RecognizerCRNN(**config_crnn)
        self.doclayout_processor = DocLayoutYolo(**config_doclayout)
        self.crnn_timeout = 5
        # list keys are allowed when run merge_box_vertical functio

        self.vocabulary = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|" \
             " 0123456789ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬ" \
             "ỶÃẴẪẼỄĨÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờù" \
             "ừỳảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ\'*:,@.-(#%\")" \
             "/~!^&_´+={}[]\\;<>?※”$€£¥₫°²™ā–"

    def isValidString(self,str):
        for c in str:
            if c not in self.vocabulary:
                return False
        return True


    def generate_json_result(self, text_boxes, doc_type, doc_score, index, customer, data_table=None):
        result_dict = {}

        if len(text_boxes) > 0 and doc_type != 'other':
            for text_box in text_boxes:
                if text_box.kie_type != 'other' and text_box.kie_type.find('_key') == -1:
                    if text_box.kie_type not in result_dict:
                        result_dict[text_box.kie_type] = [text_box]
                    else:
                        result_dict[text_box.kie_type].append(text_box)

            for key, value in result_dict.items():
                list_l = merge_box_horizontal(value)
                if key in self.list_allow_merge_vertical:
                    list_l = merge_box_vertical(list_l, threshold_iou_x=0.)
                result_dict[key] = list_l

        result_dict['doc_type'] = [doc_type, doc_score]

        result_dict = self.tpb_processor.process(
            result_dict, index, customer=customer, data_table=data_table)

        return result_dict

    def run_ocr(
        self,
        images,
        ocr_det_min_size,
        ocr_det_binary_threshold,
        ocr_det_polygon_threshold,
        ocr_det_batch_size,
        ocr_rec_image_width,
        ocr_rec_batch_size
    ):
        list_result_words = []
        list_images = []
        for image in images:
            text_boxes_word = []
            # image = auto_rotation(image)
            list_images.append(image)
        bboxes, bbox_confs = self.db_processor.run(
            list_images,
            min_size=ocr_det_min_size,
            binary_threshold=ocr_det_binary_threshold,
            polygon_threshold=ocr_det_polygon_threshold,
            batch_size=ocr_det_batch_size
        )
        for i, bb in enumerate(bboxes):
            bboxes[i] = sort_polygon(list(bb))
        for i, image in enumerate(list_images):
            text_boxes_word = self.crnn_processor.run(
                image, bboxes[i], img_width=ocr_rec_image_width, batch_size=ocr_rec_batch_size)
            # visualize(image, f"page{i+1}.jpg", text_boxes_word, ".")
            list_result_words.append(text_boxes_word)
        return list_result_words

    def run_layout_analysis(
        self,
        images,
        ocr_det_min_size,
        ocr_det_binary_threshold,
        ocr_det_polygon_threshold,
        ocr_det_batch_size,
        ocr_rec_image_width,
        ocr_rec_batch_size
    ):
        list_result_words = []
        list_images = []
        for image in images:
            text_boxes_word = []
            # image = auto_rotation(image)
            list_images.append(image)
        result = []
        bboxes, bbox_confs = self.db_processor.run(
            list_images,
            min_size=ocr_det_min_size,
            binary_threshold=ocr_det_binary_threshold,
            polygon_threshold=ocr_det_polygon_threshold,
            batch_size=ocr_det_batch_size
        )
        layout_rs = self.doclayout_processor.run(list_images)
        for i, image in enumerate(list_images):
            text_boxes_word = self.crnn_processor.run(
                image, bboxes[i], img_width=ocr_rec_image_width, batch_size=ocr_rec_batch_size)
            lrs = words_to_lines(text_boxes_word, seprate_sign=[])
            final_result = doclayout_mapping(lrs, layout_rs[i])
            final_result = sorted(final_result, key=lambda obj: (obj.ymin, obj.xmin))
            rsl = []
            for lt in final_result:
                rsl.append(lt.to_dict())
            result.append(rsl)
        return result

    def run(
        self,
        images,
        ocr_det_min_size,
        ocr_det_binary_threshold,
        ocr_det_polygon_threshold,
        ocr_det_batch_size,
        ocr_rec_image_width,
        ocr_rec_batch_size
    ):
        list_words = self.run_ocr(
            images,
            ocr_det_min_size,
            ocr_det_binary_threshold,
            ocr_det_polygon_threshold,
            ocr_det_batch_size,
            ocr_rec_image_width,
            ocr_rec_batch_size  
        )
        result = []
        for i, ws in enumerate(list_words):
            lrs = words_to_lines(ws, seprate_sign=[])
            lrs = sorted(lrs, key=lambda obj: (obj.ymin, obj.xmin))
            rsl = []
            for lt in lrs:
                rsl.append(lt.to_dict())
            result.append(rsl)
        return result

    def save_data_text(self, img, text_boxes_line, doc_type, output_txt, output_image):
        cv2.imwrite(output_image, img)
        h, w, _ = img.shape
        with open(output_txt, 'w', encoding='utf-8') as f:
            for line in text_boxes_line:
                xmin, ymin, xmax, ymax, text, type = line.xmin, line.ymin, line.xmax, line.ymax, line.text, line.kie_type
                str_w = [str(xmin), str(ymin), str(xmax), str(ymin),
                         str(xmax), str(ymax), str(xmin), str(ymax),
                         text, type]
                strw = ','.join(str_w)+'\n'
                f.write(strw)

    def save_data_json(self, img, text_boxes_word, text_boxes_line, doc_type, output_json, output_image):
        cv2.imwrite(output_image, img)
        h, w, _ = img.shape
        json_result = {}
        json_result['type'] = doc_type
        list_lines = []
        for line in text_boxes_line:
            result_l = {}
            result_l['x'] = line.xmin
            result_l['y'] = line.ymin
            result_l['width'] = line.xmax - line.xmin
            result_l['height'] = line.ymax - line.ymin
            result_l['text'] = line.text
            result_l['text-confidence-score'] = [float(x)
                                                 for x in line.conf_reg]
            result_l['class'] = line.kie_type
            result_l['class-confidence-score'] = line.conf_kie.astype('float')
            result_l['confidence-score'] = float(sum(line.conf_reg) / len(
                line.conf_reg) * line.conf_kie.astype('float'))
            list_lines.append(result_l)
        json_result["text-boxes"] = list_lines
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=4)