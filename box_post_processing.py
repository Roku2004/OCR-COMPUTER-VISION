from definitions import TextOCR
import numpy as np
import cv2


def sort_polygon(points):
    points.sort(key=lambda x: (x[0][1], x[0][0]))
    for i in range(len(points) - 1):
        for j in range(i, -1, -1):
            if abs(points[j + 1][0][1] - points[j][0][1]) < 10 and (points[j + 1][0][0] < points[j][0][0]):
                temp = points[j]
                points[j] = points[j + 1]
                points[j + 1] = temp
            else:
                break
    return points


def crop_image(image, points):
    assert len(points) == 4, "shape of points must be 4*2"
    crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                         np.linalg.norm(points[2] - points[3])))
    crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                          np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0],
                             [crop_width, 0],
                             [crop_width, crop_height],
                             [0, crop_height]])
    matrix = cv2.getPerspectiveTransform(points, pts_std)
    image = cv2.warpPerspective(image,
                                matrix, (crop_width, crop_height),
                                borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    height, width = image.shape[0:2]
    # if height * 1.0 / width >= 1.5:
    #     image = np.rot90(image, k=3)
    return image


def iou_ratio_one_dimen(x11, x12, x21, x22):
    xb = max(x11, x21)
    xe = min(x12, x22)
    if xe - xb < 0:
        return 0, 0
    return (xe - xb) / (x12 - x11), (xe - xb) / (x22 - x21)


def words_to_lines(list_word, ratio_space=1.2, seprate_sign=[":", ";"]):
    """
    :param list_word:
    :type list_wordm:
    :param max_distance:
    :type max_distance:
    :return:
    :rtype:
    """
    list_word.sort(key=lambda x: (x.xmin, x.ymax))
    list_lines = []
    # list_max_h_of_line save height max of all boxes in line
    list_max_h_of_line = []
    list_special_lines = []
    for i in range(len(list_word)):
        x1, y1, x2, y2 = (
            list_word[i].xmin,
            list_word[i].ymin,
            list_word[i].xmax,
            list_word[i].ymax,
        )
        _, center_y_word = (x1 + x2) / 2, (y1 + y2) / 2
        if x2 - x1 == 0 or y2 - y1 == 0 or len(list_word[i].text) == 0:
            continue
        new_line = True
        special_line = False
        text_c = list_word[i].text
        numbers = sum(c.isdigit() for c in text_c)
        max_len = max([len(x) for x in text_c.split()])
        if (
            ((max_len <= 16 and numbers / len(text_c) < 0.3) or max_len <= 10)
            and text_c != "SELECTED"
            and text_c != "UNSELECTED"
        ):
            for j in range(len(list_lines)):
                xl1, yl1, xl2, yl2 = (
                    list_lines[j].xmin,
                    list_lines[j].ymin,
                    list_lines[j].xmax,
                    list_lines[j].ymax,
                )
                center_x_line, center_y_line = (xl1 + xl2) / 2, (yl1 + yl2) / 2
                iouyl, iouy = iou_ratio_one_dimen(yl1, yl2, y1, y2)
                if iouyl >= 0.75 or iouy >= 0.75:
                    iouxl, ioux = iou_ratio_one_dimen(xl1, xl2, x1, x2)
                    if max(iouxl, ioux) > 0.7:
                        continue
                    wordspace = x1 - xl2
                    hmax = list_max_h_of_line[j]
                    # check height of word vs height max of line
                    if (y2 - y1) / hmax >= 2.7 or hmax / (y2 - y1) >= 2.7:
                        continue
                    # check y center of word and y center of line
                    if (
                        center_y_line < y1
                        or center_y_line > y2
                        or center_y_word < yl1
                        or center_y_word > yl2
                    ):
                        continue
                    max_distance = max((y2 - y1), hmax) * ratio_space
                    if wordspace <= max_distance:
                        if (
                            list_lines[j].text[-1] in seprate_sign
                            or list_word[i].text[0] in seprate_sign
                            and len(list_word[i].text) > 1
                        ):
                            continue
                        list_lines[j].xmin = min(x1, xl1)
                        list_lines[j].ymin = min(y1, yl1)
                        list_lines[j].xmax = max(x2, xl2)
                        list_lines[j].ymax = max(y2, yl2)
                        list_lines[j].text += " " + list_word[i].text
                        list_lines[j].conf_reg += list_word[i].conf_reg
                        list_lines[j].conf_kie += list_word[i].conf_kie
                        list_max_h_of_line[j] = max(hmax, y2 - y1)
                        new_line = False
                if not new_line:
                    break
        else:
            special_line = True
            new_line = False
        if new_line:
            new_line = TextOCR()
            new_line.text = list_word[i].text
            new_line.xmin = list_word[i].xmin
            new_line.ymin = list_word[i].ymin
            new_line.xmax = list_word[i].xmax
            new_line.ymax = list_word[i].ymax
            new_line.conf_reg = list_word[i].conf_reg
            list_max_h_of_line.append(list_word[i].ymax - list_word[i].ymin)
            list_lines.append(new_line)
        if special_line:
            new_line = TextOCR()
            new_line.text = list_word[i].text
            new_line.xmin = list_word[i].xmin
            new_line.ymin = list_word[i].ymin
            new_line.xmax = list_word[i].xmax
            new_line.ymax = list_word[i].ymax
            new_line.conf_reg = list_word[i].conf_reg
            list_special_lines.append(new_line)
    return list_lines + list_special_lines


def merge_box_horizontal(list_text, max_dis=-1):
    list_text.sort(key=lambda x: (x.xmin, x.ymax))
    list_lines = []
    number_element_l = []
    # list_max_h_of_line save height max of all boxes in line
    for i in range(len(list_text)):
        x1, y1, x2, y2 = (
            list_text[i].xmin,
            list_text[i].ymin,
            list_text[i].xmax,
            list_text[i].ymax,
        )
        if x2 - x1 == 0 or y2 - y1 == 0:
            continue
        new_line = True
        for j in range(len(list_lines)):
            xl1, yl1, xl2, yl2 = (
                list_lines[j].xmin,
                list_lines[j].ymin,
                list_lines[j].xmax,
                list_lines[j].ymax,
            )
            iouyl, iouy = iou_ratio_one_dimen(yl1, yl2, y1, y2)
            if iouyl >= 0.75 or iouy >= 0.75:
                iouxl, ioux = iou_ratio_one_dimen(xl1, xl2, x1, x2)
                if max(iouxl, ioux) > 0.7:
                    continue
                crspace = x1 - xl2
                if max_dis != -1 and crspace > max_dis:
                    continue
                list_lines[j].xmin = min(x1, xl1)
                list_lines[j].ymin = min(y1, yl1)
                list_lines[j].xmax = max(x2, xl2)
                list_lines[j].ymax = max(y2, yl2)
                list_lines[j].text += " " + list_text[i].text
                list_lines[j].conf_reg += list_text[i].conf_reg
                list_lines[j].conf_kie += list_text[i].conf_kie
                number_element_l[j] += 1
                new_line = False
            if not new_line:
                break
        if new_line:
            new_line = TextOCR()
            new_line.text = list_text[i].text
            new_line.xmin = list_text[i].xmin
            new_line.ymin = list_text[i].ymin
            new_line.xmax = list_text[i].xmax
            new_line.ymax = list_text[i].ymax
            new_line.conf_reg = list_text[i].conf_reg
            new_line.kie_type = list_text[i].kie_type
            new_line.conf_kie += list_text[i].conf_kie
            list_lines.append(new_line)
            number_element_l.append(1)
        for i, line in enumerate(list_lines):
            line.conf_kie /= number_element_l[i]
    return list_lines


def merge_box_vertical(list_text, max_dis=50):
    list_text.sort(key=lambda x: (x.ymin, x.xmax))
    list_lines = []
    number_element_l = []
    # list_max_h_of_line save height max of all boxes in line
    for i in range(len(list_text)):
        x1, y1, x2, y2 = (
            list_text[i].xmin,
            list_text[i].ymin,
            list_text[i].xmax,
            list_text[i].ymax,
        )
        if x2 - x1 == 0 or y2 - y1 == 0 or len(list_text[i].text) == 0:
            continue
        new_line = True
        for j in range(len(list_lines)):
            xl1, yl1, xl2, yl2 = (
                list_lines[j].xmin,
                list_lines[j].ymin,
                list_lines[j].xmax,
                list_lines[j].ymax,
            )
            iouyl, iouy = iou_ratio_one_dimen(yl1, yl2, y1, y2)
            iouxl, ioux = iou_ratio_one_dimen(xl1, xl2, x1, x2)
            if iouxl >= 0.5 or ioux >= 0.5:
                crspace = abs(y1 - yl2)
                if max_dis != -1 and crspace > max_dis:
                    continue
                list_lines[j].xmin = min(x1, xl1)
                list_lines[j].ymin = min(y1, yl1)
                list_lines[j].xmax = max(x2, xl2)
                list_lines[j].ymax = max(y2, yl2)
                list_lines[j].text += " " + list_text[i].text
                list_lines[j].conf_reg += list_text[i].conf_reg
                list_lines[j].conf_kie += list_text[i].conf_kie
                number_element_l[j] += 1
                new_line = False
            if not new_line:
                break
        if new_line:
            new_line = TextOCR()
            new_line.text = list_text[i].text
            new_line.xmin = list_text[i].xmin
            new_line.ymin = list_text[i].ymin
            new_line.xmax = list_text[i].xmax
            new_line.ymax = list_text[i].ymax
            new_line.conf_reg = list_text[i].conf_reg
            new_line.kie_type = list_text[i].kie_type
            new_line.conf_kie += list_text[i].conf_kie
            list_lines.append(new_line)
            number_element_l.append(1)
        for i, line in enumerate(list_lines):
            line.conf_kie /= number_element_l[i]
    return list_lines


def calculate_overlapse_area(box_a, box_b):
    # Extract coordinates for Box A and Box B
    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b
    
    # Calculate the coordinates of the intersection
    intersection_xmin = max(xmin_a, xmin_b)
    intersection_ymin = max(ymin_a, ymin_b)
    intersection_xmax = min(xmax_a, xmax_b)
    intersection_ymax = min(ymax_a, ymax_b)
    
    # Check if the intersection is valid (not zero area)
    if intersection_xmax > intersection_xmin and intersection_ymax > intersection_ymin:
        intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)
    else:
        intersection_area = 0
    
    # Calculate area of both boxes
    area_a = (xmax_a - xmin_a) * (ymax_a - ymin_a)
    area_b = (xmax_b - xmin_b) * (ymax_b - ymin_b)
    output = max(intersection_area/area_a, intersection_area/area_b)
    return output


def is_bounding_box_within_using_iou(box_a, box_b, threshold=0.3):
    iou_output = calculate_overlapse_area(box_a, box_b)
    if iou_output >= threshold:
        return True
    else:
        return False

    
def doclayout_mapping(list_line_text, list_doclayout, filter_classes=None):
    layout_result = []
    default_rs = [] 
    for i in range(len(list_doclayout)):
        layout_result.append([])
    for rs in list_line_text:
        is_ok = False
        box_text = [rs.xmin, rs.ymin, rs.xmax, rs.ymax]
        for i, rs_lo in enumerate(list_doclayout):
            if filter_classes is not None:
                if rs_lo.class_name not in filter_classes:
                    continue
            box_layout = [rs_lo.xmin, rs_lo.ymin, rs_lo.xmax, rs_lo.ymax]
            layout_type = rs_lo.class_name
            is_within = is_bounding_box_within_using_iou(box_text, box_layout)
            if is_within:
                rs.layout_type = layout_type
                layout_result[i].append(rs)
                is_ok = True
                break
        if not is_ok:
            default_rs.append(rs)
    final_rs = []
    for rs in layout_result:
        if len(rs) == 0:
            continue
        layout_type = rs[0].layout_type
        if layout_type != "table":
            rs = merge_box_vertical(rs, max_dis=10)
        for i in range(len(rs)):
            rs[i].layout_type = layout_type
        final_rs += rs
    final_rs += default_rs
    final_rs.sort(key=lambda x: (x.ymin, x.xmin))
    return final_rs

if __name__ == "__main__":
    box_a = [409, 2798, 652, 2852]
    box_b = [408, 2708, 2275, 2866]
    rs = is_bounding_box_within_using_iou(box_a, box_b)
    print(rs)