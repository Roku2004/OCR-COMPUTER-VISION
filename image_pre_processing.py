import math

import cv2
import numpy as np


def resize_image(img, maxsize=None, minsize=None):
    h, w = img.shape[0:2]
    if minsize is None and maxsize is None:
        return img
    if minsize is not None:
        min_size = min(h, w)
        if min_size < minsize:
            return img
        else:
            ratio = minsize / min_size
            img = cv2.resize(
                img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC
            )
            return img
    if maxsize is not None:
        max_size = max(h, w)
        if max_size < maxsize:
            return img
        else:
            ratio = maxsize / max_size
            img = cv2.resize(
                img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC
            )
            return img


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def invert_image(src, threshold=49):
    # h, w = src.shape
    # gray = None
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    # convert to binary
    img_bin = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold, 7
    )
    invert_img = 255 - img_bin
    return invert_img


def erase_img(img, temp, rect, pixel_cal=5, method=cv2.INPAINT_TELEA, erase_color=None):
    # format  of rect [x1, y1, x2, y2]
    # (x1, y1) coordinate top left, (x2, y2) coordinate bottom right
    if erase_color is not None and rect is not None:
        dst = img.copy()
        x1, y1, x2, y2 = rect
        cv2.rectangle(dst, (x1, y1), (x2, y2), erase_color, -1)
        return dst
    dst = None
    if temp is None:
        h = img.shape[0]
        w = img.shape[1]
        x1, y1, x2, y2 = rect
        blank_image = np.zeros(shape=[h, w, 1], dtype=np.uint8)
        for r in range(y1, (y2 + 1)):
            for c in range(x1, (x2 + 1)):
                if r < h and c < w:
                    blank_image[r, c] = 255
        # mask = cv2.threshold(blank_image,100,255,cv2.THRESH_BINARY)
        dst = cv2.inpaint(img, blank_image, pixel_cal, method)
    else:
        dst = cv2.inpaint(img, temp, pixel_cal, method)
    return dst


def rotate_image_angle(img, angle, pixel_erase=1, smart_add_bg=False):
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # rotate image
    abs_cos = abs(Mat_rotation[0, 0])
    abs_sin = abs(Mat_rotation[0, 1])

    bound_w = int(h_org * abs_sin + w_org * abs_cos)
    bound_h = int(h_org * abs_cos + w_org * abs_sin)

    Mat_rotation[0, 2] += bound_w / 2 - w_org / 2
    Mat_rotation[1, 2] += bound_h / 2 - h_org / 2

    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))
    if smart_add_bg:
        blank_image = np.zeros(shape=[h_org, w_org, 1], dtype=np.uint8)
        blank_image = 255 - blank_image
        img_result_blank = cv2.warpAffine(blank_image, Mat_rotation, (bound_w, bound_h))
        img_result_blank = 255 - img_result_blank
        img_result = erase_img(img_result, img_result_blank, None, 1)
    return img_result


def auto_rotation(img, expand_angle=3, outputangle=False, debug=False):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray_img.shape
    scale = 1500.0 / min(h_img, w_img)
    gray_img = cv2.resize(gray_img, None, fx=scale, fy=scale)
    gray_img = cv2.medianBlur(gray_img, 3)
    h_img, w_img = gray_img.shape
    edges = invert_image(gray_img)
    blank_image = np.zeros(shape=[h_img, w_img, 1], dtype=np.uint8)
    major = cv2.__version__.split(".")[0]
    if major == "3":
        _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list_bb = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            list_bb.append([x, y, x + w, y + h])
    img_cp = img.copy()
    for b in list_bb:
        centerx = int((b[2] - b[0]) / 2 + b[0])
        centery = int((b[3] - b[1]) / 2 + b[1])
        cv2.circle(blank_image, (centerx, centery), 2, (255, 255, 255), -1)
    minLineLen = min(h_img, w_img) / 10
    if debug:
        img_cp2 = cv2.resize(img, None, fx=scale, fy=scale)
    lines = cv2.HoughLinesP(
        blank_image,
        10,
        np.pi / 180.0,
        threshold=95,
        minLineLength=minLineLen,
        maxLineGap=70,
    )
    list_angle = []
    if lines is None:
        return img
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        angle = (math.atan2((y1 - y2), (x1 - x2)) * 180) / math.pi
        list_angle.append(angle)
        if debug:
            cv2.line(img_cp2, (x1, y1), (x2, y2), (255, 0, 0), 1)
    list_cluster_angle = []
    max_candidate = 0
    angle_rotate = 0
    for ag in list_angle:
        hascluster = False
        for cluster in list_cluster_angle:
            if abs(ag - cluster[0]) < expand_angle:
                cluster[0] = (cluster[0] * cluster[1] + ag) / (cluster[1] + 1)
                cluster[1] += 1
                if cluster[1] > max_candidate:
                    max_candidate = cluster[1]
                    angle_rotate = cluster[0]
                hascluster = True
                break
        if not hascluster:
            new_cluster = [ag, 1]
            list_cluster_angle.append(new_cluster)
    convert_angle_opencv = 180 - angle_rotate
    if debug:
        cv2.imshow("blank img", blank_image)
        cv2.imshow("img_cp", img_cp2)
        cv2.imshow("img ed", edges)
    if outputangle:
        return convert_angle_opencv
    img_cp = rotate_image_angle(img_cp, convert_angle_opencv)
    return img_cp
