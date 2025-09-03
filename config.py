PDF_DPI=100
PDF_THREAD_COUNT=4

config_detector = {
    "path_model": "./checkpoints/dbnet_onnx/model.onnx",
    "binary_threshold": 0.1,
    "polygon_threshold": 0.1,
    "un_clip_ratio": 0.1,
    "min_size_image": 720,
    "max_candidates": 5000,
    "result_type": "rectangle",
    "db_plus_plus": True,
    "gpu": 1,
}

config_recognizer = {
    "path_model": "./checkpoints/crnn_onnx/model.onnx",
    "img_height": 32,
    "img_width": 240,
    "vocabulary": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|"
    " 0123456789ĂÂÊÔƠƯÁẮẤÉẾÍÓỐỚÚỨÝÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨĐẺỂỈỎỔỞỦỬ"
    "ỶÃẴẪẼỄĨÕỖỠŨỮỸẠẶẬẸỆỊỌỘỢỤỰỴăâêôơưáắấéếíóốớúứýàằầèềìòồờù"
    "ừỳảẳẩđẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ'*:,@.-(#%\")"
    "/~!^&_´+={}[]\\;<>?※”$€£¥₫°²™ā–",
    "batch_size": 32,
    "rgb": False,
    "gpu": 1,
}

config_doclayout = {
    "path_model": "./checkpoints/doclayout_onnx/model.onnx",
    "class_name": [
        'title',
        'plain text',
        'abandon',
        'figure',
        'figure_caption',
        'table', 
        'table_caption',
        'table_footnote', 
        'isolate_formula',
        'formula_caption'
    ],
    "input_shape": (1024, 1024),
    "gpu": 1
}