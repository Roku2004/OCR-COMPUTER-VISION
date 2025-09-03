class TextOCR:
    def __init__(
        self, xmin=0, ymin=0, xmax=0, ymax=0, text="", conf_reg=[], conf_kie = 0.0
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.text = text
        self.kie_type = "other"
        self.conf_reg = conf_reg
        self.conf_kie = conf_kie
        self.layout_type = "other"
        
    def to_dict(self):
        tmp_dict = {
            "xmin": float(self.xmin),
            "ymin": float(self.ymin),
            "xmax": float(self.xmax),
            "ymax": float(self.ymax),
            "text": str(self.text),
            "layout_type": str(self.layout_type)
        }
        return tmp_dict


class DocLayout:
    def __init__(
        self, xmin=0, ymin=0, xmax=0, ymax=0, class_name="", conf=0
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_name = class_name
        self.conf = conf

    def to_dict(self):
        tmp_dict = {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "class_name": self.class_name,
            "conf": self.conf
        }
        return tmp_dict


class LPResult:
    def __init__(
        self, xmin=0, ymin=0, xmax=0, ymax=0, text="", conf=0
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.text = text
        self.class_name = "license_plate"
        self.conf = conf
    
    def to_dict(self):
        return {
            "xmin": float(self.xmin),
            "ymin": float(self.ymin),
            "xmax": float(self.xmax),
            "ymax": float(self.ymax),
            "text": str(self.text),
            "class_name": str(self.class_name),
            "confidence": float(self.conf)
        }


class YoloResult:
    def __init__(
        self, xmin=0, ymin=0, xmax=0, ymax=0, class_name="", conf=0
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.class_name = class_name
        self.conf = conf
        self.text = ""

    def to_dict(self):
        tmp_dict = {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "class_name": self.class_name,
            "conf": self.conf,
            "text": self.text
        }
        return tmp_dict


class CellData:
    def __init__(self):
        self.bb = None
        self.begin_row = -1
        self.end_row = -1
        self.begin_col = -1
        self.end_col = -1
        self.xmin = 1e9
        self.ymin = 1e9
        self.xmax = -1
        self.ymax = -1
        self.text = ""
        self.conf = 0.0


class ColRowTable:
    def __init__(self):
        self.xmin = 1e9
        self.ymin = 1e9
        self.xmax = -1
        self.ymax = -1
        self.list_cell = []
        self.conf = 0.0

    def add_data(self, cell_data, id):
        if cell_data is not None:
            self.xmin = int(min(cell_data[0], self.xmin))
            self.ymin = int(min(cell_data[1], self.ymin))
            self.xmax = int(max(cell_data[2], self.xmax))
            self.ymax = int(max(cell_data[3], self.ymax))
        self.list_cell.append(id)
