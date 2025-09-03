import os

import cv2
from matplotlib import patches
from networkx.drawing.tests.test_pylab import plt

from box_post_processing import TextOCR


def visualize(img, img_name, list_words: list[TextOCR], output_dir, kie=True):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(40, 40)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_gray, cmap="Greys_r")

    for word in list_words:
        x1 = word.xmin
        y1 = word.ymin
        x2 = word.xmax
        y2 = word.ymax
        plt.text(
            x1 - 5,
            y1 - 5,
            word.text.replace("$", "S"),
            fontsize=max(int((y2 - y1) / 5), 18),
            fontdict={"color": "r"},
        )
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                (x2 - x1),
                (y2 - y1),
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
        )
    fig.savefig(
        os.path.join(
            output_dir, os.path.basename(img_name).replace(".", "_ocr.")
        ).replace(".bmp", ".jpg"),
        bbox_inches="tight",
    )
    plt.clf()
    if not kie:
        return None
    fig, ax = plt.subplots(1)
    fig.set_size_inches(40, 40)
    img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_gray2, cmap="Greys_r")
    for word in list_words:
        if word.kie_type != "other":
            x1 = word.xmin
            y1 = word.ymin
            x2 = word.xmax
            y2 = word.ymax
            plt.text(
                x1 - 5,
                y1 - 5,
                word.kie_type,
                fontsize=max(int((y2 - y1) / 3), 20),
                fontdict={"color": "blue"},
            )
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    (x2 - x1),
                    (y2 - y1),
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                )
            )
    fig.savefig(
        os.path.join(
            output_dir, os.path.basename(img_name).replace(".", "_kie.")
        ).replace(".bmp", ".jpg"),
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def visualize_cv(img_path, list_words, output_dir):
    img = cv2.imread(img_path)
    for word in list_words:
        x1, y1, x2, y2 = word.boundingbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)
    img = cv2.resize(img, (600, 800))
    cv2.imshow("out", img)
