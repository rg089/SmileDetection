import os, sys
import numpy as np
import cv2
import argparse
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_args(dataset=False, save_model=False, save_plot=False, save_json=False, save_weights=False, single_image=False, pretrained_model=False, use_video=False, **kwargs):
    ap = argparse.ArgumentParser()
    if dataset:
        help_ = "path to input dataset"
        if "dataset" in kwargs:
            help_ = kwargs["dataset"]
        ap.add_argument("-d", "--dataset", required=True, help=help_)
    if save_model:
        help_ = "path to output model"
        if "model" in kwargs:
            help_ = kwargs["model"]
        ap.add_argument("-m", "--model", type=str, default="inception", help=help_)
    elif pretrained_model:
        help_ = "name of pretrained model to use"
        if "model" in kwargs:
            help_ = kwargs["model"]
        ap.add_argument("-m", "--model", required=True, help=help_)
    if save_plot:
        help_ = "path to output plot"
        if "plot" in kwargs:
            help_ = kwargs["plot"]
        ap.add_argument("-p", "--plot", required=True, help=help_)
    if save_json:
        help_ = "path to output json"
        if "json" in kwargs:
            help_ = kwargs["json"]
        ap.add_argument("-j", "--json", required=True, help=help_)
    if save_weights:
        help_ = "path to weights directory"
        if "weights" in kwargs:
            help_ = kwargs["weights"]
        ap.add_argument("-w", "--weights", required=True, help=help_)
    if use_video:
        help_ = "path to (optional) video file"
        if "video" in kwargs:
            help_ = kwargs["video"]
        ap.add_argument("-v", "--video", help=help_)
    if single_image:
        help_ = "path to the input image"
        if "image" in kwargs:
            help_ = kwargs["image"]
        ap.add_argument("-i", "--image", help=help_)
    args = vars(ap.parse_args())
    return args


def normalize(*args):
    ans = []
    for i in args:
        ans.append(np.array(i, dtype="float") / 255.0)
    if len(args)==1:
        return ans[0]
    return ans


def encodeY(*args, ohe=True):
    if ohe:
        encoder = LabelBinarizer()
    else:
        encoder = LabelEncoder()
    first = np.array(args[0])
    ans = [encoder, encoder.fit_transform(first)]
    for i in range(1, len(args)):
        ans.append(encoder.transform(np.array(args[i])))
    return ans


def indexOfFirstString(l):
    for i in range(len(l)):
        if type(l[i]) == str:
            return i
    return len(l)


def showImage(*imgs, together=False, **kwnames):
    i = indexOfFirstString(imgs)
    images = imgs[:i]
    names = list(imgs[i:])
    if not together:
        if len(images) > len(names):
            if len(kwnames) != 0:
                names.extend(list(kwnames.values())[:len(images) - len(names)])
            names.extend([f"Image{i}" for i in range(len(names) + 1, len(images) + 1)])
        for i in range(len(images)):
            cv2.imshow(names[i], images[i])
    else:
        if len(names) != 0:
            name = names[0]
        elif "name" in kwnames:
            name = kwnames["name"]
        elif len(kwnames) != 0:
            name = list(kwnames.values())[0]
        else:
            name = "Image"
        cv2.imshow(name, np.hstack(images))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, width=None, height=None, inter=cv2.INTER_CUBIC, fx=1, fy=1):
    """
    Returns the resized image.
    :param img: cv2 image object
    :param width: The fixed width of the result
    :param height: The fixed height of the result
    :param inter: Interpolation Method
    :param fx: Ratio to scale width
    :param fy: Ratio to scale height
    :return: cv2 Image
    """
    h, w = img.shape[:2]

    if not width and not height:
        wn = int(w * fx)
        hn = int(h * fy)
        dim = (wn, hn)

    elif not width:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(img, dim, interpolation=inter)


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def plot_model(history, epochs, validation=True, accuracy=True, loss=True, title="Training Analytics"):
    plt.style.use("ggplot")
    plt.figure()
    if loss:
        plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
        if validation:
            plt.plot(np.arange(0, epochs), history.history["val_loss"], label="validation_loss")
    if accuracy:
        plt.plot(np.arange(0, epochs), history.history["accuracy"], label="accuracy")
        if validation:
            plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="validation_accuracy")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Metric Values")
    plt.legend()
    return plt
