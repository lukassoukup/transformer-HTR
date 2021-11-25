import cv2
import numpy as np


def labelDictionary():
    labels = [' ', '!', '"', '„', '“', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '%', '=', '°', '§', 'ß',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'Á', 'B', 'C', 'Č', 'D', 'Ď', 'E',
              'Ě', 'É', 'F', 'G', 'H', 'I', 'Í', 'J', 'K', 'L', 'M', 'N', 'Ň', 'O', 'Ó', 'P', 'Q', 'R', 'Ř', 'S', 'Š',
              'T', 'Ť', 'U', 'Ú', 'Ů', 'V', 'W', 'X', 'Y', 'Ý', 'Z', 'Ž', '_', 'a', 'á', 'ä', 'b', 'c', 'č', 'd', 'ď',
              'e', 'ě', 'é', 'f', 'g', 'h', 'i', 'í', 'j', 'k', 'l', 'm', 'n', 'ň', 'o', 'ó', 'ö', 'p', 'q', 'r', 'ř',
              's', 'š', 't', 'ť', 'u', 'ú', 'ů', 'ü', 'v', 'w', 'x', 'y', 'ý', 'z', 'ž']

    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter


num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}  # , 'BLANK_TOKEN': 3}
num_tokens = len(tokens.keys())
vocab_size = num_classes + num_tokens

HEIGHT = 64
WIDTH = 350


def load_image(img_path):

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # resize image to same height with preserved ratio
    rate = float(HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * rate) + 1, HEIGHT), interpolation=cv2.INTER_CUBIC)
    # reverse image
    img = 255 - img

    img_width = img.shape[-1]
    #  padding
    if img_width > WIDTH:
        out_img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_width = WIDTH
    else:
        out_img = np.zeros((HEIGHT, WIDTH), dtype='uint8')
        out_img[:, :img_width] = img

    out_img = out_img / 255.  # float64
    out_img = out_img.astype('float32')

    out_img = np.vstack([np.expand_dims(out_img, 0)] * 3)  # GRAY->RGB
    out_img = np.expand_dims(out_img, 0)

    return out_img, img_width

