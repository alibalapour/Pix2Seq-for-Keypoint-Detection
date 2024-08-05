from box import Box
import torch
from transformers import top_k_top_p_filtering
import matplotlib.pyplot as plt
import cv2
import numpy as np


def load_configs(dict_config):
    """

    @param dict_config: Gets a config dictionary
    @return: tree_config which is a variable carries configs of the code
    """
    tree_config = Box(dict_config)
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def create_mask(tgt, pad_idx, device):
    """

    @param tgt: input target array
    @param pad_idx: index of padding
    @param device: device we want to run on it
    @return: tgt_mask, which is mask created for target, and tgt_padding_mask, which represents padding indices
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == pad_idx)

    return tgt_mask, tgt_padding_mask


def generate_square_subsequent_mask(sz, device):
    """

    @param sz: desired size of mask
    @param device: device we want to run on it
    @return: a mask which is squared
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1, device='cuda'):
    """
    This function get an image as input and returns corresponding prediction.

    @param model: pytorch model which we use for inference
    @param x: input image
    @param tokenizer: an object for tokenizing raw output
    @param max_len: maximum length of output sequence
    @param top_k: top k
    @param top_p: top probability
    @param device: device we want to run on it
    @return: predictions and corresponding confidence scores
    """
    x = x.to(device)
    batch_preds = torch.ones(x.size(0), 1).fill_(tokenizer.BOS_code).long().to(device)
    confs = []

    if top_k != 0 or top_p != 1:
        sample = lambda preds: torch.softmax(preds, dim=-1).multinomial(num_samples=1).view(-1, 1)
    else:
        sample = lambda preds: torch.softmax(preds, dim=-1).argmax(dim=-1).view(-1, 1)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            for i in range(max_len):
                preds = model.predict(x, batch_preds)
                # If top_k and top_p are set to default, the following line does nothing!
                preds = top_k_top_p_filtering(preds, top_k=top_k, top_p=top_p)
                if i % 4 == 0:
                    confs_ = torch.softmax(preds, dim=-1).sort(axis=-1, descending=True)[0][:, 0].cpu()
                    confs.append(confs_)
                preds = sample(preds)
                batch_preds = torch.cat([batch_preds, preds], dim=1)
    return batch_preds.cpu(), confs


def postprocess(batch_preds, batch_confs, tokenizer, no_conf=False):
    """
    This function get predicted sequence, predicted confidence scores, and a tokenizer to generate bboxes and labels.

    @param batch_preds: predictions of input images in a batch
    @param batch_confs: confidence scores of input images in a batch
    @param tokenizer: an object for tokenizing raw output
    @param no_conf: a boolean represents need to confidence scores
    @return: parsed bboxes, labels, and confidence scores
    """
    eos_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
    # sanity check
    invalid_idxs = (eos_idxs % 4 != 0).nonzero().view(-1)
    eos_idxs[invalid_idxs] = 0

    all_bboxes = []
    all_labels = []
    all_confs = []
    for i, EOS_idx in enumerate(eos_idxs.tolist()):
        if EOS_idx == 0 or EOS_idx == 1:
            all_bboxes.append(None)
            all_labels.append(None)
            all_confs.append(None)
            continue
        labels, bboxes = tokenizer.decode(batch_preds[i, :EOS_idx + 1])
        if no_conf:
            confs = []
        else:
            confs = [round(batch_confs[j][i].item(), 3) for j in range(len(bboxes))]

        all_bboxes.append(bboxes)
        all_labels.append(labels)
        all_confs.append(confs)

    return all_bboxes, all_labels, all_confs


GT_COLOR = (0, 255, 0)  # Green
PRED_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color, thickness=1):
    """
    Visualizes a single bounding box on the image

    @param img: input image
    @param bbox: list of xy coordinates for corners
    @param class_name: name of predicted class
    @param color: desired color for bbox
    @param thickness: desired thickness for bbox
    @return: output image with bbox
    """
    bbox = [int(item) for item in bbox]
    x_min, y_min = bbox

    cv2.rectangle(img, (x_min - 10, y_min - 10), (x_min + 10, y_min + 10), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(text_height * 1.3)), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min + int(text_height * 1.3)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize_line(img, curr_point, next_point, number):
    """
    Visualizes a line on the image

    @param img: input image
    @param curr_point: coordinates of first point
    @param next_point: coordinates of second point
    @param number: number of point
    @return: output image with added line
    """
    x1, y1 = curr_point
    x2, y2 = next_point

    cv2.circle(img, (int(x1), int(y1)), 3, (200, 0, 0), -1)
    cv2.putText(img, str(number), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (int(x2), int(y2)), (int(x1), int(y1)), (0, 255, 0),
             thickness=1)

    return img


def visualize(image, points, category_ids, category_id_to_name, color=PRED_COLOR, display=True, mode='bbox'):
    """
    Visualizes an image with corners and lines between corners

    @param image: input image
    @param points: list of points' coordinates
    @param category_ids: class ids
    @param category_id_to_name: class id to name dictionary
    @param color: desired color for box borders
    @param display: a boolean for displaying output image
    @param mode: bbox or line
    @return: an output image with predicted labels (bbox or line with corners)
    """
    img = image.copy()
    if mode == 'bbox':
        for point, category_id in zip(points, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, point, class_name, color)
        if display:
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.imshow(img)
            plt.show()

    elif mode == 'line':
        for point_idx in range(len(points) - 1):
            img = visualize_line(img, points[point_idx], points[point_idx + 1], point_idx + 1)
        if display:
            plt.figure(figsize=(8, 8))
            plt.axis('off')
            plt.imshow(img)
            plt.show()
    return img


class Tokenizer:
    def __init__(self, num_classes: int, num_bins: int, width: int, height: int, max_len=500):
        """
        A class for tokenizing output of language model to a meaningful sequence,
        which represents [BOS, x1, y1, cls1, DASH, x2, y2, cls2, DASH, ..., EOS]

        @param num_classes: number of classes
        @param num_bins: number of bins
        @param width: width of input image
        @param height: height of input image
        @param max_len: maximum length of output raw sequence
        """
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len

        self.BOS_code = num_classes + num_bins
        self.EOS_code = self.BOS_code + 1
        self.PAD_code = self.EOS_code + 1
        self.DASH_code = self.PAD_code + 1

        self.vocab_size = num_classes + num_bins + 4

    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """
        return (x * (self.num_bins - 1)).astype('int')

    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """
        return x.astype('float32') / (self.num_bins - 1)

    def __call__(self, labels: list, bboxes: list, shuffle=True):
        assert len(labels) == len(bboxes), "labels and bboxes must have the same length"
        bboxes = np.array(bboxes).astype(float)
        labels = np.array(labels).astype(float)
        labels += self.num_bins
        labels = labels.astype('int')[:self.max_len]

        bboxes[:, 0] = bboxes[:, 0] / self.width
        # bboxes[:, 2] = bboxes[:, 2] / self.width
        bboxes[:, 1] = bboxes[:, 1] / self.height
        # bboxes[:, 3] = bboxes[:, 3] / self.height

        bboxes = self.quantize(bboxes)[:self.max_len]

        if shuffle:
            rand_idxs = np.arange(0, len(bboxes))
            np.random.shuffle(rand_idxs)
            labels = labels[rand_idxs]
            bboxes = bboxes[rand_idxs]

        tokenized = [self.BOS_code]
        for label, bbox in zip(labels, bboxes):
            tokens = list(bbox)
            tokens.append(label)

            tokenized.extend(list(map(int, tokens)))
            tokenized.append(self.DASH_code)

        tokenized.pop()
        tokenized.append(self.EOS_code)

        return tokenized

    def decode(self, tokens: torch.tensor):
        """
        tokens: torch.LongTensor with shape [L]
        """
        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]
        assert (len(tokens) + 1) % 4 == 0, "invalid tokens"

        labels = []
        bboxes = []
        for i in range(2, len(tokens) + 1, 4):
            label = tokens[i]
            bbox = tokens[i - 2: i]
            labels.append(int(label))
            bboxes.append([int(item) for item in bbox])
        labels = np.array(labels) - self.num_bins
        bboxes = np.array(bboxes)
        bboxes = self.dequantize(bboxes)

        bboxes[:, 0] = bboxes[:, 0] * self.width
        bboxes[:, 1] = bboxes[:, 1] * self.height

        return labels, bboxes
