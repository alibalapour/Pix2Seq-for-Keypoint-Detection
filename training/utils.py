import math
import os
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import numpy as np
import pandas as pd
import json
import PIL.Image as Image
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import logging
from itertools import groupby
from xml.dom import minidom
from transformers import top_k_top_p_filtering
from mpl_toolkits.axes_grid1 import ImageGrid
from box import Box

from adabelief_pytorch import AdaBelief
from dataset import get_loaders_pix2seq, CornerDetectionDataset
from optimizer import SAM, GSAM, GSAMLinearScheduler


def load_configs(config):
    tree_config = Box(config)
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def prepare_configs(configs):
    if configs.evaluate:
        print('**************** model is on Evaluation mode ****************')
        assert configs.resume.resume, 'resume config must be true'
        configs.train_settings.num_epochs = 0
        configs.valid_settings.do_every = 1
        configs.optimizer.decay.warmup = -1


def calculate_merge_points(df, threshold):
    label_dict = {k: [] for k in df.id}
    x_dict = {k: [] for k in df.id}
    y_dict = {k: [] for k in df.id}
    width_dict = {k: 0 for k in df.id}
    height_dict = {k: 0 for k in df.id}
    path_dict = {k: '' for k in df.id}

    for i, row in df.iterrows():
        row_id = row['id']
        width = row['width']
        height = row['height']

        x_dict[row_id].append(row['x'] / width)
        y_dict[row_id].append(row['y'] / height)
        label_dict[row_id].append(row['label'])

        width_dict[row_id] = width
        height_dict[row_id] = height
        path_dict[row_id] = row['img_path']

    for iteration in range(5):
        new_label_dict = {k: [] for k in df.id}
        new_x_dict = {k: [] for k in df.id}
        new_y_dict = {k: [] for k in df.id}
        new_width_dict = {k: 0 for k in df.id}
        new_height_dict = {k: 0 for k in df.id}
        new_path_dict = {k: '' for k in df.id}

        for image_name in df['id'].unique():
            i = 0
            count = 0
            row_id = image_name

            while i < len(x_dict[row_id]) - 1:
                x1, y1 = x_dict[row_id][i], y_dict[row_id][i]
                x2, y2 = x_dict[row_id][i + 1], y_dict[row_id][i + 1]
                label1, label2 = label_dict[row_id][i], label_dict[row_id][i + 1]
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if distance < threshold:
                    new_x_dict[row_id].append(((x1 + x2) / 2))
                    new_y_dict[row_id].append(((y1 + y2) / 2))
                    new_label_dict[row_id].append(label1)
                    i += 2

                else:
                    new_x_dict[row_id].append(x1)
                    new_y_dict[row_id].append(y1)
                    new_label_dict[row_id].append(label1)
                    i += 1
            if i == len(x_dict[row_id]) - 1:
                new_x_dict[row_id].append(x_dict[row_id][i])
                new_y_dict[row_id].append(y_dict[row_id][i])
                new_label_dict[row_id].append(label_dict[row_id][i])

            new_width_dict[row_id] = width_dict[row_id]
            new_height_dict[row_id] = height_dict[row_id]
            new_path_dict[row_id] = path_dict[row_id]

        width_dict = new_width_dict
        height_dict = new_height_dict
        path_dict = new_path_dict
        x_dict = new_x_dict
        y_dict = new_y_dict
        label_dict = new_label_dict

    id_list = []
    x_list = []
    y_list = []
    label_list = []
    width_list = []
    height_list = []
    path_list = []

    for image_name in x_dict.keys():
        id_list += [image_name] * len(x_dict[image_name])
        x_list += x_dict[image_name]
        y_list += y_dict[image_name]
        label_list += label_dict[image_name]
        width_list += [width_dict[image_name]] * len(x_dict[image_name])
        height_list += [height_dict[image_name]] * len(x_dict[image_name])
        path_list += [path_dict[image_name]] * len(x_dict[image_name])

    new_df = pd.DataFrame(data={
        'id': id_list,
        'label': label_list,
        'x': x_list,
        'y': y_list,
        'width': width_list,
        'height': height_list,
        'img_path': path_list,
    })

    new_df['x'] = new_df['x'] * new_df['width']
    new_df['y'] = new_df['y'] * new_df['height']

    return new_df


def prepare_dataloaders(configs):
    if configs.import_data_format == 'json':
        train_df, classes = build_df_from_json(configs.train_settings.annotation_path,
                                               configs.train_settings.image_path)
        valid_df, classes = build_df_from_json(configs.valid_settings.annotation_path,
                                               configs.valid_settings.image_path)
    elif configs.import_data_format == 'xml':
        train_df, classes = build_df_from_xml(configs.train_settings.annotation_path,
                                              configs.train_settings.image_path)
        valid_df, classes = build_df_from_xml(configs.valid_settings.annotation_path,
                                              configs.valid_settings.image_path)

    elif configs.import_data_format == 'csv':
        train_df, classes = build_df_from_csv(configs.train_settings.annotation_path,
                                              configs.train_settings.image_path,
                                              configs.train_settings.merge_points,
                                              configs.train_settings.merge_points_threshold)
        valid_df, classes = build_df_from_csv(configs.valid_settings.annotation_path,
                                              configs.valid_settings.image_path,
                                              configs.valid_settings.merge_points,
                                              configs.valid_settings.merge_points_threshold)

    else:
        train_df, classes = build_df_from_json(configs.train_settings.annotation_path,
                                               configs.train_settings.image_path)
        valid_df, classes = build_df_from_json(configs.valid_settings.annotation_path,
                                               configs.valid_settings.image_path)

    print("Train size: ", train_df['id'].nunique())
    print("Valid size: ", valid_df['id'].nunique())

    if configs.architecture == 'pix2seq':
        tokenizer = Tokenizer(num_classes=len(classes), num_bins=configs.train_settings.num_bins,
                              width=configs.train_settings.img_size, height=configs.train_settings.img_size,
                              max_len=configs.pix2seq_model.decoder.max_len)

        train_loader, valid_loader = get_loaders_pix2seq(
            train_df, valid_df, tokenizer,
            configs.train_settings.img_size,
            configs.train_settings.batch_size,
            configs.valid_settings.batch_size,
            configs.pix2seq_model.decoder.max_len,
            tokenizer.PAD_code,
            configs.architecture,
            configs.train_settings.img_size,
            num_workers_train=configs.train_settings.num_workers,
            num_workers_valid=configs.valid_settings.num_workers,
            imagenet_normalize=configs.augmentation.imagenet_normalize)

    else:
        train_loader, valid_loader, tokenizer = None, None, None

    return train_loader, valid_loader, tokenizer


def prepare_optimizer(net, configs, num_train_samples):
    optimizer, scheduler = load_opt(net,
                                    configs.optimizer.name,
                                    configs.optimizer.lr,
                                    configs.optimizer.weight_decay,
                                    configs.optimizer.weight_decouple,
                                    configs.optimizer.eps,
                                    configs.train_settings.sam,
                                    configs.train_settings.gsam,
                                    num_train_samples,
                                    configs)
    if scheduler is None:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_train_samples * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.decay.min_lr,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)

    return optimizer, scheduler


def load_opt(model, opt_name, learning_rate, weight_decay, weight_decouple, eps, sam_option, gsam_option,
             num_train_samples, configs):
    scheduler = None
    if sam_option:
        if opt_name == 'adabelief':
            base_opt = AdaBelief
            opt = SAM(model.parameters(), base_opt, lr=learning_rate, eps=eps, print_change_log=False,
                      weight_decouple=weight_decouple, rectify=False)
        elif opt_name == 'sgd':
            base_opt = torch.optim.SGD
            opt = SAM(model.parameters(), base_opt, lr=learning_rate, weight_decay=weight_decay,
                      momentum=0.9,
                      dampening=0, nesterov=True)
        else:
            base_opt = eval('torch.optim.' + opt_name)
            opt = SAM(model.parameters(), base_opt, lr=learning_rate, eps=eps, weight_decay=weight_decay)

    elif gsam_option:
        if opt_name == 'adabelief':
            base_opt = AdaBelief(model.parameters(), lr=learning_rate, eps=eps, print_change_log=False,
                                 weight_decouple=weight_decouple, rectify=False)
        elif opt_name == 'sgd':
            base_opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9,
                                       dampening=0, nesterov=True)
        else:
            base_opt = eval('torch.optim.' + opt_name)(model.parameters(), lr=learning_rate, eps=eps,
                                                       weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=base_opt,
                                                  first_cycle_steps=num_train_samples * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts,
                                                  cycle_mult=0.0,
                                                  max_lr=learning_rate,
                                                  min_lr=configs.optimizer.decay.min_lr,
                                                  warmup_steps=configs.optimizer.decay.warmup,
                                                  gamma=configs.optimizer.decay.gamma)
        rho_scheduler = GSAMLinearScheduler(
            T_max=num_train_samples * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts,
            max_value=0.02,
            min_value=0.02)
        opt = GSAM(params=model.parameters(), base_optimizer=base_opt, model=model,
                   gsam_alpha=0.4, rho_scheduler=rho_scheduler, adaptive=False)

    else:
        if opt_name == 'adabelief':
            opt = AdaBelief(model.parameters(), lr=learning_rate, eps=eps, print_change_log=False,
                            weight_decouple=weight_decouple, rectify=False)
        elif opt_name == 'sgd':
            opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9,
                                  dampening=0, nesterov=True)
        else:
            opt = eval('torch.optim.' + opt_name)(model.parameters(), lr=learning_rate, eps=eps,
                                                  weight_decay=weight_decay)

    return opt, scheduler


# def checkpoint(epoch, net, optimizer, loss, checkpoints_every, result_path):
def save_checkpoint(epoch, tools):
    model_path = os.path.join(tools['result_path'], 'checkpoints', f'checkpoint_{epoch}.pth')
    if tools['epoch'] % tools['checkpoints_every'] == 0:
        # Saving State Dict
        torch.save({
            'epoch': tools['epoch'],
            'model_state_dict': tools['net'].state_dict(),
            'optimizer_state_dict': tools['optimizer'].state_dict(),
            'scheduler_state_dict': tools['scheduler'].state_dict(),
        }, model_path)


def save_best_model_checkpoint(epoch, tools, logging):
    best_model_path = os.path.join(tools['result_path'], 'checkpoints', f'best_model.pth')
    logging.warning('Save best model based on \"regression loss\" + \"difference of distance of points\" metrics: Epoch',
          str(epoch))
    # Saving State Dict
    torch.save({
        'epoch': tools['epoch'],
        'model_state_dict': tools['net'].state_dict(),
        'optimizer_state_dict': tools['optimizer'].state_dict(),
        'scheduler_state_dict': tools['scheduler'].state_dict(),
        # 'loss': tools['loss'],

    }, best_model_path)


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def clip_number(number, max_threshold, min_threshold=0.0):
    if number > max_threshold:
        return max_threshold
    elif number < min_threshold:
        return min_threshold
    return number


def xml_files_to_df(xml_path, image_folder_path):
    """
        Return pandas dataframe from list of XML files.
        """
    ids = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    point_id = []
    class_ = []
    image_path = []

    file = minidom.parse(xml_path)
    xml_elements = file.getElementsByTagName('image')

    for image_elemnet in xml_elements:
        name = image_elemnet.attributes['name'].value
        width = int(image_elemnet.attributes['width'].value)
        height = int(image_elemnet.attributes['height'].value)

        polyline_element = image_elemnet.getElementsByTagName('polyline')[0]
        image_label = polyline_element.attributes['label'].value
        points = polyline_element.attributes['points'].value
        points = points.split(';')

        for i, xy in enumerate(points):
            x, y = xy.split(',')
            x, y = float(x), float(y)
            xmin.append(clip_number(x - 0.1, width))
            xmax.append(clip_number(x + 0.1, width))
            ymin.append(clip_number(y - 0.1, height))
            ymax.append(clip_number(y + 0.1, height))

            ids.append(name)
            point_id.append(i)
            class_.append(image_label)
            image_path.append(image_folder_path + '/' + name)

    a = {"id": ids,
         "xmin": xmin,
         "xmax": xmax,
         "ymin": ymin,
         "ymax": ymax,
         "point_id": point_id,
         "class_": class_,
         "img_path": image_path}

    df = pd.DataFrame.from_dict(a, orient='index')
    df = df.transpose()

    return df


def json_files_to_df(json_path, image_folder_path):
    """
    Return pandas dataframe from list of XML files.
    """
    ids = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    point_id = []
    class_ = []
    image_path = []
    json_file = json.load(open(json_path))

    for image_annots in json_file['points']:
        for i, point in enumerate(image_annots['floorPoints']):
            width, height = Image.open(image_folder_path + '/' + image_annots['imageName'] + '.jpg').size
            x = point['x']
            y = point['y']
            point_id.append(i)
            ids.append(image_annots['imageName'])

            xmin.append(clip_number(x - 0.1, width))
            xmax.append(clip_number(x + 0.1, width))
            ymin.append(clip_number(y - 0.1, height))
            ymax.append(clip_number(y + 0.1, height))
            class_.append('floor')
            image_path.append(image_folder_path + '/' + image_annots['imageName'] + '.jpg')

    a = {"id": ids,
         "xmin": xmin,
         "xmax": xmax,
         "ymin": ymin,
         "ymax": ymax,
         "point_id": point_id,
         "class_": class_,
         "img_path": image_path}

    df = pd.DataFrame.from_dict(a, orient='index')
    df = df.transpose()

    return df


def concat_gt(row):
    label = row['label']

    xmin = row['xmin']
    xmax = row['xmax']
    ymin = row['ymin']
    ymax = row['ymax']

    return [label, xmin, ymin, xmax, ymax]


def group_objects(df):
    df['concatenated'] = df.apply(concat_gt, axis=1)

    df = df.groupby('id')[['concatenated', 'img_path']].agg({'concatenated': list,
                                                             'img_path': np.unique}).reset_index(drop=True)
    return df


def build_df_from_xml(xmlpath, image_folder_path):
    df = xml_files_to_df(xmlpath, image_folder_path)
    classes = sorted(df['class_'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    df['label'] = df['class_'].map(cls2id)
    df = df[['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']]
    return df, classes


def build_df_from_json(json_path, image_folder_path):
    df = json_files_to_df(json_path, image_folder_path)
    classes = sorted(df['class_'].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    df['label'] = df['class_'].map(cls2id)
    df = df[['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']]
    return df, classes


def build_df_from_csv(csv_path, image_folder_path, merge_points=False, merge_points_threshold=0.0):
    df = pd.read_csv(csv_path)

    if merge_points:
        df = calculate_merge_points(df, merge_points_threshold)

    xmin_list = []
    xmax_list = []
    ymin_list = []
    ymax_list = []

    for row in df.iterrows():
        row = row[1]
        xmin_list.append(clip_number(row.x - 0.1, row.width))
        xmax_list.append(clip_number(row.x + 0.1, row.width))
        ymin_list.append(clip_number(row.y - 0.1, row.height))
        ymax_list.append(clip_number(row.y + 0.1, row.height))

    df['xmin'] = xmin_list
    df['xmax'] = xmax_list
    df['ymin'] = ymin_list
    df['ymax'] = ymax_list

    df['img_path'] = df['img_path'].apply(lambda x: image_folder_path + '/' + x.split('\\')[-1])
    df = df[['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']]
    return df, df.label.unique()


class Tokenizer:
    def __init__(self, num_classes: int, num_bins: int, width: int, height: int, max_len=500):
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
        toekns: torch.LongTensor with shape [L]
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


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(tgt, pad_idx, device):
    """
    tgt: shape(N, L)
    """
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == pad_idx)

    return tgt_mask, tgt_padding_mask


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


GT_COLOR = (0, 255, 0)  # Green
PRED_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color, thickness=1):
    """Visualizes a single bounding box on the image"""
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
    x1, y1 = curr_point
    x2, y2 = next_point

    cv2.circle(img, (int(x1), int(y1)), 3, (200, 0, 0), -1)
    cv2.putText(img, str(number), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (int(x2), int(y2)), (int(x1), int(y1)), (0, 255, 0),
             thickness=1)

    return img


def visualize(image, points, category_ids, category_id_to_name, color=PRED_COLOR, display=True, mode='bbox'):
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


def denorm(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    x = x * std + mean
    return x.permute(1, 2, 0)


def generate(model, x, tokenizer, max_len=50, top_k=0, top_p=1, device='cuda'):
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
    eos_idxs = (batch_preds == tokenizer.EOS_code).float().argmax(dim=-1)
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


# adopted from pytorch.org
def matplotlib_imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)).astype('uint8'))


def reverse_normalization(images):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    un_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return un_normalize(images)


def add_image_to_tensorboard(images, writer, description='images'):
    images = reverse_normalization(images)
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    writer.add_image(description, img_grid)


def prepare_tensorboard(result_path, model, dataloader):
    """
    Initialize tensorboard writers for capturing training and validation stats.
    :return: a writer for training and a writer for validation
    """
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    # display a sample of images from train_dataloader
    images, labels = next(iter(dataloader))
    add_image_to_tensorboard(images, train_writer, 'training images')

    return train_writer, val_writer


def get_logging(result_path):
    logging.basicConfig(filename=os.path.join(result_path, "logs.txt"),
                        format='%(asctime)s - %(message)s',
                        filemode='a')
    logging.getLogger().addHandler(logging.StreamHandler())
    return logging


def print_stats(epoch, logs, logging):
    logging.warning(
        f'Epoch: {epoch} | Train Loss: {logs["train_loss"]:.4f}, Train ACC: {logs["train_acc"]:.4f}, Validation Loss: '
        f'{logs["valid_loss"]:.4f}, Validation ACC: {logs["valid_acc"]:.4f}, Validation F1: {logs["valid_f1"]:.4f},'
        f' Time: {logs["time"]:.1f}, lr: {logs["lr"]:.6f}')


def save_model(epoch, model, opt, result_path, scheduler, description='best_model'):
    Path(os.path.join(result_path, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch
    }, os.path.join(result_path, 'checkpoints', description + '.pth'))


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def display(images, sample_idx):
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes
                     )
    for idx, (ax, im) in enumerate(zip(grid, images)):
        ax.set_axis_off()
        ax.imshow(im)
    plt.show()


if __name__ == '__main__':
    regression_output = torch.rand(20, 2)
    corner_reg = torch.rand(20, 2)

    classification_output = torch.rand(20, 1)
    corner_cls = torch.rand(20, 1)

    print('Done !!!!')
