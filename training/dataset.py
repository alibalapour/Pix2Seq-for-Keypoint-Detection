import argparse
import random

import yaml
from torch.utils.data import Dataset
from functools import partial
import albumentations
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils import *


def get_transform_train(size, imagenet_normalize):
    t = []
    t.append(albumentations.Sharpen())
    # t.append(albumentations.RandomBrightnessContrast())
    # t.append(albumentations.Solarize(p=0.2))
    # t.append(albumentations.ColorJitter())
    # t.append(albumentations.GaussNoise(var_limit=(10, 100)))
    # t.append(albumentations.InvertImg())
    # t.append(albumentations.SafeRotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.5))
    # t.append(albumentations.CLAHE())
    # t.append(albumentations.UnsharpMask())
    t.append(albumentations.RandomContrast())
    # t.append(albumentations.ColorJitter())
    t.append(albumentations.Cutout(max_h_size=48, max_w_size=48, num_holes=12))
    t.append(albumentations.Resize(size, size))
    if imagenet_normalize:
        t.append(albumentations.Normalize())

    t = albumentations.Compose(t, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    return t


def get_transform_valid(size, imagenet_normalize):
    t = [albumentations.Resize(size, size)]
    if imagenet_normalize:
        t.append(albumentations.Normalize())

    t = albumentations.Compose(t, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    return t


class CornerDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, architecture, resolution, sort=False, transforms=None, tokenizer=None,
                 horizontal_flip=False):
        self.ids = df['id'].unique()
        self.df = df
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.sort = sort
        self.mode = architecture
        self.maxlen = 150
        self.res = resolution
        self.horizontal_flip = horizontal_flip

    def __getitem__(self, idx):
        sample = self.df[self.df['id'] == self.ids[idx]]
        img_path = sample['img_path'].values[0]

        try:
            img = cv2.imread(img_path)[..., ::-1]
        except:
            print(img_path)
        labels = sample['label'].values
        bboxes = sample[['xmin', 'ymin', 'xmax', 'ymax']].values

        new_bboxes = []
        height, width, _ = img.shape
        for row in bboxes:
            new_bboxes.append(list(row))
            if row[0] >= width:
                new_bboxes[-1][0] = width - 1
            if row[1] >= height:
                new_bboxes[-1][1] = height - 1
        bboxes = np.array(new_bboxes)

        if self.transforms is not None:
            transformed = self.transforms(**{
                'image': img,
                'bboxes': bboxes,
                'labels': labels
            })
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

            if self.horizontal_flip:
                if random.random() > 0.5:
                    bboxes = bboxes[::-1]
                    bboxes = [(self.res - x[0], x[1], self.res - x[2], x[3]) for x in bboxes]
                    labels = labels[::-1]
                    img = cv2.flip(img, 1)

        bboxes = [list(a) for a in np.round(np.array(bboxes).astype(np.int16), 0)[:, [0, 1]]]
        img = torch.FloatTensor(img).permute(2, 0, 1)

        if self.sort:
            # ordering points based on their distance to origin
            temp = list(zip([x for x, y in bboxes], labels))
            temp.sort()
            labels = [x[1] for x in temp]
            bboxes = sorted(bboxes, key=lambda x: x[0])

        if self.mode == 'pix2seq':
            if self.tokenizer is not None:
                seqs = self.tokenizer(labels, bboxes, shuffle=False)
                seqs = torch.LongTensor(seqs)
                return img, seqs

        else:
            print('wrong architecture!')
            exit()

        return img, labels, bboxes

    def __len__(self):
        return len(self.ids)


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length
    """
    image_batch, seq_batch = [], []
    for image, seq in batch:
        image_batch.append(image)
        seq_batch.append(seq)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len - seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch


def get_loaders_pix2seq(train_df, valid_df, tokenizer, img_size, train_batch_size, valid_batch_size, max_len, pad_idx,
                        architecture, resolution, num_workers_train=2, num_workers_valid=2, imagenet_normalize=True,
                        horizontal_flip=True):
    train_ds = CornerDetectionDataset(train_df, transforms=get_transform_train(
        img_size, imagenet_normalize=imagenet_normalize), tokenizer=tokenizer,
                                      architecture=architecture,
                                      resolution=resolution,
                                      horizontal_flip=horizontal_flip)

    trainloader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers_train,
        pin_memory=True,
    )

    valid_ds = CornerDetectionDataset(valid_df, transforms=get_transform_valid(
        img_size, imagenet_normalize=imagenet_normalize), tokenizer=tokenizer, architecture=architecture,
                                      resolution=resolution)

    validloader = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx),
        num_workers=num_workers_valid,
        pin_memory=True,
    )

    return trainloader, validloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Corner Detection")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    configs = load_configs(config_file)

    train_df, classes = build_df_from_csv(configs.train_settings.annotation_path,
                                          configs.train_settings.image_path)

    tokenizer = Tokenizer(num_classes=len(classes), num_bins=configs.train_settings.num_bins,
                          width=configs.train_settings.img_size, height=configs.train_settings.img_size,
                          max_len=configs.pix2seq_model.decoder.max_len)

    dataset = CornerDetectionDataset(train_df, transforms=get_transform_train(configs.train_settings.img_size,
                                                                              imagenet_normalize=configs.augmentation.imagenet_normalize),
                                     tokenizer=tokenizer,
                                     architecture=configs.architecture,
                                     resolution=configs.train_settings.img_size,
                                     horizontal_flip=configs.augmentation.horizontal_flip)

    sample_idx = random.sample(list(range(len(dataset))), 16)
    images = []
    for i in sample_idx:
        image, bbox = dataset[i]
        if configs.augmentation.imagenet_normalize:
            image = reverse_normalization(image).numpy()
        else:
            image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        images.append(image)
        bbox = bbox.unsqueeze(0)
        conf = torch.ones_like(bbox).unsqueeze(0)
        bbox, label, conf = postprocess(bbox, conf, tokenizer, no_conf=True)

        visualize(image, bbox[0], label[0], None, display=True, mode='line')
