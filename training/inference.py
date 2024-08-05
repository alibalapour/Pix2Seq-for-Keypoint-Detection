import os
import random
import shutil
import cv2
import argparse
from pathlib import Path
import torch
import yaml
from box import Box
import albumentations
from tqdm import tqdm

from model import Encoder, Decoder, EncoderDecoder
from utils import Tokenizer, generate, postprocess, visualize


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, size, imagenet_normalize=False):
        """
        creates a dataset for inference procedure

        @param img_paths: paths of target images in a directory
        @param size: image size for inference
        @param imagenet_normalize: boolean specifying usage of normalization or not
        """

        self.img_paths = img_paths
        if imagenet_normalize:
            self.transforms = albumentations.Compose([albumentations.Resize(size, size), albumentations.Normalize()])
        else:
            self.transforms = albumentations.Compose([albumentations.Resize(size, size)])

    def __getitem__(self, idx):
        """
        gets an index and returns an image

        @param idx: index of image in dataset
        @return: an image
        """
        image_path = self.img_paths[idx]
        image = cv2.imread(image_path)[..., ::-1]
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        image = torch.FloatTensor(image).permute(2, 0, 1)
        return image

    def __len__(self):
        """
        returns length of dataset

        @return: length of dataset
        """
        return len(self.img_paths)


def load_configs(config):
    """
    gets a config dictionary and returns a tree-based config

    @param config: configuration dictionary
    @return: tree-based config
    """
    tree_config = Box(config)
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def get_arguments():
    """
    creates a parser and get arguments

    @return: arguments
    """
    parser = argparse.ArgumentParser(description="Corner Detection Inference")
    parser.add_argument(
        "--config_path", "-c", help="The location of curl config file",
        default=os.path.join(result_path, 'config.yaml'))
    arguments = parser.parse_args()
    return arguments


def load_model(configs, tokenizer):
    """
    creates an encoder and decoder as a model

    @param tokenizer: an object for tokenizing output of the model
    @param configs: configuration of the code
    @return: a model for inference
    """

    encoder = Encoder(model_name=configs.pix2seq_model.encoder.model_name,
                      model_type=configs.pix2seq_model.encoder.model_type,
                      pretrained=True,
                      out_dim=configs.pix2seq_model.decoder.dimension,
                      img_size=configs.pix2seq_model.encoder.img_size,
                      patch_size=configs.pix2seq_model.encoder.patch_size)
    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      encoder_length=configs.pix2seq_model.decoder.num_patches,
                      dim=configs.pix2seq_model.decoder.dimension,
                      num_heads=configs.pix2seq_model.decoder.num_heads,
                      num_layers=configs.pix2seq_model.decoder.num_layers,
                      max_len=configs.pix2seq_model.decoder.max_len,
                      pad_idx=tokenizer.PAD_code,
                      device=configs.train_settings.device,
                      dim_feedforward=configs.pix2seq_model.decoder.dim_feedforward,
                      pretrained=configs.pix2seq_model.decoder.pretrained.pretrained,
                      pretrained_path=configs.pix2seq_model.decoder.pretrained.pretrained_path)
    encoder_decoder = EncoderDecoder(encoder, decoder)

    return encoder_decoder


def infer(model, x, img_path, idx, tokenizer, generation_steps, id2cls, img_size, output_path, mixed_precision):
    """
    does inference on an input image

    @param model: model for inference
    @param x: input image
    @param img_path: path of input image
    @param idx: index of image in dataset
    @param tokenizer: an object for tokenizing output
    @param generation_steps: number of iterations for generating output of model
    @param id2cls: a dictionary with ids as keys and classes as values
    @param img_size: an integer represents size of image
    @param output_path: path of a directory where outputs will be saved
    @param mixed_precision: a boolean for specifying usage of mixed precision
    @return: None
    """

    # get input image
    x = x.unsqueeze(0)

    with torch.no_grad() and torch.cuda.amp.autocast(enabled=mixed_precision):
        # generates raw output of the model
        batch_preds, batch_confs = generate(
            model=model,
            x=x,
            tokenizer=tokenizer,
            max_len=generation_steps,
            top_k=0,
            top_p=1
        )

        # returns processes bboxes and labels
        bboxes, labels, confs = postprocess(
            batch_preds=batch_preds,
            batch_confs=batch_confs,
            tokenizer=tokenizer
        )

    # displays or saves inference results
    if bboxes[0] is not None:
        img = cv2.imread(img_path)[..., ::-1]
        img = cv2.resize(img, (img_size, img_size))
        img = visualize(img, bboxes[0], labels[0], id2cls, display=False, mode='line')
        img = cv2.resize(img, (img_size * 2, img_size * 2))
        cv2.imwrite(os.path.join(output_path, str(idx) + '.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def run(configs):
    # config variables
    num_bins = int(configs.train_settings.num_bins)
    img_size = int(configs.train_settings.img_size)
    mixed_precision = bool(configs.train_settings.mixed_precision)
    generation_steps = int(configs.pix2seq_model.generation_steps)
    max_len = int(configs.pix2seq_model.decoder.max_len)
    device = configs.train_settings.device

    # prepares classes
    classes = ['floor']
    id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

    # builds a tokenizer
    tokenizer = Tokenizer(
        num_classes=len(classes),
        num_bins=num_bins,
        width=img_size,
        height=img_size,
        max_len=max_len
    )

    # builds and loads the model
    model = load_model(configs, tokenizer)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(result_path, 'checkpoints', checkpoint_name))['model_state_dict'])
    model.to(device)
    model.eval()

    # check and creates output directory
    output_path = 'Output'
    try:
        shutil.rmtree(output_path)
    except FileNotFoundError:
        pass
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # creates dataset
    path = configs.valid_settings.image_path
    image_paths = [os.path.join(path, file_path) for file_path in os.listdir(path)]
    dataset = InferenceDataset(image_paths, size=img_size, imagenet_normalize=configs.augmentation.imagenet_normalize)

    # main loop of inference
    for idx, x in enumerate(tqdm(dataset)):
        infer(model, x, image_paths[idx], idx, tokenizer, generation_steps, id2cls, img_size, output_path,
              mixed_precision)


if __name__ == '__main__':
    # loads and prepares configs
    result_path = 'results/2023-02-28__05-22-53'
    checkpoint_name = 'checkpoint_64.pth'
    args = get_arguments()
    config_path = args.config_path
    with open(config_path) as file:
        dict_config = yaml.full_load(file)
    configs = load_configs(dict_config)

    run(configs)
