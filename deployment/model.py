import timm
import torch
from torch import nn
from timm.models.layers import trunc_normal_
import cv2
import albumentations

from utils import create_mask, generate, postprocess, visualize, Tokenizer


class Encoder(nn.Module):
    def __init__(self, model_name='deit3_small_patch16_384_in21ft1k', pretrained=True, out_dim=256,
                 img_size=384, model_type='transformer'):
        """
        Model for encoding an image to create a representation for language model

        @param model_name: name of encoder model
        @param pretrained: is pretrained or not
        @param out_dim: output dimension of encoder
        @param img_size: size of input image
        @param model_type: type of model, transformer or cnn
        """
        super().__init__()
        self.model_type = model_type
        self.out_dim = out_dim
        if model_type == 'transformer':
            self.model = timm.create_model(model_name, num_classes=0, global_pool='',
                                           img_size=img_size,
                                           pretrained=pretrained)

            self.bottleneck = nn.AdaptiveAvgPool1d(out_dim)
        elif model_type == 'cnn':
            self.model = timm.create_model(
                model_name, num_classes=0, global_pool='', pretrained=pretrained)

    def forward(self, x):
        """

        @param x: input image
        @return: a representation
        """
        if self.model_type == 'transformer':
            features = self.model(x)
            return self.bottleneck(features[:, 1:])
        elif self.model_type == 'cnn':
            features = self.model(x)
            features = nn.Conv2d(features.shape[1], 576, kernel_size=(1, 1), stride=1, padding=2).to('cuda')(features)
            features = features.view(features.shape[0], -1, features.shape[2] * features.shape[3])
            return nn.Linear(features.shape[2], self.out_dim).to('cuda')(features)


class Decoder(nn.Module):
    def __init__(self, vocab_size, encoder_length, dim, num_heads, num_layers, dim_feedforward, max_len, pad_idx,
                 device, pretrained=False, pretrained_path='decoder.pth'):
        """
        initiates a decoder which is a decoder of a language model

        @param vocab_size: vocabulary size
        @param encoder_length: length of encoder output
        @param dim: hidden dimension of decoder
        @param num_heads: number of heads in decoder
        @param num_layers: number of layers in decoder
        @param dim_feedforward: dimension of decoder feedforward layers
        @param max_len: maximum length of decoder output
        @param pad_idx: index of padding
        @param device: device we want to run on it
        @param pretrained: a boolean for specifying use of pretrained model or not
        @param pretrained_path: path of pre-trained checkpoint
        """
        super().__init__()
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, max_len - 1, dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=0.05)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim, vocab_size)

        self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_length, dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=0.05)

        self.max_len = max_len
        self.pad_idx = pad_idx
        self.device = device

        if pretrained:
            checkpoint_model = torch.load(pretrained_path, map_location='cpu')
            checkpoint_model.embedding = nn.Embedding(vocab_size, dim)
            checkpoint_model.output = nn.Linear(dim, vocab_size)

            self.load_state_dict(checkpoint_model.state_dict())
        else:
            self.init_weights()

    def init_weights(self):
        """
        initiates weights of the model

        @return:
        """
        for name, p in self.named_parameters():
            if 'encoder_pos_embed' in name or 'decoder_pos_embed' in name:
                print("skipping pos_embed...")
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.decoder_pos_embed, std=.02)

    def forward(self, encoder_out, tgt):
        """
        encoder_out: shape(N, L, D)
        tgt: shape(N, L)
        """

        tgt_mask, tgt_padding_mask = create_mask(tgt, self.pad_idx, self.device)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(memory=encoder_out,
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_padding_mask)

        preds = preds.transpose(0, 1)
        return self.output(preds)

    def predict(self, encoder_out, tgt):
        """
        predicts output of an image

        @param encoder_out: output of encoder
        @param tgt: ???
        @return: output of the decoder
        """
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), self.max_len - length - 1).fill_(self.pad_idx).long().to(tgt.device)
        tgt = torch.cat([tgt, padding], dim=1)
        tgt_mask, tgt_padding_mask = create_mask(tgt, self.pad_idx, self.device)
        # is it necessary to multiply it by math.sqrt(d) ?
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.decoder_pos_drop(
            tgt_embedding + self.decoder_pos_embed
        )

        encoder_out = self.encoder_pos_drop(
            encoder_out + self.encoder_pos_embed
        )

        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)

        preds = self.decoder(memory=encoder_out,
                             tgt=tgt_embedding,
                             tgt_mask=tgt_mask,
                             tgt_key_padding_mask=tgt_padding_mask)

        preds = preds.transpose(0, 1)
        return self.output(preds)[:, length - 1, :]


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        """
        creates a model consisting of encoder and decoder

        @param encoder: specified encoder
        @param decoder: specified decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image, tgt):
        encoder_out = self.encoder(image)
        preds = self.decoder(encoder_out, tgt)
        return preds

    def predict(self, image, tgt):
        encoder_out = self.encoder(image)
        preds = self.decoder.predict(encoder_out, tgt)
        return preds


class ModelInterface:
    def __init__(self, configs, model_path='best_model.pth', num_classes=1):
        """
        initiates an object as an interface for inferring an input image

        @param configs: input loaded config of trained model
        @param model_path: path of fine-tuned model
        @param num_classes: number of classes
        """
        self.configs = configs
        self.device = self.configs.train_settings.device
        self.model_path = model_path
        self.num_classes = num_classes
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        tokenizer = Tokenizer(num_classes=self.num_classes, num_bins=self.configs.train_settings.num_bins,
                              width=self.configs.train_settings.img_size, height=self.configs.train_settings.img_size,
                              max_len=self.configs.pix2seq_model.decoder.max_len)
        return tokenizer

    def load_model(self):
        encoder = Encoder(model_name=self.configs.pix2seq_model.encoder.model_name,
                          model_type=self.configs.pix2seq_model.encoder.model_type,
                          pretrained=True,
                          out_dim=self.configs.pix2seq_model.decoder.dimension,
                          img_size=self.configs.pix2seq_model.encoder.img_size)
        decoder = Decoder(vocab_size=self.tokenizer.vocab_size,
                          encoder_length=self.configs.pix2seq_model.decoder.num_patches,
                          dim=self.configs.pix2seq_model.decoder.dimension,
                          num_heads=self.configs.pix2seq_model.decoder.num_heads,
                          num_layers=self.configs.pix2seq_model.decoder.num_layers,
                          max_len=self.configs.pix2seq_model.decoder.max_len,
                          pad_idx=self.tokenizer.PAD_code,
                          device=self.configs.train_settings.device,
                          dim_feedforward=self.configs.pix2seq_model.decoder.dim_feedforward,
                          pretrained=self.configs.pix2seq_model.decoder.pretrained.pretrained,
                          pretrained_path=self.configs.pix2seq_model.decoder.pretrained.pretrained_path)
        model = EncoderDecoder(encoder, decoder)

        model.load_state_dict(torch.load(self.model_path)['model_state_dict'])

        model.to('cuda')

        return model

    def inference(self, original_img):
        """
        gets an RGB image (h*w*c) and returns its predicted class

        @param original_img: input image
        @return: predicted label
        """

        classes = ['floor']
        id2cls = {i: cls_name for i, cls_name in enumerate(classes)}

        transforms = albumentations.Compose(
            [albumentations.Resize(self.configs.train_settings.img_size, self.configs.train_settings.img_size),
             albumentations.Normalize()])

        img = transforms(image=original_img)['image']
        img = torch.FloatTensor(img).permute(2, 0, 1)

        with torch.no_grad():
            batch_preds, batch_confs = generate(
                self.model, img[None, :, :, :], self.tokenizer, max_len=self.configs.pix2seq_model.generation_steps,
                top_k=0, top_p=1)
            bboxes, labels, confs = postprocess(
                batch_preds, batch_confs, self.tokenizer)

        if bboxes[0] is not None:
            img = cv2.resize(original_img, (self.configs.train_settings.img_size, self.configs.train_settings.img_size))
            img = visualize(img, bboxes[0], labels[0], id2cls, display=True, mode='line')
            img = cv2.resize(img, (self.configs.train_settings.img_size * 2, self.configs.train_settings.img_size * 2))

        return img, len(bboxes)
