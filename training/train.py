import datetime
import os
from pathlib import Path
import numpy as np
import yaml
import argparse
from time import time
import shutil
from utils import load_configs, test_gpu_cuda, prepare_dataloaders, prepare_tensorboard, \
    prepare_optimizer, save_best_model_checkpoint, save_checkpoint, get_logging, prepare_configs, optimizer_to
from optimizer import gsam_function, sam_function
from metrics import calculate_metric

from model import *


def train(tools):
    """
    train the model for one epoch

    @param tools: a dictionary of selected configs
    @return: train loss
    """
    # Set model to training mode and define training metrics
    tools['net'].train()
    tools['net'].to(tools['train_device'])

    for param in tools['net'].parameters():
        param.grad = None

    train_loss = 0.0
    target_seqs = torch.zeros((len(tools['train_loader'].dataset), tools['tokenizer'].max_len))
    predicted_seqs = torch.zeros(
        (len(tools['train_loader'].dataset), tools['tokenizer'].max_len - 1, tools['tokenizer'].vocab_size))

    for idx, data in enumerate(tools['train_loader']):
        img, seq = data
        img, seq = img.to(tools['train_device']), seq.to(tools['train_device'])
        seq_input = seq[:, :-1]
        seq_expected = seq[:, 1:]

        if tools['sam_option']:
            loss, preds = sam_function(model=tools['net'],
                                       img=img,
                                       seq_input=seq_input,
                                       seq_expected=seq_expected,
                                       criterion=tools['loss_function'],
                                       scaler=tools["scaler"],
                                       mixed_precision=tools["mixed_precision"],
                                       sam_optimizer=tools['optimizer'],
                                       grad_clip=tools['grad_clip'])
            target_seqs[idx * tools['train_batch_size']: min((idx + 1) * tools['train_batch_size'],
                                                             len(target_seqs))] = seq
            predicted_seqs[idx * tools['train_batch_size']: min((idx + 1) * tools['train_batch_size'],
                                                                len(target_seqs))] = preds

        elif tools['gsam_option']:
            loss, preds = gsam_function(loss_fn=tools['loss_function'],
                                        inputs=img,
                                        seq_input=seq_input,
                                        seq_expected=seq_expected,
                                        scaler=tools["scaler"],
                                        mixed_precision=tools["mixed_precision"],
                                        grad_clip=tools['grad_clip'],
                                        gsam_optimizer=tools['optimizer'],
                                        lr_scheduler=tools['scheduler'])
            target_seqs[idx * tools['train_batch_size']: min((idx + 1) * tools['train_batch_size'],
                                                             len(target_seqs))] = seq
            predicted_seqs[idx * tools['train_batch_size']: min((idx + 1) * tools['train_batch_size'],
                                                                len(target_seqs))] = preds

        else:
            with torch.cuda.amp.autocast(enabled=tools["mixed_precision"]):
                preds = tools['net'](img, seq_input)
                target_seqs[idx * tools['train_batch_size']: min((idx + 1) * tools['train_batch_size'],
                                                                 len(target_seqs))] = seq
                predicted_seqs[idx * tools['train_batch_size']: min((idx + 1) * tools['train_batch_size'],
                                                                    len(target_seqs))] = preds
                loss = tools['loss_function'](preds.reshape(-1, preds.shape[-1]), seq_expected.reshape(-1))

            tools["scaler"].scale(loss / tools['accum_iter']).backward()
            if ((idx + 1) % tools['accum_iter'] == 0) or (idx + 1 == len(tools['train_loader'])):
                tools["scaler"].unscale_(tools["optimizer"])
                torch.nn.utils.clip_grad_norm_(tools['net'].parameters(), max_norm=tools['grad_clip'])
                tools["scaler"].step(tools["optimizer"])
                tools["scaler"].update()
                tools['optimizer'].zero_grad()  # Sets the gradients of all optimized torch.Tensor s to zero

        # scheduler step
        if tools['scheduler'] is not None and not tools['gsam_option']:
            tools['scheduler'].step()

        train_loss += loss.data.item() * img.size(0)

        tools['train_writer'].add_scalar('step loss', loss.data.item(),
                                         idx + tools['epoch'] * len(tools['train_loader']))
        tools['train_writer'].add_scalar('learning rate variation', tools['optimizer'].param_groups[0]['lr'],
                                         idx + tools['epoch'] * len(tools['train_loader']))

    metrics = calculate_metric(predicted_seqs, target_seqs, tools['tokenizer'], len(tools['train_loader'].dataset),
                               tools['logging'], normalize=True, img_size=tools['img_size'])
    train_loss = train_loss / len(tools['train_loader'].dataset)

    tools['train_writer'].add_scalar('correct dimension accuracy', metrics[0], tools['epoch'])
    tools['train_writer'].add_scalar('in range labels accuracy', metrics[1], tools['epoch'])
    tools['train_writer'].add_scalar('labels classification accuracy', metrics[2], tools['epoch'])
    tools['train_writer'].add_scalar('bounding boxes loss', metrics[3], tools['epoch'])
    tools['train_writer'].add_scalar('bounding boxes loss (unoptimized)', metrics[4], tools['epoch'])
    tools['train_writer'].add_scalar('in range xy accuracy', metrics[5], tools['epoch'])
    tools['train_writer'].add_scalar('difference of distances of points', metrics[6], tools['epoch'])
    tools['train_writer'].add_scalar('difference of number of predicted points', metrics[7], tools['epoch'])
    tools['train_writer'].add_scalar('loss', np.round(train_loss, 4), tools['epoch'])

    return train_loss


def valid(tools):
    """
    validates the model on valid dataset

    @param tools: a dictionary of selected configs
    @return: validation loss and metrics
    """
    # Set model to evaluation mode
    tools['net'].eval()
    valid_loss = 0

    target_seqs = torch.zeros((len(tools['valid_loader'].dataset), tools['tokenizer'].max_len))
    predicted_seqs = torch.zeros(
        (len(tools['valid_loader'].dataset), tools['tokenizer'].max_len - 1, tools['tokenizer'].vocab_size))

    # Batch loop for validation
    for idx, data in enumerate(tools['valid_loader'], 0):
        img, seq = data
        img, seq = img.to(tools['train_device']), seq.to(tools['train_device'])
        seq_input = seq[:, :-1]
        seq_expected = seq[:, 1:]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=tools["mixed_precision"]):
                preds = tools['net'](img, seq_input)

                target_seqs[idx * tools['valid_batch_size']: min((idx + 1) * tools['valid_batch_size'],
                                                                 len(target_seqs))] = seq
                predicted_seqs[idx * tools['valid_batch_size']: min((idx + 1) * tools['valid_batch_size'],
                                                                    len(target_seqs))] = preds

                loss = tools['loss_function'](preds.reshape(-1, preds.shape[-1]), seq_expected.reshape(-1))

        valid_loss += loss.data.item() * img.size(0)

    metrics = calculate_metric(predicted_seqs, target_seqs, tools['tokenizer'], len(tools['valid_loader'].dataset),
                               tools['logging'], normalize=True, img_size=tools['img_size'])
    valid_loss = valid_loss / len(tools['valid_loader'].dataset)

    if not tools['evaluate']:
        tools['valid_writer'].add_scalar('correct dimension accuracy', metrics[0], tools['epoch'])
        tools['valid_writer'].add_scalar('in range labels accuracy', metrics[1], tools['epoch'])
        tools['valid_writer'].add_scalar('labels classification accuracy', metrics[2], tools['epoch'])
        tools['valid_writer'].add_scalar('bounding boxes loss', metrics[3], tools['epoch'])
        tools['valid_writer'].add_scalar('bounding boxes loss (unoptimized)', metrics[4], tools['epoch'])
        tools['valid_writer'].add_scalar('in range xy accuracy', metrics[5], tools['epoch'])
        tools['valid_writer'].add_scalar('difference of distances of points', metrics[6], tools['epoch'])
        tools['valid_writer'].add_scalar('difference of number of predicted points', metrics[7], tools['epoch'])
        tools['valid_writer'].add_scalar('loss', np.round(valid_loss, 4), tools['epoch'])

    return valid_loss, metrics


def main(dict_config):
    """
    Gets a config file and run the code

    @param dict_config: a dictionary for configs
    @return: None
    """
    configs = load_configs(dict_config)
    prepare_configs(configs)

    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    test_gpu_cuda()

    # creates a directory to save config and checkpoints
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    if configs.evaluate:
        run_id += '_evaluation'
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    shutil.copy('config.yaml', result_path)
    logging = get_logging(result_path)

    # prepare dataloader
    train_loader, valid_loader, tokenizer = prepare_dataloaders(configs)

    # creates the model
    encoder = Encoder(model_name=configs.pix2seq_model.encoder.model_name,
                      model_type=configs.pix2seq_model.encoder.model_type,
                      pretrained=True,
                      out_dim=configs.pix2seq_model.decoder.dimension,
                      img_size=configs.pix2seq_model.encoder.img_size,
                      patch_size=configs.pix2seq_model.encoder.patch_size)
    logging.warning('Encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))
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
    logging.warning('Decoder parameters: ' + str(sum(p.numel() for p in decoder.parameters())))
    net = EncoderDecoder(encoder, decoder)
    logging.warning('All parameters: ' + str(sum(p.numel() for p in net.parameters())))

    # creates optimizer and scheduler
    optimizer, scheduler = prepare_optimizer(net, configs, len(train_loader))

    # prepares the code for resuming from a checkpoint
    start_epoch = 0
    if configs.resume.resume:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        net.load_state_dict(model_checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
            start_epoch = model_checkpoint['epoch'] + 1
        logging.warning('Model is loaded to resume training!')
        optimizer_to(optimizer, configs.train_settings.device)
        net.to(configs.train_settings.device)

    # initialize tensorboards
    train_writer, valid_writer = prepare_tensorboard(result_path, net, train_loader)

    # creates loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_code)

    tools = {
        'net': net,
        'architecture': configs.architecture,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'train_device': configs.train_settings.device,
        'valid_device': configs.valid_settings.device,
        'train_batch_size': configs.train_settings.batch_size,
        'valid_batch_size': configs.valid_settings.batch_size,
        'img_size': configs.pix2seq_model.encoder.img_size,
        'optimizer': optimizer,
        'mixed_precision': configs.train_settings.mixed_precision,
        'tensorboard_log': False,
        'train_writer': train_writer,
        'valid_writer': valid_writer,
        'scaler': torch.cuda.amp.GradScaler(enabled=configs.train_settings.mixed_precision),
        'sam_option': configs.train_settings.sam,
        'gsam_option': configs.train_settings.gsam,
        'accum_iter': configs.train_settings.accum_iter,
        'loss_function': criterion,
        'grad_clip': configs.optimizer.grad_clip_norm,
        'checkpoints_every': configs.checkpoints_every,
        'scheduler': scheduler,
        'result_path': result_path,
        'checkpoint_path': checkpoint_path,
        'tokenizer': tokenizer,
        'logging': logging,
        'evaluate': configs.evaluate
    }

    # starts training procedure
    best_valid = 100000
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        tools['epoch'] = epoch

        start_time = time()
        train_loss = train(tools)
        end_time = time()
        logging.warning(
            f'epoch {epoch} - time {np.round(end_time - start_time, 2)}s, train loss {np.round(train_loss, 4)}')

        if epoch % configs.valid_settings.do_every == 0:
            start_time = time()
            valid_loss, valid_metrics = valid(tools)
            end_time = time()
            logging.warning(
                f'evaluation - time {np.round(end_time - start_time, 2)}s, valid loss {np.round(valid_loss, 4)}')

            if valid_metrics[3] + valid_metrics[5] < best_valid and epoch != 0:
                best_valid = valid_metrics[3] + valid_metrics[5]
                save_best_model_checkpoint(epoch, tools, logging)

        save_checkpoint(epoch, tools)

    # evaluates the model
    if configs.evaluate:
        start_time = time()
        valid_loss, valid_metrics = valid(tools)
        end_time = time()
        logging.warning(
            f'evaluation - time {np.round(end_time - start_time, 2)}s, valid loss {np.round(valid_loss, 4)}')

    # closes writers
    train_writer.close()
    valid_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Corner Detection")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file)
    print('done!')
