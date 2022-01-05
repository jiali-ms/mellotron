import os
import time
import argparse
import math
from numpy import finfo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from model import load_model
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams,
                           speaker_ids=trainset.speaker_ids)

    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    
    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=4, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    train_sampler = DistributedSampler(trainset)

    return train_loader, valset, collate_fn, train_sampler


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(rank, checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def parse_batch(device, batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, speaker_ids, f0_padded = batch
    text_padded = text_padded.to(device) #to_gpu(text_padded).long()
    input_lengths = input_lengths.to(device) #to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = mel_padded.to(device) #to_gpu(mel_padded).float()
    gate_padded = gate_padded.to(device) #to_gpu(gate_padded).float()
    output_lengths = output_lengths.to(device) #to_gpu(output_lengths).long()
    speaker_ids = speaker_ids.to(device) #to_gpu(speaker_ids.data).long()
    f0_padded = f0_padded.to(device) # to_gpu(f0_padded).float()
    return ((text_padded, input_lengths, mel_padded, max_len,
                output_lengths, speaker_ids, f0_padded),
            (mel_padded, gate_padded))

def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = parse_batch(rank, batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss = loss.item()
            val_loss += loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(rank, output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank_, group_name, hparams):

    print(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, group_name)
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    #if hparams.distributed_run:
        #init_distributed(hparams, n_gpus, rank, group_name)

    init_process_group(backend="nccl", init_method="tcp://localhost:54321",
                           world_size=1 * n_gpus, rank=rank)
    device = torch.device('cuda:{:d}'.format(rank))

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams).to(device)
    model = DistributedDataParallel(model, device_ids=[rank]).to(device)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn, train_sampler = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        print('load checkpoint ', checkpoint_path)
        model, optimizer, _learning_rate, iteration = load_checkpoint(rank, 
            checkpoint_path, model, optimizer)
        if hparams.use_saved_learning_rate:
            learning_rate = _learning_rate
        iteration += 1  # next iteration is iteration + 1
        epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            if iteration > 0 and iteration % hparams.learning_rate_anneal == 0:
                learning_rate = max(
                    hparams.learning_rate_min, learning_rate * 0.5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = parse_batch(rank, batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)            
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, loss, grad_norm, duration))
                logger.log_training(
                    loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % 50 == 0):
                validate(model, criterion, valset, iteration,
                        hparams.batch_size, n_gpus, collate_fn, logger,
                        hparams.distributed_run, rank)
                if rank == 0 and (iteration % hparams.iters_per_checkpoint == 0):
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=2,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    #train(args.output_directory, args.log_directory, args.checkpoint_path, args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
    mp.spawn(train, nprocs=args.n_gpus, args=(args.output_directory, args.log_directory, args.checkpoint_path, args.warm_start, args.n_gpus, args.rank, args.group_name, hparams))