import argparse
import json
import os
import random
import time
from datetime import datetime
from pprint import pprint
from test import evaluate

import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
from sklearn.metrics import accuracy_score, confusion_matrix
from warpctc_pytorch import CTCLoss

from data.data_loader import (AudioDataLoader, BucketingSampler,
                              DistributedBucketingSampler,
                              RandomBucketingSampler, SpectrogramDataset)
from decoder import GreedyDecoder, MyDecoder
from logger import TensorBoardLogger, VisdomLogger
from model import DeepSpeech, supported_rnns
from utils import check_loss, reduce_tensor

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='../Data/downsampled-16k/manifest_train_vowels-42.txt')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='../Data/downsampled-16k/manifest_val_vowels-42.txt')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=16, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=1024, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.01, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='libri', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='../Data/Models/deepspeech.pytorch/parkinson-vowels/', help='Location to save epoch models')
parser.add_argument('--model-path', default='../Data/Models/deepspeech.pytorch/parkinson-vowels/deepspeech_final.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true', help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', default='O1', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', default=1,
                    help='Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parser.parse_args()

    print('AAAAAAAHHHHH IT STARTED')

    PATH_DATA = '/home/rebeca.araripe/data/FinalProject'

    # Try to add keys to arg
    args.cuda = True
    args.visdom = False
    args.checkpoint = True
    args.finetune = True
    args.no_sorta_grad = True
    args.continue_from = os.path.join(PATH_DATA, 'Models/librispeech_pretrained_v2.pth')
    metadata_path = os.path.join(PATH_DATA, 'raw/PCGITA_metadata.xlsx')
    args.log_dir = os.path.join(PATH_DATA, 'Results/visualize/deepspeech')
    args.cuda = True
    which_cuda = 'cuda'

    args.tensorboard = True
    args.log_params = True  # for now while I fix the other stuff
    generate_graph = False
    args.epochs = 15
    reg = 1e-2
    args.batch_size = 32
    args.lr = 3e-4  # 3e-4 default
    opt_alg = 'sgd'
    sampler = 'random' # random, bucketing

    freeze_conv = False
    freeze_rnns = True

    data_category = 'vowels'  # vowels, read-text, monologues
    data_subcategory = 'A'  # AEIOU

    args.train_manifest = os.path.join(PATH_DATA, 'downsampled-16k/manifest_train_%s-42-%s.txt'%((data_category, data_subcategory)))
    args.val_manifest = os.path.join(PATH_DATA, 'downsampled-16k/manifest_val_%s-42-%s.txt'%((data_category, data_subcategory)))


    # Create sufix for logging
    sufix = '%s-data=%s-batchsize=%i-reg=%.2E-freeze_conv=%s-freeze_rnns=%s-opt_alg=%s-sampler=%s'%((str(datetime.now()), data_category + data_subcategory, args.batch_size, reg, str(freeze_conv), str(freeze_rnns), opt_alg, sampler))


    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(which_cuda if args.cuda else "cpu")
    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device(which_cuda if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    # loss_results, acc_results, std_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
    #     args.epochs)
    # best_wer = None
    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
    if main_proc and args.tensorboard:
        # import pdb; pdb.set_trace()
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params, comment=sufix)

    avg_loss, start_epoch, start_iter, optim_state, amp_state = 0, 0, 0, None, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = model.labels
        audio_conf = model.audio_conf
        if not args.finetune:  # Don't want to restart training
            # Nao eh finetuning. Entao os parametros de otimizacao (repare que nao os pesos,que sao o state_dict e esses foram carregados) nao voltam ao estado inicial. O modelo eh retreinado mas com a inicializacao dos pesos de antes. TO DO: quando for usar pro meu, checar as coisas de batch normalization, a running average nao pode ser usada por exemplo, ver se ela ta no state ou no optim.
            optim_state = package['optim_dict']
            # amp_state = package['amp']  # what is it?
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
            avg_loss = int(package.get('avg_loss', 0))
            # loss_results, acc_results, std_results = package['loss_results'], package['cer_results'], \
            #                                          package['wer_results']
            # best_wer = std_results[start_epoch]
            if main_proc and args.visdom:  # Add previous scores to visdom graph
                visdom_logger.load_previous_values(start_epoch, package)
            if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
                tensorboard_logger.load_previous_values_libri(start_epoch, package)
    else:

        labels = [0, 1]

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)

    decoder = MyDecoder(labels)

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, metadata_file_path=metadata_path, labels=labels,
                                       normalize=True, speed_volume_perturb=args.speed_volume_perturb, spec_augment=args.spec_augment)

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest,  metadata_file_path=metadata_path, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    print('\n Num samples train dataset: ', len(train_dataset))
    print('\n Num samples val   dataset: ', len(test_dataset))


    if not args.distributed:
        if sampler == 'bucketing':
            train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
        if sampler == 'random':
            train_sampler = RandomBucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size, num_replicas=args.world_size, rank=args.rank)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True)


    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    if args.tensorboard and generate_graph:  # TO DO get some audios also
        with torch.no_grad(): # sla vai que ne
            inputs, targets, input_percentages, target_sizes = next(iter(train_loader))
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            tensorboard_logger.add_image(inputs, input_sizes, targets, network=model) # add graph doesn't work if model is in gpu

    if freeze_conv:
        model.conv.requires_grad_(requires_grad=False)

    # import pdb; pdb.set_trace()

    if freeze_rnns:
        model.rnns.requires_grad_(requires_grad=False)




    model = model.to(device)
    parameters = model.parameters()

    if opt_alg == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=reg)
    if opt_alg == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=reg)


    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    if amp_state is not None:
        amp.load_state_dict(amp_state)

    if args.distributed:
        model = DistributedDataParallel(model)
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    loss_epochs = []  # list of list epochs->iterations in epoch
    avg_loss_epochs = [] # list
    accuracy_train_in_epochs = [] # list of list
    accuracy_train_epochs = [] # list
    accuracy_val_epochs = []  # list

    start_epochs = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        losses_epoch = []
        y_true_train_epoch = np.array([])
        y_pred_train_epoch = np.array([])
        accuracies_train_epoch = []
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):  # QUE pq isso deus
                break
            inputs, targets, input_percentages, target_sizes = data
            # print('inputs: ', inputs)
            # print('targets: ', targets)
            # print('input size 0: ', inputs.size(0))
            # print('input_percentages: ', input_percentages)  # de onde isso vem??? ah, acho que eh o tamanho do audio
            # print('target sizes: ', target_sizes)
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # print('input_sizes: ', input_sizes)
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)
            targets = targets.to(device)

            out, output_sizes = model(inputs, input_sizes)

            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss. They don't seem to be normalized :)
            # print('float out: ', float_out)
            # print('output sizes ', output_sizes)

            # POR QUE SEMPRE OS MESMOS VALORES PRA TODOS MUNDO?? EXCETO O PRIMEIRO EXEMPLO DO MINIBATCH. EU DEVERIA MSM PEGAAR O ULTIMO?
            float_out_last = float_out[output_sizes.long() - 1, torch.arange(float_out.size(1)), :]
            # float_out_last = float_out[-1, :, :]
            # import pdb; pdb.set_trace()

            loss = criterion(float_out_last, targets.long()).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            # Sanity check
            probs = torch.nn.functional.softmax(float_out_last, dim=1)
            _, pred_labels = torch.max(probs, 1)
            y_true = targets.cpu()
            y_pred = pred_labels.cpu()

            y_true_train_epoch = np.concatenate((y_true_train_epoch, y_true.numpy()))  # maybe I should do it with tensors?
            y_pred_train_epoch = np.concatenate((y_pred_train_epoch, y_pred.numpy()))

            accuracy_train = accuracy_score(y_true, y_pred) * 100.0
            accuracies_train_epoch.append(accuracy_train)
            print('accuracy train step: ', accuracy_train)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            print('confusion matrix train step:')
            print( pd.DataFrame(cm))


            if args.distributed:
                loss = loss.to(device)
                loss_value = reduce_tensor(loss, args.world_size).item()
            else:
                loss_value = loss.item()

            losses_epoch.append(loss_value)

            # Log iteration results
            if args.tensorboard:
                tensorboard_logger.update(len(train_loader) * epoch + i, {
                    'Loss/through_iterations': loss_value,
                    'Accuracy/train_through_iterations': accuracy_train
                })

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:

                optimizer.zero_grad()
                # compute gradient
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))

            del loss, out, float_out

        end_epochs = time.time()

        print('==================================Training total time: %.02f min'%((end_epochs - start_epochs) / 60.0))

        avg_loss /= len(train_sampler)

        avg_loss_epochs.append(avg_loss)
        loss_epochs.append(losses_epoch)

        accuracy_train_in_epochs.append(accuracies_train_epoch)
        accuracy_train = accuracy_score(y_true_train_epoch, y_pred_train_epoch) * 100.0
        accuracy_train_epochs.append(accuracy_train)


        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        with torch.no_grad():
            accuracy_val, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=decoder,
                                             target_decoder=decoder)
        accuracy_val_epochs.append(accuracy_val)

        # Log epoch results
        # import pdb; pdb.set_trace()
        if args.tensorboard:
            tensorboard_logger.update(epoch, {
                'Loss/through_epochs': avg_loss},
                parameters=model.named_parameters)
        if args.tensorboard:
            tensorboard_logger.update(epoch, {
                'train': accuracy_train,
                'validation': accuracy_val
            }, together=True, name='Accuracy/through_epochs')

        print('Validation Summary Epoch: [{0}]\t'
              'Avg Loss {loss:.3f}\t'
              'Train Accuracy {acc_train:.3f}\t'
              'Val Accuracy {acc_val:.3f}\t'.format(
            epoch + 1, loss=avg_loss, acc_train=accuracy_train, acc_val=accuracy_val))


        if main_proc and args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, amp=amp, epoch=epoch),
                       file_path)
        # anneal lr
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        # TO DO: refazer isso do meu jeito eh importante sim salvar o best validated model
        # if main_proc and (best_wer is None or best_wer > acc_mean):
        #     print("Found better validated model, saving to %s" % args.model_path)
        #     torch.save(DeepSpeech.serialize(model, optimizer=optimizer, amp=amp, epoch=epoch, loss_results=loss_results,
        #                                     wer_results=std_results, cer_results=acc_results)
        #                , args.model_path)
        #     best_wer = acc_mean
        #     avg_loss = 0

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
        if sampler == 'random':
            train_sampler.recompute_bins()

    if args.tensorboard:
        tensorboard_logger.close()
