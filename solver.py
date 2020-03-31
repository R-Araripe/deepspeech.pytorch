


import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, confusion_matrix

from apex import amp
from apex.parallel import DistributedDataParallel
from data.data_loader import (AudioDataLoader, BucketingSampler,
                              DistributedBucketingSampler,
                              RandomBucketingSampler, SpectrogramDataset)
from logger import TensorBoardLogger


class Solver(object):

    def __init__(self, model, data, decoder, **kwargs):
        """
        Construct a new Solver instance.

        Required:
        - model: A model object nn.Model
        - data: A dictionary of training and validation data containing:
          'train': Object of SpectrogramDataset API.
          'val': Object of SpectrogramDataset API.
        - decoder: Object that decode model output to predicted labels.


        Optional:
        - audio_conf (dict)
        - speed_volume_perturb (bool): Use random tempo and gain perturbations.
        - spec_augment (bool): Use simple spectral augmentation on mel spectograms.
        - num_workers (int): Number of workers used in data-loading.
        - loss_scale: Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients.


        Optional arguments:
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - num_epochs: The number of epochs to run for during training.
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        """
        self.model = model
        self.data_train = data['train']
        self.data_val = data['val']
        self.decoder = decoder

        # Unpack keyword arguments
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 5)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', None)  # will be removed soon.
        self.num_val_samples = kwargs.pop('num_val_samples', None)  #

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.print_every = kwargs.pop('print_every', 1)
        self.verbose = kwargs.pop('verbose', True)
        self.device = kwargs.pop('device', 'cuda')
        self.max_norm = kwargs.pop('max_norm', 400)
        self.sufix = kwargs.pop('sufix', None)
        self.save_dir = kwargs.pop('save_dir', None)


        self.audio_conf = kwargs.pop('audio_conf', None)
        self.train_manifest = kwargs.pop('train_manifest', None)
        self.val_manifest = kwargs.pop('val_manifest', None)
        self.metadata_path = kwargs.pop('metadata_path', None)
        self.labels = kwargs.pop('labels', None)
        self.speed_volume_perturb = kwargs.pop('speed_volume_perturb', False)
        self.spec_augment = kwargs.pop('spec_augment', False)

        self.num_workers = kwargs.pop('num_workers', 16)
        self.sampler_type = kwargs.pop('sampler_type', 'random')
        self.batch_size = kwargs.pop('batch_size', 5)
        self.start_epoch = kwargs.pop('start_epoch', 0)

        self.tensorboard = kwargs.pop('tensorboard', True)
        self.generate_graph = kwargs.pop('generate_graph', False)
        self.id = kwargs.pop('id', 'libri')
        self.log_dir = kwargs.pop('log_dir', None)
        self.log_params = kwargs.pop('log_params', True)

        self.continue_from = kwargs.pop('continue_from', None)
        self.optim_state = kwargs.pop('optim_state', None)
        self.amp_state = kwargs.pop('amp_state', None)

        self.lr = kwargs.pop('lr', 3e-4)
        self.reg = kwargs.pop('reg', 1e-2)
        self.criterion_type = kwargs.pop('criterion_type', 'cross_entropy_loss')
        self.learning_anneal = kwargs.pop('learning_anneal', 1.01)

        self.opt_level = kwargs.pop('opt_level', 'O1')
        self.loss_scale = kwargs.pop('loss_scale', 1)
        self.keep_batchnorm_fp32 = kwargs.pop('keep_batchnorm_fp32', None)

        self.intra_epoch_sanity_check = kwargs.pop('intra_epoch_sanity_check', True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)


        self.optimizer = None
        self.distributed = None
        self.criterion = None



        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p, val in self.model.named_parameters:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d


    def _step(self, inputs, input_sizes, targets):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.

        Parameters
        ----------

        inputs:
        inputs_sizes:
        targets:

        """

        output = self.model(inputs, input_sizes)

        loss = self.criterion(output, targets.long())

        loss = loss / inputs.size(0)  # average the loss by minibatch

        if self.distributed:
            loss = loss.to(self.device)
            loss_value = reduce_tensor(loss, self.world_size).item()
        else:
            loss_value = loss.item()

        # Check to ensure valid loss was calculated
        valid_loss, error = check_loss(loss, loss_value)

        if valid_loss:

            self.optimizer.zero_grad()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_norm)

            self.optimizer.step()
        else:
            print(error)
            print('Skipping grad update')
            loss_value = 0

        return output, loss_value

    def __save_checkpoint(self):
        print('Not implemented.')


    def check_accuracy(self, targets, inputs=None, input_sizes=None, y_pred=None):
        """
        Check accuracy of the model on the provided data.

        Parameters
        ----------

        - inputs:
        - inputs_sizes:
        - targets:

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        if y_pred is None:
            output = self.model(inputs, input_sizes)
            y_pred = self.decoder.decode(output).cpu()

        return accuracy_score(targets, y_pred) * 100.0, y_pred

    def train(self, **kwargs):
        """
        Run optimization to train the model.

        Parameters
        ----------


        """
        world_size = kwargs.pop('world_size', 1)
        gpu_rank = kwargs.pop('gpu_rank', 0)
        rank = kwargs.pop('rank', 0)
        dist_backend = kwargs.pop('dist_backend', 'nccl')
        dist_url = kwargs.pop('dist_url', None)

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '1234'

        self.distributed = world_size > 1

        if self.distributed:
            if self.gpu_rank:
                torch.cuda.set_device(int(gpu_rank))
            dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                    world_size=world_size, rank=rank)
            print('Initiated process group')
            main_proc = rank == 0  # Only the first proc should save models

        if main_proc and self.tensorboard:
            tensorboard_logger = TensorBoardLogger(self.id, self.log_dir, self.log_params, comment=self.sufix)

        train_dataset = SpectrogramDataset(audio_conf=self.audio_conf, manifest_filepath=self.train_manifest, metadata_file_path=self.metadata_path, labels=self.labels, normalize=True, speed_volume_perturb=self.speed_volume_perturb, spec_augment=self.spec_augment)

        val_dataset = SpectrogramDataset(audio_conf=self.audio_conf, manifest_filepath=self.val_manifest,  metadata_file_path=self.metadata_path, labels=self.labels, normalize=True, speed_volume_perturb=False, spec_augment=False)


        if self.distributed:
            train_sampler = DistributedBucketingSampler(train_dataset, batch_size=self.batch_size, num_replicas=world_size, rank=rank)
        else:
            if self.sampler_type == 'bucketing':
                train_sampler = BucketingSampler(train_dataset, batch_size=self.batch_size, shuffle=True)
            if self.sampler_type == 'random':
                train_sampler = RandomBucketingSampler(train_dataset, batch_size=self.batch_size)

        print("Shuffling batches for the following epochs..")
        train_sampler.shuffle(self.start_epoch)

        train_loader = AudioDataLoader(train_dataset, num_workers=self.num_workers, batch_sampler=train_sampler)
        val_loader = AudioDataLoader(val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=True)

        if self.tensorboard and self.generate_graph:  # TO DO get some audios also
            with torch.no_grad():
                inputs, targets, input_percentages, target_sizes = next(iter(train_loader))
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                tensorboard_logger.add_image(inputs, input_sizes, targets, network=self.model)

        self.model = self.model.to(self.device)
        parameters = self.model.parameters()

        if self.update_rule == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.reg)
        if self.update_rule == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.reg)


        self.model, self.optimizer = amp.initialize(self.model, optimizer, opt_level=self.opt_level, keep_batchnorm_fp32=self.keep_batchnorm_fp32, loss_scale=self.loss_scale)

        if self.optim_state is not None:
            self.optimizer.load_state_dict(self.optim_state)

        if self.amp_state is not None:
            amp.load_state_dict(self.amp_state)

        if self.distributed:
            self.model = DistributedDataParallel(self.model)

        print(self.model)
        print("Number of parameters: %d" % DeepSpeech.get_param_size(self.model))

        if self.criterion_type == 'cross_entropy_loss':
            self.criterion = torch.nn.CrossEntropyLoss()

        #  Useless for now because I don't save.
        accuracies_train_iters = []
        losses_iters = []
        loss_epochs = []
        accuracy_train_epochs = []
        accuracy_val_epochs = []

        avg_loss = 0
        batch_time = AverageMeter()
        epoch_time = AverageMeter()
        losses = AverageMeter()
        start_training = time.time()
        for epoch in range(self.start_epoch, self.epochs):

            # Put model in train mode
            self.model.train()

            y_true_train_epoch = np.array([])
            y_pred_train_epoch = np.array([])

            start_epoch = time.time()
            for i, (data) in enumerate(train_loader, start=0):
                start_batch = time.time()
                if i == len(train_sampler):  # QUE pq isso deus
                    break

                inputs, targets, input_percentages, target_sizes = data

                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                output, loss_value = self._step(self, inputs, input_sizes, targets)

                y_pred = self.decoder.decode(output)

                y_true_train_epoch = np.concatenate((y_true_train_epoch, targets.cpu().numpy()))  # maybe I should do it with tensors?
                y_pred_train_epoch = np.concatenate((y_pred_train_epoch, y_pred.numpy()))

                if self.intra_epoch_sanity_check:

                    acc, _ = self.check_accuracy(targets.cpu(), y_pred=y_pred.cpu())
                    accuracies_train_iters.append(acc)
                    losses_iters.append(loss_value)

                    cm = confusion_matrix(targets.cpu(), y_pred.cpu(), labels=self.labels)
                    print('Confusion matrix train step:')
                    print(pd.DataFrame(cm))

                    if self.tensorboard:
                        tensorboard_logger.update(len(train_loader) * epoch + i, {
                            'Loss/through_iterations': loss_value,
                            'Accuracy/train_through_iterations': acc
                        })

                del output

                batch_time.update(time.time() - start_batch)

            epoch_time.update(time.time() - start_epoch)
            losses.update(loss_value, inputs.size(0))

            # Write elapsed time (and loss) to terminal
            print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Epoch {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=epoch_time, loss=losses))

            # Loss log
            avg_loss += loss_value
            loss_epochs.append(avg_loss)

            # Accuracy train log
            acc_train, _ = self.check_accuracy(y_true_train_epoch, y_pred=y_pred_train_epoch)
            accuracy_train_epochs.append(acc_train)

            # Accuracy val log
            for data in val_loader:
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            acc_val, y_pred_val = self.check_accuracy(targets.cpu(), inputs=inputs, input_sizes=input_sizes)
            accuracy_val_epochs.append(acc_val)
            cm = confusion_matrix(targets.cpu(), y_pred, labels=self.labels)
            print('Confusion matrix validation:')
            print(pd.DataFrame(cm))

            # Write epoch stuff to tensorboard
            if self.tensorboard:
                tensorboard_logger.update(epoch, {'Loss/through_epochs': avg_loss}, parameters=self.model.named_parameters)

                tensorboard_logger.update(epoch, {
                    'train': acc_train,
                    'validation': acc_val
                }, together=True, name='Accuracy/through_epochs')

            # Anneal learning rate. TO DO: find better way to this this specific to every parameter as cs231n does.
            for g in self.optimizer.param_groups:
                g['lr'] = g['lr'] / self.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

            # Shuffle batches order
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)

            # Rechoose batches elements
            if self.sampler_type == 'random':
                train_sampler.recompute_bins()


        end_training = time.time()

        if self.tensorboard:
            tensorboard_logger.close()

        print('Elapsed time in training: %.02f '%((end_training - start_training) / 60.0) )
