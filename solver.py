


import os

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, confusion_matrix

from apex import amp
from utils import check_loss, reduce_tensor


class Solver(object):

    def __init__(self, model, data, criterion, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object nn.Model
        - data: A dictionary of training and validation data containing:
          'train': Object of SpectrogramDataset API.
          'val': Object of SpectrogramDataset API.
        - criterion: loss object.

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
        self.criterion = criterion
        self.data_train = data['train']
        self.data_val = data['val']

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

        self.continue_from = kwargs.pop('continue_from', None)

        self.labels = kwargs.pop('labels', None)  # It shouldn't be here.

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)


        self.train_sampler = None
        self.optimizer = None
        self.distributed = None


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

        out, output_sizes = self.model(inputs, input_sizes)

        out = out.transpose(0, 1)  # TxNxH

        float_out = out.float()  # ensure float32 for loss.

        float_out_last = float_out[output_sizes.long() - 1, torch.arange(float_out.size(1)), :]

        loss = self.criterion(float_out_last, targets.long())
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

    def __save_checkpoint(self):
        print('Not implemented.')


    def check_accuracy(self, inputs, input_sizes, targets, y_true=None, y_pred=None):
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

        if y_true is None or y_pred is None:
            out, output_sizes = self.model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            float_out = out.float()
            float_out_last = float_out[output_sizes.long() - 1, torch.arange(float_out.size(1)), :]

            probs = torch.nn.functional.softmax(float_out_last, dim=1)
            _, pred_labels = torch.max(probs, 1)
            y_true = targets.cpu()
            y_pred = pred_labels.cpu()

        return accuracy_score(y_true, y_pred) * 100.0

    def train(self, **kwargs):
        """
        Run optimization to train the model.
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


        if self.continue_from:
            print('Not implemented continued from yet. Do save checkpoint model first.')
        else:


        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):

            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                       t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                    num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                    num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                           self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
