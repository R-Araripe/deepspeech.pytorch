


import os
import random
from datetime import datetime

import numpy as np
import torch

from data.data_loader import SpectrogramDataset
from decoder import DeepSpeechDecoder
from model import DeepSpeech, supported_rnns
from solver import Solver

#------------Settings-----------------------------
transfer = False
tensorboard = True

num_iter_cv = 30

rnn_type = 'lstm'
hidden_size = 1024
hidden_layers = 5
update_rule = 'sgd'  # 'sgd', 'adam'
criterion_type = 'cross_entropy_loss'
batch_size = 7
sampler = 'random'
epochs = 30

seed = 42
device = torch.device('cuda')

labels = [0, 1]
audio_conf = dict(sample_rate=16000,
                  window_size=.02,
                  window_stride=.01,
                  window='hamming',
                  noise_dir=None,
                  noise_prob=0.4,
                  noise_levels=(0.0, 0.5))
bidirectional = True

PATH_DATA = '/home/rebeca.araripe/data/FinalProject'
metadata_path =  os.path.join(PATH_DATA, 'raw/PCGITA_metadata.xlsx')
data_category = 'monologues'  # vowels, read-text, monologues
data_subcategory = ''  # AEIOU
train_manifest = os.path.join(PATH_DATA, 'downsampled-16k/manifest_train_%s-42-%s.txt'%((data_category, data_subcategory)))
val_manifest = os.path.join(PATH_DATA, 'downsampled-16k/manifest_val_%s-42-%s.txt'%((data_category, data_subcategory)))
log_dir = os.path.join(PATH_DATA, 'Results/visualize/deepspeech')

# Not sure about this here
world_size = 1  # doesn't do anything for the moment

#-------------------------------------------------


if transfer:
    rnn_type = 'lstm'
    hidden_size = 1024
    hidden_layers = 5
    audio_conf = dict(sample_rate=16000,
                      window_size=.02,
                      window_stride=.01,
                      window='hamming',
                      noise_dir=None,
                      noise_prob=0.4,
                      noise_levels=(0.0, 0.5))
    bidirectional = True

    id = 'libri'
else:
    id = 'scratch'

best_val = 0.0
for it in range(num_iter_cv):

    lr = 10**np.random.uniform(-4, -2)
    reg = 10**np.random.uniform(-3, -1)

    model = DeepSpeech(rnn_hidden_size=hidden_size, nb_layers=hidden_layers, labels=labels, rnn_type=supported_rnns[rnn_type], audio_conf=audio_conf, bidirectional=bidirectional)

    decoder = DeepSpeechDecoder()

    # Get dataset
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=train_manifest, metadata_file_path=metadata_path, labels=labels, normalize=True, speed_volume_perturb=False, spec_augment=False)

    val_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest,  metadata_file_path=metadata_path, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    print('\n Num samples train dataset: ', len(train_dataset))
    print('\n Num samples val   dataset: ', len(val_dataset))

    # Compose sufix for loging
    sampler = 'distributed' if world_size > 1 else sampler
    sufix = '%s-data=%s-batchsize=%i-reg=%.2E-opt_alg=%s-sampler=%s-transfer=%s'%((str(datetime.now()), data_category + data_subcategory, batch_size, reg, update_rule, sampler, str(transfer)))

    # Set seeds for determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initiate  solver
    solver = Solver(model, {'train': train_dataset, 'val': val_dataset}, decoder,
                    lr=lr,
                    reg=reg,
                    epochs=epochs,
                    criterion_type=criterion_type,
                    update_rule=update_rule,
                    id=id,
                    sufix=sufix,
                    log_dir=log_dir,
                    tensorboard=tensorboard,
                    device=device)

    solver.train()

    val_accuracy = solver.best_acc_val

    stats = {'loss_history': solver.loss_epochs,
             'val_accuracy': val_accuracy,
             'train_acc_history': solver.accuracy_train_epochs,
             'val_acc_history': solver.accuracy_val_epochs
            }

    if solver.best_acc_val > best_val:
        best_val = solver.best_acc_val
        best_stats = stats
        best_model = model
        best_solver = solver

    # Print results
    print('lr %e reg %e  val accuracy: %f' % (lr, reg, val_accuracy))


print('best validation accuracy achieved: %f' % best_val)
