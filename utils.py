import pandas as pd
import torch
import torch.distributed as dist

from model import DeepSpeech


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device, model_path, use_half):
    model = DeepSpeech.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model


def get_subject_id_from_filepath(filepath):

    '''
        Obs: For now, specific for ParkinsonSpanishSpeech dataset.

    '''

    temp = filepath.split('AVPEPUDEA', 1)[1]
    if temp[0] == 'C':
        id_subj = 'AVPEPUDEAC' + temp[1:5]
    else:
        id_subj = 'AVPEPUDEA' + temp[0:4]
    return id_subj


def parse_dataset(metadata_filepath, recpath_filepath):
    '''
        Parameters
        ----------
        metadata_filepath: path to metadata csv file containing information about every subject.

        recpath_filepath: path to txt or csv file where every line is the path to a single recording.

        Returns
        -------
        List of tuples (recording path, category)
    '''

    metadata_df = pd.read_excel(metadata_filepath)
    metadata_df.set_index(metadata_df.columns.tolist()[0], inplace=True)

    with open(recpath_filepath) as f:
        rec_path_list = f.readlines()

    ids = []
    for filepath in rec_path_list:

        # Remove potential \n
        filepath = filepath.split('\n')[0]

        id_subj = get_subject_id_from_filepath(filepath)
        label = 1 if int(metadata_df['H/Y'][id_subj]) > 0 else 0  # I think this will be codified later we do not have to worry now

        ids.append((filepath, label))

    return ids


