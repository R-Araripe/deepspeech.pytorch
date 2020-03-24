import argparse
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder, MyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='../Data/downsampled-16k/manifest_vowels.txt')
# parser.add_argument('--metadata', metavar='DIR',
#                     help='path to metadata csv', default='data/metadata.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser = add_decoder_args(parser)


def get_number_of_correct(y_true, y_pred):
    return np.sum([true==pred for true, pred in zip(y_true, y_pred)])


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False):
    model.eval()
    output_data = []
    y_true = np.array([])
    y_pred = np.array([])
    total_correct = 0
    # import pdb; pdb.set_trace()
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half: # WHAT IS HALF????
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        # print('inputs: ', inputs)
        # print('input sizes: '. list(inputs.size()))

        out, output_sizes = model(inputs, input_sizes)   # evaluation happens here I think
        out = out.transpose(0, 1)  # TxNxH
        # print('out: ', out)
        # print('output_sizs: ', output_sizes)
        # print('split_targets: ', split_targets)

        # import pdb; pdb.set_trace()

        decoded_output = decoder.decode(out, output_sizes)
        target_labels = target_decoder.convert_to_labels(split_targets)

        y_true = np.concatenate((y_true, target_labels))
        y_pred = np.concatenate((y_pred, decoded_output))

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_labels))


        # print('target_strings: ', target_labels)
        # print('len de target_strings: ', len(target_labels))
        # print('decoded output: ', decoded_output)
        # decoded_output = np.array(decoded_output)
        # print('Sequence for every rec in batch: ')
        # for i in range(len(decoded_output)):
        #     print('%i: '%(i), decoded_output[i])
        #     print('\n')

        print('val pred labels: ', decoded_output)
        print('val targ labels: ', target_labels)

        total_correct += get_number_of_correct(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred) * 100.0
    cm = confusion_matrix(target_labels, decoded_output, labels=[0, 1])

    print('confusion matrix:')
    print(pd.DataFrame(cm))

    # accuracy_list = np.array(accuracy_list)
    # print('accuracy_list: ', accuracy_list)
    # accuracy_mean = accuracy_list.mean()  # nao eh assim que faz, eh pra somar todos os certos e fazer uma razao no final https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    # accuracy_std = accuracy_list.std()

    return accuracy, output_data

if __name__ == '__main__':

    metadata_path  = '../Data/raw/PCGITA_metadata.xlsx'
    model_path = '../Data/Models/librispeech_pretrained_v2.pth'

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, model_path, args.half)

    # if args.decoder == "beam":
    #     from decoder import BeamCTCDecoder

    #     decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
    #                              cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
    #                              beam_width=args.beam_width, num_processes=args.lm_workers)
    # elif args.decoder == "greedy":
    #     decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    # else:
        # decoder = None
    # target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    decoder = MyDecoder(model.labels)
    target_decoder = MyDecoder(model.labels)

    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest, metadata_file_path=metadata_path,
                                      labels=model.labels, normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True) # in train, the manifest will be already stratified with only train data so it's ok.
    accuracy_mean, acccuracy_std, output_data = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=target_decoder,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half)

    print('Test Summary \t'
          'Average accuracy {acc_mean:.3f}\t'
          'Standard deviation accuracy {acc_std:.3f}\t'.format(acc_mean=accuracy_mean, acc_std=acccuracy_std))

    if args.save_output is not None:
        np.save(args.save_output, output_data)
