import os
import numpy as np

from utils import tokens, index2letter, num_tokens
from pathlib import Path


def write_predict(epoch, index, pred, flag, args):  # [batch_size, vocab_size] * max_output_len
    folder_name = Path(args.output_dir)  # 'pred_logs_' + str(args.run_id)
    folder_name.mkdir(exist_ok=True, parents=True)
    file_prefix = folder_name / (flag + '_predict_')

    if not flag == 'test':
        pred = pred.data
        pred2 = pred.topk(1)[1].squeeze(2)  # (15, 32)
        pred2 = pred2.transpose(0, 1)  # (32, 15)
        pred2 = pred2.cpu().numpy()
    else:
        pred2 = [pred]

    if not isinstance(index, np.ndarray):
        index = [index]

    with open(str(file_prefix) + (str(epoch)+'.log'), 'a') as f:
        for n, seq in zip(index, pred2):
            f.write(n+' ')
            for idx, letter in enumerate(seq):
                if letter == tokens['END_TOKEN']:
                    # f.write('<END>')
                    break
                else:
                    if letter == tokens['GO_TOKEN']:
                        f.write('<GO>')
                    elif letter == tokens['PAD_TOKEN']:
                        f.write('<PAD>')
                    else:
                        try:
                            letter_str = index2letter[letter-num_tokens]
                        except KeyError:
                            letter_str = '<unk>'
                        f.write(letter_str)
            f.write('\n')


def write_loss(loss_value, flag, args, rank=0):
    folder_name = Path(args.output_dir)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if flag == 'train':
        file_name = folder_name / f'loss_train_{rank}.log'
    elif flag == 'valid':
        file_name = folder_name / f'loss_valid_{rank}.log'
    elif flag == 'test':
        file_name = folder_name / f'loss_test_{rank}.log'
    else:
        file_name = folder_name / 'loss.log'
    with open(file_name, 'a') as f:
        f.write(str(loss_value))
        f.write(' ')
