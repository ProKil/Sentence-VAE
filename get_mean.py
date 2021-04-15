import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE


def main(args):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    with open(os.path.join(args.save_model_path, 'model_params.json'), 'r') as f:
        params = json.load(f)
    model = SentenceVAE(**params)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    with torch.no_grad():

        input_sent = "the n stock specialist firms on the big board floor the buyers and sellers of last resort who were criticized after the n crash once again could n't handle the selling pressure"
        batch_input = torch.LongTensor(
            [[w2i[i] for i in input_sent.split()]]).cuda()
        batch_len = torch.LongTensor([len(input_sent.split())]).cuda()
        input_mean = model(batch_input, batch_len, output_mean=True)

        data_loader = DataLoader(
            dataset=datasets["train"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        print('---------CALCULATING NEAREST SENTENCES--------')

        sim = []
        all_sentences = []
        for iteration, batch in enumerate(data_loader):

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            all_sentences.append(batch['input'])

            # Forward pass
            mean = model(batch['input'], batch['length'], output_mean=True)
            batch_sim = torch.abs(mean - input_mean)
            sim.append(batch_sim)
        sim = torch.cat(sim, dim=0)
        _, most_similar_per_dim = torch.topk(-sim, k=20, dim=0)
        most_similar_per_dim = most_similar_per_dim.transpose(0, 1)
        all_sentences = torch.cat(all_sentences, dim=0)
        for dim, i in enumerate(most_similar_per_dim):
            sentences = torch.index_select(all_sentences, dim=0, index=i)
            print(f"{dim=}")
            print(*idx2word(sentences, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function',
                        type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')
    parser.add_argument('-c', '--load_checkpoint', type=str)

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
