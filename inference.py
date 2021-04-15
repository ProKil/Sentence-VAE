import os
import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate


def main(args):
    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    # samples, z = model.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # z_ = torch.randn([args.latent_size]).numpy()
    # input_sent = "the n stock specialist firms on the big board floor the buyers and sellers of last resort who were criticized after the n crash once again could n't handle the selling pressure"
    input_sent = "looking for a job was one of the most anxious periods of my life and is for most people"
    batch_input = torch.LongTensor(
        [[w2i[i] for i in input_sent.split()]]).cuda()
    batch_len = torch.LongTensor([len(input_sent.split())]).cuda()
    input_mean = model(batch_input, batch_len, output_mean=True)
    z_ = input_mean.cpu().detach().numpy()
    print(z_.shape)
    # z2 = torch.randn([args.latent_size]).numpy()
    for i in range(args.latent_size):
        print(f"-------Dimension {i}------")
        z1, z2 = z_.copy(), z_.copy()
        z1[i] -= 0.5
        z2[i] += 0.5
        z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=5)).float())
        samples, _ = model.inference(z=z)
        print('-------INTERPOLATION-------')
        print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
