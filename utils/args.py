#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')

    arg_parser.add_argument('--scheduler', default='step', choices=['step', 'cosine'], help='type of scheduler')
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    arg_parser.add_argument('--step_size', type=int, default=50, help='step size for step scheduler')
    arg_parser.add_argument('--milestones', type=list, default=[50, 100], help='milestones for step scheduler')
    arg_parser.add_argument('--gamma', type=float, default=0.5, help='gamma for step scheduler')
    arg_parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps for scheduler')
    arg_parser.add_argument('--max_lr', type=float, default=0.01, help='max learning rate for cosine scheduler')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    # arg_parser.add_argument('--vocab_embed_size', default=300, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of embeded ...')
    arg_parser.add_argument('--hidden_size', default=768, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')
    arg_parser.add_argument('--model_name', default="hfl/chinese-lert-base", help='name of pretrained model')
    arg_parser.add_argument('--info', default="", help='info of this run')
    arg_parser.add_argument('--use_asr', default=True, help='use asr or manual script')

    return arg_parser