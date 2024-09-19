import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    """-------------------"""
    """Hybrid model parser"""
    hybrid_model_parser = argparse.ArgumentParser(description='hybrid model general')

    # basic config
    hybrid_model_parser.add_argument('--is_training', type=int, default=1, help='status')

    # data loader
    hybrid_model_parser.add_argument('--data', type=str, default='bankrupt_companies_with_17_variables_5_years', help='dataset type')
    hybrid_model_parser.add_argument('--root_path', type=str, default='./data/bankrupt_companies_with_17_variables_5_years/', help='root path of the data file')
    hybrid_model_parser.add_argument('--numerical_data_path', type=str, default='financial_data/bankrupt_companies_with_17variables_5years_split_version4_complete.csv', help='data csv file')
    hybrid_model_parser.add_argument('--raw_textual_data_path', type=str, default='textual_data/raw_corpora/textual_data_matched_split_version3.csv', help='data csv file')
    hybrid_model_parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # GPU
    # TODO: add code to run on gpu for gpu use case (Wojtek)
    hybrid_model_parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    hybrid_model_parser.add_argument('--gpu', type=int, default=0, help='gpu')
    hybrid_model_parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    hybrid_model_parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # optimization
    hybrid_model_parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    hybrid_model_parser.add_argument('--itr', type=int, default=1, help='experiments times')

    # learning
    hybrid_model_parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    hybrid_model_parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    hybrid_model_parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test and validation input data')
    hybrid_model_parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
    hybrid_model_parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    hybrid_model_parser.add_argument('--des', type=str, default='test', help='exp description')
    hybrid_model_parser.add_argument('--lradj', type=str, default='const', help='adjust learning rate')
    hybrid_model_parser.add_argument('-r', '--random_seeds', nargs='+', default=['2024'], help="List of random seeds")

    hybrid_model_args = hybrid_model_parser.parse_args()

    """-------------------"""
    """iTransformer parser"""
    iTransformer_parser = argparse.ArgumentParser(description='iTransformer')

    # forecasting task
    iTransformer_parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
    # iTransformer_parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    iTransformer_parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    iTransformer_parser.add_argument('--enc_in', type=int, default=17, help='encoder input size')
    iTransformer_parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    iTransformer_parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
    iTransformer_parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    iTransformer_parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    iTransformer_parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    iTransformer_parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    iTransformer_parser.add_argument('--factor', type=int, default=1, help='attn factor')
    iTransformer_parser.add_argument('--distil', action='store_false',
                                     help='whether to use distilling in encoder, using this argument means not using distilling',
                                     default=True)
    iTransformer_parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    iTransformer_parser.add_argument('--embed', type=str, default='timeF',
                                     help='time features encoding, options:[timeF, fixed, learned]')  # todo simplify
    iTransformer_parser.add_argument('--activation', type=str, default='gelu', help='activation')
    iTransformer_parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    iTransformer_parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # iTransformer
    iTransformer_parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    iTransformer_parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    iTransformer_parser.add_argument('--target_root_path', type=str, default='./data/bankrupt_companies_with_17_variables_5_years/', help='root path of the data file')
    iTransformer_parser.add_argument('--target_data_path', type=str, default='bankrupt_companies_with_17_variables_5_years_version2_split.csv', help='data file')
    iTransformer_parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

    iTransformer_args = iTransformer_parser.parse_args()

    """-------------------"""
    """Hierarchical BERT parser"""
    HBERT_parser = argparse.ArgumentParser(description='Hierarchical BERT')
    HBERT_parser.add_argument("-d", "--dataset_name", type=str, default='longer_moviereview', help="directory of raw text data")
    HBERT_parser.add_argument("-t", "--encoding_method", nargs="+", default=['hbm'], help="Encoding methods to use (hbm, roberta, fasttext)", choices=['hbm', 'roberta', 'fasttext'])
    HBERT_parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the transformer model")
    HBERT_parser.add_argument("--num_hidden_layers", type=int, default=4, help="Number of hidden layers")
    HBERT_parser.add_argument("--num_attention_heads", type=int, default=1, help="Number of attention heads")
    HBERT_parser.add_argument("--intermediate_size", type=int, default=3072, help="Intermediate size in the FFN")
    HBERT_parser.add_argument("--hidden_act", type=str, default="relu", help="Activation function (relu, gelu, etc.)")
    HBERT_parser.add_argument("--hidden_dropout_prob", type=float, default=0.01, help="Dropout probability for hidden layers")
    HBERT_parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.01, help="Dropout probability for attention")
    HBERT_parser.add_argument("--seq_length", type=int, default=128, help="Maximum sequence length")
    HBERT_parser.add_argument("--initializer_range", type=float, default=0.02, help="Initializer range for weights")
    HBERT_parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="Epsilon for layer normalization")
    HBERT_parser.add_argument("--output_attentions", action="store_true", help="Whether to output attention weights")
    HBERT_parser.add_argument("--output_hidden_states", action="store_true", help="Whether to output hidden states")
    HBERT_parser.add_argument("--num_labels", type=int, default=2, help="Number of classification labels")

    HBERT_args = HBERT_parser.parse_args()

    """-------------------"""

    hybrid_model_args.use_gpu = True if torch.cuda.is_available() and hybrid_model_args.use_gpu else False

    if hybrid_model_args.use_gpu and hybrid_model_args.use_multi_gpu:
        hybrid_model_args.devices = hybrid_model_args.devices.replace(' ', '')
        device_ids = hybrid_model_args.devices.split(',')
        hybrid_model_args.device_ids = [int(id_) for id_ in device_ids]
        hybrid_model_args.gpu = hybrid_model_args.device_ids[0]

    print('Args in experiment:')
    print(hybrid_model_args)
    print(iTransformer_args)
    print(HBERT_args)

    Exp = Exp_Long_Term_Forecast

    if hybrid_model_args.is_training:
        for ii in range(hybrid_model_args.itr):
            # setting record of experiments
            setting = '{}_ft{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                hybrid_model_args.data,
                iTransformer_args.seq_len,
                iTransformer_args.pred_len,
                iTransformer_args.d_model,
                iTransformer_args.n_heads,
                iTransformer_args.e_layers,
                iTransformer_args.d_ff,
                iTransformer_args.factor,
                iTransformer_args.embed,
                iTransformer_args.distil,
                hybrid_model_args.des,
                iTransformer_args.class_strategy, ii)

            exp = Exp(hybrid_model_args, iTransformer_args, HBERT_args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if iTransformer_args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            iTransformer_args.model_id,
            iTransformer_args.model,
            iTransformer_args.data,
            iTransformer_args.features,
            iTransformer_args.seq_len,
            iTransformer_args.label_len,
            iTransformer_args.pred_len,
            iTransformer_args.d_model,
            iTransformer_args.n_heads,
            iTransformer_args.e_layers,
            iTransformer_args.d_ff,
            iTransformer_args.factor,
            iTransformer_args.embed,
            iTransformer_args.distil,
            iTransformer_args.des,
            iTransformer_args.class_strategy, ii)

        exp = Exp(hybrid_model_args, iTransformer_args, HBERT_args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
