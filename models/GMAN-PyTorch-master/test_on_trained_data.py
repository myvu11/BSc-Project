import argparse
import time
from model.test import test

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=1,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='./data/pems-bay.h5',
                    help='traffic file')
parser.add_argument('--SE_file', default='./data/SE(PeMS).txt',
                    help='spatial embedding file')
parser.add_argument('--model_file', default='./data/GMAN-pems.pkl',
                    help='save the model to disk')
parser.add_argument('--log_file', default='./data/log-pems',
                    help='log file')
args = parser.parse_args()



log = open(args.log_file, 'w')
# trainPred, valPred, testPred = test(args, log)
testPred = test(args, log)

