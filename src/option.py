import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=16,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/home/ljm/datasets/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test/new1080p',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,  #5*1e-4
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',  #8
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,  #0.1
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',  #'0.5*L1+0.5*MSE'   '1*L1'  '0.5*L1+0.5*L12'
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

# if add Adafm 
parser.add_argument('--adafm', action='store_true',
                    help='edsr + adafm')
parser.add_argument('--mlp', action='store_true',
                    help='edsr + adafm')
parser.add_argument('--adafm_espcn', action='store_true',
                    help='espcn + adafm')
parser.add_argument('--edsr_espcn', action='store_true',
                    help='edsr + espcn, side tuning')
parser.add_argument('--edsr_res', action='store_true',
                    help='edsr only fine tune part resblock')

parser.add_argument('--segnum', type=int, default=1,
                    help='segnumber')

parser.add_argument('--sidetuning', action='store_true',
                    help='sidetuning')

parser.add_argument('--adafm_side', action='store_true',
                        help='espcn + sidetuning')

parser.add_argument('--data_partion', type=float, default=0.05,
                    help='data_partion')

parser.add_argument('--tcloss_v1', action='store_true',
                        help='tcloss_v1')

parser.add_argument('--tcloss_v2', action='store_true',
                        help='tcloss_v2')

parser.add_argument('--tcloss_seg', type=int, default=0,
                        help='which seg is selected to finetuning')

parser.add_argument('--dvp', action='store_true',
                        help='dvp: use SR as the GT')

parser.add_argument('--use_adafm', action='store_true',
                        help='using adafm block')

parser.add_argument('--is45s', action='store_true',
                        help='is 45s video')

parser.add_argument('--is15s', action='store_true',
                        help='is 15s video')

parser.add_argument('--is30s', action='store_true',
                        help='is 15s video')

parser.add_argument('--finetune', action='store_true',
                        help='is 15s video')

parser.add_argument('--is5min', action='store_true',
                        help='is 15s video')

parser.add_argument('--isCactus', action='store_true',
                        help='is 15s video')

parser.add_argument('--ismamlseg10', action='store_true',
                        help='is 15s video')

parser.add_argument('--ismaml5seg300', action='store_true',
                        help='is 15s video')

parser.add_argument('--ismamlmix_ytbx6_vix1_hevcx3', action='store_true',
                        help='is 15s video')

parser.add_argument('--ismamlmix_45sx4-lolx2', action='store_true',
                        help='is 15s video')

parser.add_argument('--ismaml15s450f', action='store_true',
                        help='is 15s video')

parser.add_argument('--maml', action='store_true',
                        help='using maml method')

parser.add_argument('--use_maml', action='store_true',
                        help='using maml method')

parser.add_argument('--lr_son', type=float, default=2,
                    help='data_partion')

parser.add_argument('--new_espcn', action='store_true',
                        help='using maml method')

parser.add_argument('--grad_mask', action='store_true',
                        help='using grad mask')

parser.add_argument('--inner_lr', type=float, default=1e-4,  #5*1e-4
                    help='learning rate')

parser.add_argument('--inner_lr_decay', type=float, default=1e-4,  #5*1e-4
                    help='learning rate')

parser.add_argument('--espcn_resblock', action='store_true',
                        help='using grad mask')

parser.add_argument('--espcn_resblock_skip', action='store_true',
                        help='using grad mask')

parser.add_argument('--ismaml45s3seg', action='store_true',
                        help='using grad mask')
parser.add_argument('--ismaml45s2seg', action='store_true',
                        help='using grad mask')
parser.add_argument('--is2min4seg', action='store_true',
                        help='using grad mask')
parser.add_argument('--is5min4seg', action='store_true',
                        help='using grad mask')
parser.add_argument('--ismaml45s6seg', action='store_true',
                        help='using grad mask')
parser.add_argument('--ismix15s6', action='store_true',
                        help='using grad mask')
parser.add_argument('--ismixhevc', action='store_true',
                        help='using grad mask')
parser.add_argument('--isKongfu', action='store_true',
                        help='using grad mask')
parser.add_argument('--ohem', action='store_true',
                        help='using grad mask')

parser.add_argument('--find_mask', action='store_true',
                        help='find grad mask')

parser.add_argument('--ismixflew', action='store_true',
                        help='find grad mask')
parser.add_argument('--isflew17', action='store_true',
                        help='find grad mask')
parser.add_argument('--iscvprtype1', action='store_true',
                        help='find grad mask')
parser.add_argument('--istype1few50', action='store_true',
                        help='find grad mask')
parser.add_argument('--istype1few10', action='store_true',
                        help='find grad mask')
parser.add_argument('--istype2few10', action='store_true',
                        help='find grad mask')
parser.add_argument('--istype4few50', action='store_true',
                        help='find grad mask')
parser.add_argument('--istype4few150', action='store_true',
                        help='find grad mask')

parser.add_argument('--save_every', action='store_true',
                        help='find grad mask')

parser.add_argument('--patchnet', action='store_true',
                        help='find grad mask')
parser.add_argument('--patchnet1', action='store_true',
                        help='find grad mask')
parser.add_argument('--patchnet_test', action='store_true',
                        help='find grad mask')
parser.add_argument('--p_mask', type=float, default=0.20,  
                    help='the percentage of grademask using')
parser.add_argument('--dir_mask', type=str, default='../grad_mask/mask_1106.pt',
                    help='dataset directory')
args = parser.parse_args()
#print(args.template.find('--scale'))
template.set_template(args)
#print('9999')


args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

if args.model == 'SRVC':
    #print('111111')
    args.model = 'SRVC'
    args.f = 64
    args.F = 64
    args.n_feats = 32
    args.patch_h = 7
    args.patch_w = 7

    args.lr = 1e-3
    args.loss = '1*L1'
    args.patch_size = 140
    #args.scale = 2
    args.batch_size = 16
    #args.epochs = 2

    args.device = "0"
    args.n_GPUs = 1
    #args.dir_data = dir_data
    args.ext = "sep"
    args.print_every = 100