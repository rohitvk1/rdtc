import os
import sys
import shutil
from argparse import ArgumentParser

import torch

from trainer import Trainer
from models.rdtc import RDTC
from utils.data_loader import DataLoader


def parse_args():
    parser = ArgumentParser(description='Recurrent Decision Tree through Communication')
    parser.add_argument('--name', '-n', default='rdtc', type=str,
                        help='Name of the experiment (default: %(default)s)')
    parser.add_argument('--dataset', '-ds', default='awa2', type=str,
                        help='Dataset for experiment (default: %(default)s)',
                        choices=['awa2', 'cub'])
    parser.add_argument('--device-id', '-d', default=0, type=int,
                        help='CUDA device ID (ignored when CUDA is not available) (default: %(default)s)')
    parser.add_argument('--data-path', '-dp', default='./data', type=str,
                        help='Path to datasets (default: %(default)s)')
    parser.add_argument('--tau', '-t', default=5., type=float,
                        help='Initial tau parameter for gumbel softmax (default: %(default)s)')
    parser.add_argument('--use-pretrained', '-up', action='store_true',
                        help='Use a pretrained cnn backbone (default: %(default)s)')
    parser.add_argument('--max-iters', '-mi', type=int, default=20,
                        help='Maximum number of communication iterations T (default: %(default)s)')
    parser.add_argument('--optimizer', '-o', default='adam', type=str,
                        help='Optimizer (default: %(default)s)', choices=['adam', 'sgd'])
    parser.add_argument('--step-size', '-s', default=25, type=int,
                        help='Step size for reducing the lerning rate a factor of 1/10 in epochs (default: %(default)s)')
    parser.add_argument('--attribute-size', '-as', default=256, type=int,
                        help='Number of learned attributes when attribute coefficient is 0. (default: %(default)s)')
    parser.add_argument('--attribute-coef', '-ac', default=0.2, type=float,
                        help='Coefficient for attribute loss, i.e., lambda (default: %(default)s)')
    parser.add_argument('--num-epochs', '-e', default=100, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--batch-size', '-b', default=128, type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--learning-rate', '-lr', default=0.001, type=float,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--weight-decay', '-wd', default=0., type=float,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--num_workers', '-nw', default=8, type=int,
                        help='Number of data loader workers (default: %(default)s)')
    parser.add_argument('--decision-size', '-dec', default=2, type=int,
                        help='Tree width (default: %(default)s)')
    parser.add_argument('--hidden-size', '-hs', default=1024, type=int,
                        help='Hidden layer size (default: %(default)s)')
    parser.add_argument('--log-dir', type=str, default='./log',
                        help=('path where all outputs are stored '
                              '(default: %(default)s)'))
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Force overwrite log directory if it already '
                             'exists (default: %(default)s)')
    parser.add_argument('--eval', type=str, default=None,
                        help='Evaluate model checkpoint')
    parser.add_argument('--threshold', type=float, default=1.,
                        help='Threshold for pruning at test/val time (default: %(default)s)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # Namespace(name='ardtc_cub', dataset='cub', device_id=0, data_path='./data', 
    # tau=5.0, use_pretrained=True, max_iters=25, optimizer='adam', step_size=25, 
    # attribute_size=256, attribute_coef=0.2, num_epochs=100, batch_size=128, learning_rate=0.001, 
    # weight_decay=0.0, num_workers=8, decision_size=2, hidden_size=1024, log_dir='./log', 
    # overwrite=False, eval=None, threshold=1.0)

    device = torch.device('cuda:{}'.format(args.device_id)
                          if torch.cuda.is_available() else 'cpu')   # cuda:0
   

    # Initialize dataset and loader
    dl = DataLoader(args.dataset)
    
    dataloaders, n_classes = dl.load_data(args.batch_size, args.num_workers,
                                          root=args.data_path)      
    #dict of dataloader objects(train,val,test), n_classes=200

    # Initialize model
    attribute_mtx = None
    attribute_size = args.attribute_size    
    # attribute_size: 256

    if args.attribute_coef > 0.:
        # Use attribute data
        attribute_mtx = dataloaders['train'].dataset.dataset.attribute_mtx  
        # attribute_mtx: torch.Size([200, 312])

        # Binarize attribute data
        attribute_mtx[attribute_mtx < 0.5] = 0.
        attribute_mtx[attribute_mtx >= 0.5] = 1.
        attribute_mtx = attribute_mtx.to(device)
        attribute_size = attribute_mtx.size(1)   
        # attribute_size: 312

    model = RDTC(n_classes, args.dataset, args.decision_size,
                 args.max_iters, attribute_size, attribute_mtx,
                 args.attribute_coef, args.hidden_size, tau_initial=args.tau,
                 use_pretrained=args.use_pretrained, threshold=args.threshold)

    model.to(device)

    # Initialize optimizer and scheduler
    params = filter(lambda p: p.requires_grad, model.parameters())

    # params

    # [torch.Size([1024]), torch.Size([1024]), torch.Size([4096, 1024]), torch.Size([4096, 1024]), 
    # torch.Size([4096]), torch.Size([4096]), torch.Size([1024, 624]), torch.Size([1024]), 
    # torch.Size([1024]), torch.Size([1024]), torch.Size([200, 1024]), torch.Size([200]), 
    # torch.Size([1]), torch.Size([1024]), torch.Size([1024]), torch.Size([1024, 1024]), 
    # torch.Size([1024]), torch.Size([1024]), torch.Size([1024]), torch.Size([312, 1024]), 
    # torch.Size([312]), torch.Size([1]), torch.Size([2048]), torch.Size([2048]), 
    # torch.Size([1024, 2048]), torch.Size([1024]), torch.Size([1024]), torch.Size([1024]), 
    # torch.Size([1024, 1024]), torch.Size([1024]), torch.Size([1024]), torch.Size([1024]), 
    # torch.Size([624, 1024]), torch.Size([624]), torch.Size([1024, 1248]), torch.Size([1024]), 
    # torch.Size([1024]), torch.Size([1024])]


    if args.optimizer == 'adam':    # args.optimizer=adam, args.learning_rate=0.001, args.weight_decay=0.0
        optimizer = torch.optim.Adam(params, lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, # args.step_size=25
                                                gamma=0.1)

    if args.eval:
        model_state_dict = torch.load(args.eval)
        model.load_state_dict(model_state_dict)
        log_dir = os.path.join('log', 'test')
    else:
        # Initialize trainer and run the model
        log_dir = os.path.join(args.log_dir, args.name)     # ./log/ardtc_cub

        if os.path.exists(log_dir):
            args.overwrite = True
            if args.overwrite:
                print('Overwrite specified, deleting existing log directory: '
                      '{}'.format(log_dir))
                shutil.rmtree(log_dir)
            else:
                print("Log directory already exists. Overwrite existing data "
                      "by passing the '--overwrite' flag. Exiting.")
                sys.exit()
        os.makedirs(log_dir)

    trainer = Trainer(model, dataloaders, optimizer, scheduler,
                      args.num_epochs, device, log_dir)
    if args.eval:
        trainer.test()
    else:
        trainer.train()
