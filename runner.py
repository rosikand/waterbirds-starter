"""
File: runner.py
------------------
Runner script to train the model. This is the script that calls the other modules.
Execute this one to execute the program! 
Command line arg is which experiment to run as defined in experiments.py. 
"""


import argparse
import warnings
import pdb 
import experiments
import configs


def main(args):
    # load config 
    if args.config is None:
        config_class = 'BaseConfig'
    else:
        config_class = args.config

    cfg = getattr(configs, config_class)

    # load experiment 
    if args.experiment is None:
        exp_class = 'BaseExp'
    else:
        exp_class = args.experiment
    exp_class = getattr(experiments, exp_class)
    exp = exp_class(cfg)

	# train the model
    if args.train:
        exp.train(args.num_epochs, gradient_accumulate_every_n_batches=args.grad_accum, display_batch_loss=True)
    
    # test the model
    if args.test:
        exp.test()



if __name__ == '__main__':
    # example run 
    # python runner.py -e BaseExp -c BaseConfig -train -test -num_epochs 10 -grad_accum 1
    # configure args 
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-experiment", type=str, help='specify experiment.py class to use.') 
    parser.add_argument("-config", type=str, help='specify config.py class to use.') 
    parser.add_argument('-train', action='store_true', help='Do you want to train the model?')
    parser.add_argument('-test', action='store_true', help='Do you want to test the model?')
    parser.add_argument('-num_epochs', type=int, help='Number of epochs to train for.')
    parser.add_argument('-grad_accum', type=int, help='Number of gradient accumulations per batch.')
    args = parser.parse_args()
    main(args)
	