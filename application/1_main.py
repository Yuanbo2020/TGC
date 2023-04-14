import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from framework.data_generator import *
from framework.processing import *
from framework.context_model import *


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-times', type=int, required=True)
    parser.add_argument('-epochs', type=int, required=True)
    parser.add_argument('-check_step', type=int, required=True)
    parser.add_argument('-batch_size', type=int, required=True)
    parser.add_argument('-window', type=int, required=True)
    parser.add_argument('-encoder_layer', type=int, required=True)
    parser.add_argument('-adamw', action="store_true")
    parser.add_argument('-lr_init', type=float, required=True)
    args = parser.parse_args()

    times = args.times
    lr_init = args.lr_init
    epochs = args.epochs
    window = args.window
    batch_size = args.batch_size
    check_step = args.check_step
    adamw = args.adamw
    encoder_layers = args.encoder_layer

    generator = DataGenerator_data(batch_size=batch_size, normalization=True, window=window)

    basic_name = 'sys_' + str(times) + '_l' + str(encoder_layers) \
                 + '_w' + str(window) + '_' + str(lr_init).replace('-', '')
    if adamw:
        basic_name = basic_name + '_adamw'

    suffix, system_name = define_system_name(basic_name=basic_name, batch_size=batch_size, epochs=epochs)
    system_path = os.path.join(os.getcwd(), system_name)

    models_dir = system_path

    log_path = models_dir + '_log'

    if not os.path.exists(log_path):
        create_folder(log_path)

        filename = os.path.basename(__file__).split('.py')[0]
        print_log_file = os.path.join(log_path, filename + '_print.log')
        sys.stdout = Logger(print_log_file, sys.stdout)
        console_log_file = os.path.join(log_path, filename + '_console.log')
        sys.stderr = Logger(console_log_file, sys.stderr)

        model = Transformer_context(input_dim=64,
                                    window=window,
                                    ntoken=len(config.labels),
                                    encoder_layers=encoder_layers,
                                    d_model=512)
        print(model)

        if config.cuda:
            model.cuda()

        training = training_aec_only_testing_validation
        training(generator, model, config.cuda, models_dir, epochs, adamw,
                 batch_size=batch_size, check_step=check_step,
                 lr_init=lr_init, log_path=log_path)

        print('Training is done!!!')






if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















