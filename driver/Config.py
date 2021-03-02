from configparser import ConfigParser
import configparser
import sys, os

sys.path.append('..')


class Configurable(object):
    def __init__(self, config_file, extra_args, isTrain=True):
        config = ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if isTrain:
            config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    # ------------data config reader--------------------
    @property
    def patch_x(self):
        return self._config.getint('Data', 'patch_x')

    @property
    def patch_y(self):
        return self._config.getint('Data', 'patch_y')

    @property
    def patch_z(self):
        return self._config.getint('Data', 'patch_z')

    @property
    def patch_each(self):
        return self._config.getint('Data', 'patch_each')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def split_pickle(self):
        return self._config.get('Data', 'split_pickle')

    @property
    def data_name(self):
        return self._config.get('Data', 'data_name')

    @property
    def data_root(self):
        return self._config.get('Data', 'data_root')

    @property
    def unsu_root(self):
        return self._config.get('Data', 'unsu_root')

    @property
    def covid_data_root(self):
        return self._config.get('Data', 'covid_data_root')

    @property
    def sc_model_path(self):
        return self._config.get('Data', 'sc_model_path')
    # ------------save path config reader--------------------

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def tmp_dir(self):
        return self._config.get('Save', 'tmp_dir')

    @property
    def tensorboard_dir(self):
        return self._config.get('Save', 'tensorboard_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def log_file(self):
        return self._config.get('Save', 'log_file')

    @property
    def submission_dir(self):
        return self._config.get('Save', 'submission_dir')

    @property
    def model(self):
        return self._config.get('Network', 'model')

    @property
    def backbone(self):
        return self._config.get('Network', 'backbone')

    @property
    def classes(self):
        return self._config.getint('Network', 'classes')

    @property
    def channel(self):
        return self._config.getint('Network', 'channel')

    @property
    def epochs(self):
        return self._config.getint('Run', 'N_epochs')

    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def gpu(self):
        return self._config.getint('Run', 'gpu')

    @property
    def printfreq(self):
        return self._config.getint('Run', 'printfreq')

    @property
    def gpu_count(self):
        gpus = self._config.get('Run', 'gpu_count')
        gpus = gpus.split(',')
        return [int(x) for x in gpus]

    @property
    def run_num(self):
        return self._config.getint('Run', 'run_num')

    @property
    def workers(self):
        return self._config.getint('Run', 'workers')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')
    @property
    def bad_step(self):
        return self._config.getint('Run', 'bad_step')

    # ------------Optimizer path config reader--------------------
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def learning_rate_d(self):
        return self._config.getfloat('Optimizer', 'learning_rate_d')

    @property
    def max_patience(self):
        return self._config.getint('Optimizer', 'max_patience')
    @property
    def infer_epoch(self):
        return self._config.getint('Optimizer', 'infer_epoch')
    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizer', 'min_lrate')

    @property
    def beta_1(self):
        return self._config.getfloat('Optimizer', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Optimizer', 'beta_2')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')
