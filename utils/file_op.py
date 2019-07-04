# encoding: utf-8
import errno
import json
import os
import os.path as osp
import shutil


def create_exp_dir(path, scripts_to_save=None):
    if not osp.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(osp.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = osp.join(path, 'scripts', osp.basename(script))
            shutil.copyfile(script, dst_file)


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile
