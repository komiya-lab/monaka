# -*- coding: utf-8 -*-

import logging
import os


def get_logger(name):
    return logging.getLogger(name)


def init_logger(logger: logging.Logger,
                path=None,
                mode='w',
                level=None,
                handlers=None,
                verbose=True):
    level = level or logging.WARNING
    if not handlers:
        handlers = [logging.StreamHandler()]
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=level,
                            handlers=handlers)
        if path:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            handlers.append(logging.FileHandler(path, mode))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


logger = get_logger("monaka")