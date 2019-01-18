#

import logging.handlers
import os

LOG_PATH = './log'
MAX_LOG_SIZE = 2560000
LOG_BACKUP_NUM = 4000

def check_log_dir(dir_name):
    #dir = os.path.dirname(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        pass

check_log_dir(LOG_PATH)
logger = logging.getLogger('disentangle')
log_file = os.path.join(LOG_PATH, 'disentangle.log')
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
formatter = logging.Formatter('%(asctime)s %(process)d %(processName)s %(threadName)s %(filename)s %(lineno)d %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.DEBUG)