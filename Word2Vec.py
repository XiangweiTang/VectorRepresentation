from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector


current_path=os.path.dirname(os.path.realpath(sys.argv[0]))

parser=argparse.ArgumentParser()
parser.add_argument(
	'--log_dir',
	type=str,
	default=os.path.join(current_path,'log'),
	help='The log directory for Tensorboard summaries.'
)

FLAGS, unparsed=parser.parse_known_args()

if not  os.path.exists(FLAGS.log_dir):
	os.makedirs(FLAGS.log_dir)

url='http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
	local_filename=os.path.join("SampleData",filename)
	if not os.path.exists(local_filename):
		local_filename, _=urllib.request.urlretrieve(url+filename,local_filename)
	
	statinfo=os.stat(local_filename)
	if statinfo.st_size==expected_bytes:
		print("Found and verified", filename)
	else:
		print(statinfo.st_size)
		raise Exception("Failed to verify "+local_filename+'. Can you get to it with a browser?')
	
	return local_filename

filename=maybe_download('text8.zip', 31344016)

def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		data=tf.compat.as_str(f.read(f.namelist()[0])).split()
	return data

vocabulary=read_data(filename)

print('Data size', len(vocabulary))