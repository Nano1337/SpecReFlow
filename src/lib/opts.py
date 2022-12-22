from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # demos 
        self.parse.add_argument('--demo_input', default='', help='path to Image / Directory / Video')
        self.parse.add_argument('--demo_output', default='', help='path to output of Image / Directory / Video')
        self.parse.add_argument('--load_model', default='', help='path to model')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1, help='input height and width. -1 for default from dataset. Will be overridden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1, help='input height. -1 for default from dataset')
        self.parser.add_argument('--input_w', type=int, default=-1, help='input width. -1 for default from dataset')
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else: 
            opt = self.parser.parse_args(args)

        input_h = opt.input_res if opt.input_res > 0 else opt.input_h
        input_w = opt.input_res if opt.input_res > 0 else opt.input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w

        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt
