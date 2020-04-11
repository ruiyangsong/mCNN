#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.system('./retrieve_pdb_chain.py')
os.system('./map_pos.py')
os.system('./TMalign.py')
os.system('./CalSA.py')
os.system('./coord.py')