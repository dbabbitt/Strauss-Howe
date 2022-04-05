
#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from storage import Storage

data_folder = r'../'*int(sys.argv[1]) + 'data/'
print('data_folder = ' + data_folder)

saves_folder = r'../'*int(sys.argv[1]) + 'saves/'
print('saves_folder = ' + saves_folder)

print(dir(Storage))