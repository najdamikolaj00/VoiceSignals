'''

Function for converting .m4a extension files to .wav extension from two directories

'''
from pydub import AudioSegment
import os

def converter():
    rootdir = 'DataVoices'

    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            if filename.endswith('.wav'):
                continue
            else:
                AudioSegment.from_file(os.path.join(subdir, filename), os.path.splitext(filename)[1][1:]).export(os.path.join(subdir
                , filename.split(".")[0] + '.wav'), format = 'wav')
converter()
