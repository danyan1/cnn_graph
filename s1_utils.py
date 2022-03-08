# DIRECTORY MANIPULATION
import os


def list_files(dir_path, keyword):
    fileNames = []
    for f in os.listdir(dir_path):
        if keyword in f:
            # os.path.isfile(dir_path + f) or os.path.isdir(dir_path + f):
            fileNames += [f, ]
    return sorted(fileNames)