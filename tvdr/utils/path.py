import shutil
import os


def create_folder(path):
    dir = path
    check = os.path.isdir(dir)

    if not check:
        os.makedirs(dir)


def delete_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))


def remove_file(path):
    os.remove(path)


def find_wav_files(path):
    listFile = os.listdir(path)
    listWAV = []
    for fileName in listFile:
        if fileName[-4:] == ".wav":
            listWAV.append(fileName)

    return listWAV


def rename_file(oldName, newName):
    os.rename(oldName, newName)


def copy_file(source_path, destination_path):
    shutil.copyfile(source_path, destination_path)


def get_dirname(path):
    return os.path.dirname(path)
