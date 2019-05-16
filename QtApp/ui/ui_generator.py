import os
from os.path import join
import sys
import shutil

def files_in_dir(path, recursive=True):
    files = []
    for f in os.listdir(path):
        f = path + f
        if os.path.isdir(os.path.abspath(f)):
            if not recursive:
                continue
            subFiles = files_in_dir(f + '/')
            files.extend(subFiles)
        else:
            files.append(f)
    return files

if __name__ == "__main__":
    ROOT_DIR = "./"
    rel = os.path.relpath(os.path.dirname(__file__))
    ROOT_DIR = join(ROOT_DIR, rel) + "/"
    OUT_DIR = os.path.abspath(ROOT_DIR + "../QtUI") + "/"
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.mkdir(OUT_DIR)

    UIs = files_in_dir(ROOT_DIR)
    for ui in UIs:
        ui = ui.replace(ROOT_DIR, "")
        fileNameNExt = os.path.splitext(ui)
        filename = fileNameNExt[0]
        ext = fileNameNExt[1]
        if ext != '.ui':
            continue
        ui_path = join(ROOT_DIR, ui)
        ui_out = join(OUT_DIR, filename + ".py")
        dir_check = os.path.dirname(ui_out)
        if not os.path.exists(dir_check):
            os.makedirs(dir_check)

        print("Generating with target '{}'...".format(ui))
        os.system("pyuic5 {} -o {}".format(ui_path, ui_out))
        print("Generated: {}".format(ui_out))