import os
import sys
import re

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

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect InjeAIs.')
    parser.add_argument("fromName",
                        metavar="<FromName>",
                        help="From name")
    parser.add_argument('toName',
                        metavar="<ToName>",
                        help='To name')
    parser.add_argument('-e', '--ext',
                        default=False,
                        action="store_true",
                        help="Enable replacing extensions also.")
    parser.add_argument('-r', '--rec',
                        default=False,
                        action="store_true",
                        help="Enable replacing extensions also.")
    args = parser.parse_args()

    assert args.fromName is not None and args.toName, "Requirements are not satisfied!"

    regex = re.compile(args.fromName, re.IGNORECASE)
    changed_cnt = 0
    for f in files_in_dir('./', args.rec):
        filename = os.path.basename(f)
        ext = ''
        dir = os.path.dirname(f)
        if not args.ext:
            filename = os.path.splitext(filename)
            ext = filename[1]
            filename = filename[0]
        rename = regex.sub(args.toName, filename)
        rename = "{}/{}".format(dir, rename)
        if len(ext) > 0:
            rename = rename + ext
        if regex.search(filename) == None:
            continue

        print('"{}" => "{}"'.format(f, rename))
        os.rename(f, rename)
        changed_cnt = changed_cnt + 1
    
    print("{} file(s) changed.".format(changed_cnt))
    