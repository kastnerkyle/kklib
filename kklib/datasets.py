from __future__ import print_function
import numpy as np
import os
import xml
import xml.etree.cElementTree as ElementTree
import HTMLParser
from collections import Counter, OrderedDict
import tarfile
import pickle
import subprocess
import copy
import shutil

"""
- all points:
>> [[x1, y1, e1], ..., [xn, yn, en]]
- indexed values
>> [h1, ... hn]
"""

def distance(p1, p2, axis=None):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=axis))


def clear_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = distance(p1, p2) + distance(p2, p3)
        if dist > 1500:
            to_remove.add(i)
    npts = []
    for i in range(len(pts)):
        if i not in to_remove:
            npts += [pts[i]]
    return np.array(npts)


def separate(pts):
    seps = []
    for i in range(0, len(pts) - 1):
        if distance(pts[i], pts[i+1]) > 600:
            seps += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seps, seps + [len(pts)])]


def iamondb_extract(partial_path):
    """
    Lightly modified from https://github.com/Grzego/handwriting-generation/blob/master/preprocess.py
    """
    data = []
    charset = set()

    file_no = 0
    pth = os.path.join(partial_path, "original")
    for root, dirs, files in os.walk(pth):
        # sort the dirs to iterate the same way every time
        # https://stackoverflow.com/questions/18282370/os-walk-iterates-in-what-order
        dirs.sort()
        for file in files:
            file_name, extension = os.path.splitext(file)
            if extension == '.xml':
                file_no += 1
                print('[{:5d}] File {} -- '.format(file_no, os.path.join(root, file)), end='')
                xml = ElementTree.parse(os.path.join(root, file)).getroot()
                transcription = xml.findall('Transcription')
                if not transcription:
                    print('skipped')
                    continue
                #texts = [html.unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                texts = [HTMLParser.HTMLParser().unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                points = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]
                strokes = []
                mid_points = []
                for ps in points:
                    pts = np.array([[int(p.get('x')), int(p.get('y')), 0] for p in ps])
                    pts[-1, 2] = 1

                    pts = clear_middle(pts)
                    if len(pts) == 0:
                        continue

                    seps = separate(pts)
                    for pss in seps:
                        if len(seps) > 1 and len(pss) == 1:
                            continue
                        pss[-1, 2] = 1

                        xmax, ymax = max(pss, key=lambda x: x[0])[0], max(pss, key=lambda x: x[1])[1]
                        xmin, ymin = min(pss, key=lambda x: x[0])[0], min(pss, key=lambda x: x[1])[1]

                        strokes += [pss]
                        mid_points += [[(xmax + xmin) / 2., (ymax + ymin) / 2.]]
                distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                             for p1, p2 in zip(mid_points, mid_points[1:])]
                splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)
                lines = []
                for b, e in zip([0] + splits, splits + [len(strokes)]):
                    lines += [[p for pts in strokes[b:e] for p in pts]]
                print('lines = {:4d}; texts = {:4d}'.format(len(lines), len(texts)))
                charset |= set(''.join(texts))
                data += [(texts, lines)]
    print('data = {}; charset = ({}) {}'.format(len(data), len(charset), ''.join(sorted(charset))))

    translation = {'<NULL>': 0}
    for c in ''.join(sorted(charset)):
        translation[c] = len(translation)

    def translate(txt):
        return list(map(lambda x: translation[x], txt))

    dataset = []
    labels = []
    for texts, lines in data:
        for text, line in zip(texts, lines):
            line = np.array(line, dtype=np.float32)
            line[:, 0] = line[:, 0] - np.min(line[:, 0])
            line[:, 1] = line[:, 1] - np.mean(line[:, 1])

            dataset += [line]
            labels += [translate(text)]

    whole_data = np.concatenate(dataset, axis=0)

    std_y = np.std(whole_data[:, 1])
    norm_data = []
    for line in dataset:
        line[:, :2] /= std_y
        norm_data += [line]
    dataset = norm_data

    print('datset = {}; labels = {}'.format(len(dataset), len(labels)))

    save_path = os.path.join(partial_path, 'preprocessed_data')
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass
    np.save(os.path.join(save_path, 'dataset'), np.array(dataset))
    np.save(os.path.join(save_path, 'labels'), np.array(labels))
    with open(os.path.join(save_path, 'translation.pkl'), 'wb') as file:
        pickle.dump(translation, file)
    print("Preprocessing finished and cached at {}".format(save_path))


USER = os.getenv("USER")
def get_dataset_dir(dataset_name):
    local_cache_dir = "/Tmp/" + USER + "/_data_cache/" + dataset_name
    if not os.path.exists(local_cache_dir):
        os.makedirs(local_cache_dir)
    return local_cache_dir


def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    partial_path = get_dataset_dir("iamondb")
    #partial_path = get_dataset_dir("iamondb")
    #partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "iamondb"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(combined_data_path):
        files = "original-xml-part.tar.gz"
        url = "http://www.iam.unibe.ch/fki/databases/"
        url += "iam-on-line-handwriting-database/"
        url += "download-the-iam-on-line-handwriting-database"
        err = "Path %s does not exist!" % combined_data_path
        err += " Download the %s files from %s" % (files, url)
        err += " and place them in the directory %s" % partial_path
        print("WARNING: {}".format(err))
    return partial_path


def fetch_iamondb():
    partial_path = check_fetch_iamondb()
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(untarred_data_path):
        print("Now untarring {}".format(combined_data_path))
        tar = tarfile.open(combined_data_path, "r:gz")
        tar.extractall(partial_path)
        tar.close()

    saved_dataset_path = os.path.join(partial_path, 'preprocessed_data')

    if not os.path.exists(saved_dataset_path):
        iamondb_extract(partial_path)

    dataset_path = os.path.join(saved_dataset_path, "dataset.npy")
    labels_path = os.path.join(saved_dataset_path, "labels.npy")
    translation_path = os.path.join(saved_dataset_path, "translation.pkl")

    dataset = np.load(dataset_path)
    dataset = [np.array(d) for d in dataset]

    temp = []
    for d in dataset:
        # dataset stores actual pen points, but we will train on differences between consecutive points
        offs = d[1:, :2] - d[:-1, :2]
        ends = d[1:, 2]
        temp += [np.concatenate([[[0., 0., 1.]], np.concatenate([offs, ends[:, None]], axis=1)], axis=0)]
    # because lines are of different length, we store them in python array (not numpy)
    dataset = temp
    labels = np.load(labels_path)
    labels = [np.array(l) for l in labels]
    with open(translation_path, 'rb') as f:
        translation = pickle.load(f)
    # be sure of consisten ordering
    new_translation = OrderedDict()
    for k in sorted(translation.keys()):
        new_translation[k] = translation[k]
    translation = new_translation
    dataset_storage = {}
    dataset_storage["data"] = dataset
    dataset_storage["target"] = labels
    inverse_translation = {v: k for k, v in translation.items()}
    dataset_storage["target_phrases"] = ["".join([inverse_translation[ci] for ci in labels[i]]) for i in range(len(labels))]
    dataset_storage["vocabulary_size"] = len(translation)
    dataset_storage["vocabulary"] = translation
    return dataset_storage


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=True, verbose=True):
    """
    Print and execute command on system
    """
    all_lines = []
    for line in execute(cmd, shell=shell):
        if verbose:
            print(line, end="")
        all_lines.append(line.strip())
    return all_lines


# https://mrcoles.com/blog/3-decorator-examples-and-awesome-python/
def rsync_fetch(fetch_func, machine_to_fetch_from, *args, **kwargs):
    """
    assumes the filename in IOError is a subdir, will rsync one level above that

    be sure not to call it as
    rsync_fetch(fetch_func, machine_name)
    not
    rsync_fetch(fetch_func(), machine_name)
    """
    try:
        r = fetch_func(*args, **kwargs)
    except Exception as e:
        if isinstance(e, IOError):
            full_path = e.filename
            filedir = str(os.sep).join(full_path.split(os.sep)[:-1])
            if not os.path.exists(filedir):
                if filedir[-1] != "/":
                    fd = filedir + "/"
                else:
                    fd = filedir
                os.makedirs(fd)

            if filedir[-1] != "/":
                fd = filedir + "/"
            else:
                fd = filedir

            if not os.path.exists(full_path):
                sdir = str(machine_to_fetch_from) + ":" + fd
                cmd = "rsync -avhp --progress %s %s" % (sdir, fd)
                pe(cmd, shell=True)
        else:
            print("unknown error {}".format(e))
        r = fetch_func(*args, **kwargs)
    return r
