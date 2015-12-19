#!/usr/bin/env python

import os
import mxnet as mx
import random
import numpy as np
import argparse
import threading
import scipy.io as sio
import cv, cv2
import time

def list_files(root, exts):
    """ List all files with extension names in exts in the root directory
    """
    print root
    file_list = []
    all_files = os.listdir(root)
    all_files.sort()
    for fname in all_files:
        fpath = os.path.join(root, fname)
        suffix = os.path.splitext(fname)[1].lower()
        if os.path.isfile(fpath) and (suffix in exts):
            file_list.append(os.path.relpath(fpath, root))
    return file_list

def list_images(root, gt, exts):
    """ List all image files with extension names in ext in the root directory 
    and assign labels (either groundtruth ones if provided or 0 by default).
    """    
    image_files = list_files(root, exts)    
    if not gt:
        image_list = [ (x, 0) for x in image_files ]
    else:
        if len(gt) != len(image_files):
            image_list = []
        else:
            image_list = []
            i = 0
            for f in image_files:
                image_list.append((f, gt[i]))
                i += 1
    return image_list

def list_images_in_subfolders(root, synsets, exts):
    """ List all image files with extension names in exts
    in the subfolders in the root directory and assign labels based on the 
    subfolder names and the provided synsets
    """
    image_list = []        
    subfolders = next(os.walk(root))[1]
    if not synsets:
        return image_list
    for f in subfolders:
        label = synsets[f]
        path = os.path.join(root, f)
        image_files = list_files(path, exts)
        cur_image_list = [ (os.path.join(f, img), label) for img in image_files ]
        image_list.extend(cur_image_list)    
    return image_list

def write_list(path_out, image_list):
    """ Write an image list """
    with open(path_out, 'w') as fout:
        for i in xrange(len(image_list)):
            fout.write('%d \t %d \t %s\n'%(i, image_list[i][1], image_list[i][0]))

def read_list(path_in):
    """ Read an image list """
    image_list = []
    with open(path_in) as fin:
        for line in fin.readlines():
            line = [i.strip() for i in line.strip().split('\t')]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            image_list.append(item)
    return image_list

def get_synsets(meta_file):
    """ Read the synsets provided by the ImageNet Toolkit """
    tmp = sio.loadmat(meta_file)    
    synsets = tmp['synsets']
    names = synsets.dtype.names
    synset_classId = {}
    synset_words = {}    
    for s in synsets:
        Class_ID = s[names[0]][0][0]
        WNID = s[names[1]][0][0]
        words = s[names[2]][0][0]
        synset_classId[WNID] = int(Class_ID)
        synset_words[WNID] = str(words)
    return synset_classId, synset_words                        

def make_list(prefix_out, root, gt_file, exts, num_chunks):
    """ Get the image list """
    if not gt_file: # test set 
        gt = []
        image_list = list_images(root, gt, exts)
    elif os.path.exists(gt_file):
        suffix = os.path.splitext(gt_file)[1].lower()
        if suffix == '.mat': # train set
            synset_classId, synset_words = get_synsets(gt_file)                        
            image_list = list_images_in_subfolders(root, synset_classId, exts)
        elif suffix == '.txt': # validation set
            gt = []
            with open(gt_file, 'r') as fin:
                for l in fin:
                   gt.append(int(l))            
            image_list = list_images(root, gt, exts)    
    random.shuffle(image_list)
    N = len(image_list)
    chunk_size = N/num_chunks
    for i in xrange(num_chunks):
        chunk = image_list[i*chunk_size:(i+1)*chunk_size]
        if num_chunks > 1:
            str_chunk = '_%d'%i
        else:
            str_chunk = ''
        write_list(prefix_out+str_chunk+'.lst', chunk)

def write_rec(args, image_list):
    source = image_list
    sink = []
    record = mx.recordio.MXRecordIO(args.prefix+'.rec', 'w')
    lock = threading.Lock()
    tic = [time.time()]
    def worker(i):
        item = source.pop(0)
        img = cv2.imread(os.path.join(args.root, item[1]))
        if args.center_crop:
            if img.shape[0] > img.shape[1]:
                margin = (img.shape[0] - img.shape[1]) / 2
                img = img[margin:margin+img.shape[1], :]
            else:
                margin = (img.shape[1] - img.shape[0]) / 2
                img = img[:, margin:margin+img.shape[0]]
        if args.resize:
            if img.shape[0] > img.shape[1]:
                newsize = [img.shape[0]*args.resize/img.shape[1], args.resize]
            else:
                newsize = [args.resize, img.shape[1]*args.resize/img.shape[0]]
            img = cv2.resize(img, newsize)
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)
        s = mx.recordio.pack_img(header, img, quality=args.quality)
        lock.acquire()
        record.write(s)
        sink.append(item)
        if len(sink)%1000 == 0:
            print len(sink), time.time() - tic[0]
            tic[0] = time.time()
        lock.relase()

    try:
        from multiprocessing.pool import ThreadPool
        multi_available = True
    except ImportError:
        print 'multiprocessing not available, fall back to single threaded encoding'
        multi_available = False

    if multi_available and args.num_thread > 1:
        p = ThreadPool(args.num_thread)
        p.map(worker, [i for i in range(len(source))])
        write_list(args.prefix+'.lst', sink)
    else:
        while len(source):
            worker(len(sink))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Make image list files for ImageNet and/or create the image record')
    parser.add_argument('root', help='path to folder that contain images.')
    parser.add_argument('prefix', help='prefix of output list files.')
    parser.add_argument('gt_file', help='meta matlab file for trainin set, \
        txt file for validation set provided by the ImageNet toolkit,\
        or empty for test set')
    parser.add_argument('--exts', type=list, default=['.jpeg','.jpg'],
                        help='list of acceptable image extensions.')
    parser.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    parser.add_argument('--resize', type=int, default=0,
                        help='resize the short edge of an image to the newsize')
    parser.add_argument('--center_crop', type=bool, default=False,
                        help='whether to crop the center image to make it square')
    parser.add_argument('--quality', type=int, default=80,
                        help='JPEG quality for encoding, 1-100')
    parser.add_argument('--num_thread', type=int, default=1,
                        help='number of threads to use for encoding. order of images will\
                        be different from the input list if >1. the input list will be rewritten\
                        accordingly.')
    args = parser.parse_args()

    make_list(args.prefix, args.root, args.gt_file, args.exts, args.chunks)

    image_list = read_list(args.prefix+'.lst')
    write_rec(args, image_list)
    
def test():
    prefix = 'val'
    root = '/media/tfwu/tfwuData/disk2.0tb/ImageNet/ILSVRC2012_img_train'
    gt_file = '/media/tfwu/tfwuData/disk2.0tb/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat'
    exts = ['.jpeg','.jpg']
    chunks = 10

    make_list(prefix, root, gt_file, exts, chunks)
    
if __name__ == '__main__':
    main()