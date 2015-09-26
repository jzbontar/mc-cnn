#! /usr/bin/env python2
# wget -r -np -A png,pfm,pgm,txt http://vision.middlebury.edu/stereo/data/scenes2014/datasets/
# wget -r -np -A png,pfm,pgm,txt http://vision.middlebury.edu/stereo/data/scenes2006/FullSize/

import os
import re
import sys
import subprocess

import numpy as np
import cv2

def load_pfm(fname, downsample):
  if downsample:
        if not os.path.isfile(fname + '.H.pfm'):
            x, scale = load_pfm(fname, False)
            x = x / 2
            x_ = np.zeros((x.shape[0] // 2, x.shape[1] // 2), dtype=np.float32)
            for i in range(0, x.shape[0], 2):
                for j in range(0, x.shape[1], 2):
                    tmp = x[i:i+2,j:j+2].ravel()
                    x_[i // 2,j // 2] = np.sort(tmp)[1]
            save_pfm(fname + '.H.pfm', x_, scale)
            return x_, scale
        else:
            fname += '.H.pfm'
  color = None
  width = None
  height = None
  scale = None
  endian = None
  
  file = open(fname)
  header = file.readline().rstrip()
  if header == 'PF':
    color = True    
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')
 
  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')
 
  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian
 
  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.flipud(np.reshape(data, shape)), scale

def save_pfm(fname, image, scale=1):
  file = open(fname, 'w') 
  color = None
 
  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')
 
  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))
 
  endian = image.dtype.byteorder
 
  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale
 
  file.write('%f\n' % scale)
 
  np.flipud(image).tofile(file)

def read_im(fname, downsample):
    if downsample:
        if not os.path.isfile(fname + '.H.png'):
            subprocess.check_call('convert {} -resize 50% {}.H.png'.format(fname, fname).split())
        fname += '.H.png'
    x = cv2.imread(fname).astype(np.float32)
    if color == 'rgb':
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose(2, 0, 1)
    else:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)[None]
    x = (x - x.mean()) / x.std()
    return x[None]

def tofile(fname, x):
    if x is None:
        open(fname + '.dim', 'w').write('0\n')
        open(fname, 'w')
    else:
        x.tofile(fname)
        open(fname + '.type', 'w').write(str(x.dtype))
        open(fname + '.dim', 'w').write('\n'.join(map(str, x.shape)))

rectification, color = sys.argv[1:]
assert(rectification in set(['perfect', 'imperfect']))
assert(color in set(['gray', 'rgb']))
output_dir = 'data.mb.{}_{}'.format(rectification, color)
assert(os.path.isdir(output_dir))

num_channels = 3 if color == 'rgb' else 1

X = []
dispnoc = []
meta = []
nnz_tr = []
nnz_te = []
te = np.arange(1, 11)

### 2014 dataset ###
base1 = 'data.mb/unzip/vision.middlebury.edu/stereo/data/scenes2014/datasets'
for dir in sorted(os.listdir(base1)):
    if dir.endswith('imperfect'):
        print(dir.split('-')[0])

        base2_imperfect = os.path.join(base1, dir)
        base2_perfect = base2_imperfect.replace('imperfect', 'perfect')

        calib = open(os.path.join(base2_imperfect, 'calib.txt')).read()
        ndisp = int(re.search('ndisp=(.*)', calib).group(1)) / 2

        x0 = read_im(os.path.join(base2_imperfect, 'im0.png'), True)
        x1 = read_im(os.path.join(base2_imperfect, 'im1.png'), True)
        x1E = read_im(os.path.join(base2_imperfect, 'im1E.png'), True)
        x1L = read_im(os.path.join(base2_imperfect, 'im1L.png'), True)
        XX = [np.concatenate((x0, x1, x1E, x1L))]

        base3 = os.path.join(base2_perfect if rectification == 'perfect' else base2_imperfect, 'ambient')
        num_light = len(os.listdir(base3))

        num_exp = [], []
        for fname in os.listdir(base3 + '/L1'):
            num_exp[int(fname[2])].append(int(fname[4]) + 1)
        num_exp = min(max(num_exp[0]), max(num_exp[1]))
        rng = {
            8: [1, 3, 5],
            7: [1, 3, 5],
            6: [0, 2, 4],
            5: [0, 2, 4],
            3: [0, 1, 2],
            2: [0, 1],
        }
        for light in range(num_light):
            imgs = []
            base4 = os.path.join(base3, 'L{}'.format(light + 1))
            for exp in rng[num_exp]:
                for cam in range(2):
                    im = read_im(base4 + '/im{}e{}.png'.format(cam, exp), True)
                    imgs.append(im)
            _, _, height, width = imgs[0].shape
            XX.append(np.concatenate(imgs).reshape(len(imgs) // 2, 2, num_channels, height, width))

        disp0, scale0 = load_pfm(os.path.join(base2_imperfect, 'disp0.pfm'), True)
        disp1, scale1 = load_pfm(os.path.join(base2_imperfect, 'disp1.pfm'), True)
        disp0y, scale0y = load_pfm(os.path.join(base2_imperfect, 'disp0y.pfm'), True)

        save_pfm('tmp/disp0.pfm', disp0, 1)
        save_pfm('tmp/disp1.pfm', disp1, 1)
        save_pfm('tmp/disp0y.pfm', disp0y, 1)

        subprocess.check_output('computemask tmp/disp0.pfm tmp/disp0y.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

        mask = cv2.imread('tmp/mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        X.append(XX)
        nnz = nnz_te if len(X) in te else nnz_tr
        nnz.append(np.column_stack((np.zeros_like(y) + len(X), y, x, disp0[y, x])).astype(np.float32))
        dispnoc.append(disp0.astype(np.float32))
        meta.append((x0.shape[2], x0.shape[3], ndisp))        

print(np.vstack(nnz_tr).shape)

### 2006 & 2005 dataset ###
for year in (2006, 2005):
    base1 = 'data.mb/unzip/vision.middlebury.edu/stereo/data/scenes{}/HalfSize'.format(year)
    for dir in sorted(os.listdir(base1)):
        base2 = os.path.join(base1, dir)
        if not os.path.isfile(base2 + '/disp1.png'):
            continue

        print(dir)

        XX = []
        XX.append(None)  # there are no test images for this dataset
        for light in range(3):
            imgs = []
            for exp in (0, 1, 2):
                base3 = os.path.join(base2, 'Illum{}/Exp{}'.format(light + 1, exp))
                x0 = read_im(os.path.join(base3, 'view1.png'), False)
                x1 = read_im(os.path.join(base3, 'view5.png'), False)
                imgs.append(x0)
                imgs.append(x1)
            _, _, height, width = imgs[0].shape
            XX.append(np.concatenate(imgs).reshape(len(imgs) // 2, 2, num_channels, height, width))

        disp0 = cv2.imread(base2 + '/disp1.png', 0).astype(np.float32) / 2
        disp1 = cv2.imread(base2 + '/disp5.png', 0).astype(np.float32) / 2

        ndisp = int(np.ceil(disp0.max()))
        disp0[disp0 == 0] = np.inf
        disp1[disp1 == 0] = np.inf

        save_pfm('tmp/disp0.pfm', disp0, 1)
        save_pfm('tmp/disp1.pfm', disp1, 1)

        subprocess.check_output('computemask tmp/disp0.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

        mask = cv2.imread('tmp/mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        X.append(XX)
        nnz_tr.append(np.column_stack((np.zeros_like(y) + len(X), y, x, disp0[y, x])).astype(np.float32))
        dispnoc.append(disp0.astype(np.float32))
        meta.append((x0.shape[2], x0.shape[3], ndisp))
    print(np.vstack(nnz_tr).shape)

### 2003 dataset ###
for dir in ('conesH', 'teddyH'):
    print(dir)
    base1 = 'data.mb/unzip/vision.middlebury.edu/stereo/data/scenes2003/{}'.format(dir)

    XX = []
    XX.append(None)

    x0 = read_im(base1 + '/im2.ppm', False)
    x1 = read_im(base1 + '/im6.ppm', False)
    _, _, height, width = x0.shape
    XX.append(np.concatenate((x0, x1)).reshape(1, 2, num_channels, height, width))

    disp0 = cv2.imread(base1 + '/disp2.pgm', 0).astype(np.float32) / 2
    disp1 = cv2.imread(base1 + '/disp6.pgm', 0).astype(np.float32) / 2
    ndisp = int(np.ceil(disp0.max()))
    disp0[disp0 == 0] = np.inf
    disp1[disp1 == 0] = np.inf

    save_pfm('tmp/disp0.pfm', disp0, 1)
    save_pfm('tmp/disp1.pfm', disp1, 1)

    subprocess.check_output('computemask tmp/disp0.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

    mask = cv2.imread('tmp/mask.png', 0)
    disp0[mask != 255] = 0
    y, x = np.nonzero(mask == 255)

    X.append(XX)
    nnz_tr.append(np.column_stack((np.zeros_like(y) + len(X), y, x, disp0[y, x])).astype(np.float32))
    dispnoc.append(disp0.astype(np.float32))
    meta.append((x0.shape[2], x0.shape[3], ndisp))

print(np.vstack(nnz_tr).shape)

### 2001 dataset ###
base1 = 'data.mb/unzip/vision.middlebury.edu/stereo/data/scenes2001/data'
for dir in sorted(os.listdir(base1)):
    if dir == 'tsukuba':
        fname_disp0, fname_disp1, fname_x0, fname_x1 = 'truedisp.row3.col3.pgm', '', 'scene1.row3.col3.ppm', 'scene1.row3.col4.ppm'
    elif dir == 'map':
        fname_disp0, fname_disp1, fname_x0, fname_x1 = 'disp0.pgm', 'disp1.pgm', 'im0.pgm', 'im1.pgm'
    else:
        fname_disp0, fname_disp1, fname_x0, fname_x1 = 'disp2.pgm', 'disp6.pgm', 'im2.ppm', 'im6.ppm'

    base2 = os.path.join(base1, dir)
    if os.path.isfile(os.path.join(base2, fname_disp0)):
        print(dir)

        XX = []
        XX.append(None)

        x0 = read_im(os.path.join(base2, fname_x0), False)
        x1 = read_im(os.path.join(base2, fname_x1), False)
        _, _, height, width = x0.shape
        XX.append(np.concatenate((x0, x1)).reshape(1, 2, num_channels, height, width))

        if dir == 'tsukuba':
            disp0 = cv2.imread(os.path.join(base2, fname_disp0), 0).astype(np.float32) / 16
            mask = cv2.imread(os.path.join(base2, 'nonocc.png'), 0)
        else:
            disp0 = cv2.imread(os.path.join(base2, fname_disp0), 0).astype(np.float32) / 8
            disp1 = cv2.imread(os.path.join(base2, fname_disp1), 0).astype(np.float32) / 8

            save_pfm('tmp/disp0.pfm', disp0, 1)
            save_pfm('tmp/disp1.pfm', disp1, 1)
            subprocess.check_output('computemask tmp/disp0.pfm tmp/disp1.pfm -1 tmp/mask.png'.split())

            mask = cv2.imread('tmp/mask.png', 0)
        disp0[mask != 255] = 0
        y, x = np.nonzero(mask == 255)

        X.append(XX)
        nnz_tr.append(np.column_stack((np.zeros_like(y) + len(X), y, x, disp0[y, x])).astype(np.float32))
        dispnoc.append(disp0.astype(np.float32))
        meta.append((x0.shape[2], x0.shape[3], -1))

### test ###
fname_submit = []

base1 = 'data.mb/unzip/MiddEval3'
for dir1 in ['trainingH', 'testH']:
    base2 = os.path.join(base1, dir1)
    for dir2 in sorted(os.listdir(base2)):
        base3 = os.path.join(base2, dir2)
        print(os.path.join(dir1, dir2))

        calib = open(os.path.join(base3, 'calib.txt')).read()
        ndisp = int(re.search('ndisp=(.*)', calib).group(1))

        x0 = read_im(os.path.join(base3, 'im0.png'), False)
        x1 = read_im(os.path.join(base3, 'im1.png'), False)

        X.append([np.concatenate((x0, x1)).astype(np.float32)])
        meta.append((x0.shape[2], x0.shape[3], ndisp))
        fname_submit.append(os.path.join(dir1, dir2))

meta = np.array(meta, dtype=np.int32)
nnz_tr = np.vstack(nnz_tr)
nnz_te = np.vstack(nnz_te)

subprocess.check_call('rm -f {}/*.{{bin,dim,txt,type}} tmp/*'.format(output_dir), shell=True)
for i in range(len(X)):
    for j in range(len(X[i])):
        tofile('{}/x_{}_{}.bin'.format(output_dir, i + 1, j + 1), X[i][j])
    if i < len(dispnoc):
        tofile('{}/dispnoc{}.bin'.format(output_dir, i + 1), dispnoc[i])
tofile('{}/meta.bin'.format(output_dir), meta)
tofile('{}/nnz_tr.bin'.format(output_dir), nnz_tr)
tofile('{}/nnz_te.bin'.format(output_dir), nnz_te)
tofile('{}/te.bin'.format(output_dir), te)
open('{}/fname_submit.txt'.format(output_dir), 'w').write('\n'.join(fname_submit))
