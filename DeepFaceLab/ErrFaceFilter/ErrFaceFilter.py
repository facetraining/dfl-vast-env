# -*- coding:utf-8 -*-

import os, sys, shutil, time
import numpy as np
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from DFLIMG import DFLIMG
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Value

func = int(sys.argv[1])
if func == 1:
    import xgboost as xgb
    model_path = Path(__file__).parent / 'CL.model'
    if not os.path.exists(os.path.join(os.getcwd(), model_path)):
        raise FileNotFoundError("No model files found, please place the provided model files in the source code directory!")
    Classifier = xgb.XGBClassifier()
    Classifier.load_model(model_path)
elif func == 2:
    import cv2
    jaw = (255, 255, 255)
    eyelash = (255, 255, 255)
    nose = (255, 255, 255)
    eyes = (255, 255, 255)
    mouth = (255, 255, 255)

def LstCleaning(lst: list, path):
    dellst = []
    for it in lst:
        if not (it.lower().endswith(('.jpg', '.jpeg'))):
            dellst.append(it)
    for it in dellst:
        lst.remove(it)
    if len(lst) < 1:
        raise FileNotFoundError('No objects were found in the selected directory, please check and re-run!')
    else:
        print('Number of images: \033[32m%d\033[0m' % len(lst))
    lst.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))

def draw(lst, fpath, lmdir, cnt):
    for fname in lst:
        with cnt.get_lock():
            cnt.value += 1
        path = Path(fpath + '/' + fname)
        dflimg = DFLIMG.load(path)
        if dflimg is None or not dflimg.has_data():
            print(f'\n{fname} Not a valid DFL avatar file!')
            continue
        else:
            x, y = np.split(dflimg.get_landmarks(), 2, axis=1)
        size = dflimg.get_shape()
        im = np.zeros(size, dtype=np.uint8)
        if size[0] <= 768:
            line_width = 5
        else:
            line_width = 12

        for i in range(0, 16):  # jaw
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), jaw, line_width)

        for i in range(17, 26):  # eyelash
            if i == 21:
                continue
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), eyelash, line_width)

        for i in range(27, 35):  # nose
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), nose, line_width)

        for i in range(36, 47):  # eyes
            if i == 41:
                img = cv2.line(im, (int(x[41][0]), int(y[41][0])), (int(x[36][0]), int(y[36][0])), eyes, line_width)
                continue
            if i == 46:
                img = cv2.line(im, (int(x[47][0]), int(y[47][0])), (int(x[42][0]), int(y[42][0])), eyes, line_width)
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), eyes, line_width)

        for i in range(48, 68):  # mouth & lips
            if i == 59:
                img = cv2.line(im, (int(x[59][0]), int(y[59][0])), (int(x[48][0]), int(y[48][0])), mouth, line_width)
                continue
            if i == 67:
                img = cv2.line(im, (int(x[67][0]), int(y[67][0])), (int(x[60][0]), int(y[60][0])), mouth, line_width)
                continue
            img = cv2.line(im, (int(x[i][0]), int(y[i][0])), (int(x[i + 1][0]), int(y[i + 1][0])), mouth, line_width)

        filename = os.path.join(lmdir, fname)
        cv2.imwrite(filename, img)

def CF(flst, path, errfacelst, cnt):
    for fname in flst:
        with cnt.get_lock():
            cnt.value += 1
        dflimg = DFLIMG.load(Path(path + '/' + fname))
        if dflimg is None or not dflimg.has_data():
            print(f'\n{fname} Not a valid DFL avatar file!')
            continue
        else:
            lm = dflimg.get_landmarks()
        h = dflimg.get_shape()[0]
        if h != 512:
            lm = lm * 512 / h
        cat = Classifier.predict(lm.reshape((1, 136)).astype(np.int64))
        if cat == 1:
            errfacelst.append(fname)

if __name__ == '__main__':
    while 1:
        os.system('cls')
        print("➜ Free tools, rest assured enjoy\n\033[33m➜ Author: Fruit-eating Raccoon\033[0m")

        path = input('Please fill in the full folder path where the DFL avatar is stored (e.g. aligned), and press Enter \n(\033[33m Warning: Do not include any Chinese characters or spaces in the directory and avatar file name\033[0m)\n➜ ')
        if os.path.isdir(path):
            imagelst = os.listdir(path)
        else:
            raise ValueError('Please enter the correct catalog')
        os.system('cls')
        print('Selected Catalog: ', path)
        LstCleaning(imagelst, path)
        if func == 3:
            print('Script Function: Wrong Face Cleanup')
            indir = os.path.join(path, 'Landmarks')
            if not os.path.exists(indir):
                print('\nLandmarks directory not found, please check and try again!\n')
                input('Press any key to restart the script...')
                continue
            lmlst = os.listdir(indir)
            print('%scatalogs, ' % indir, end='')
            LstCleaning(lmlst, path)
            if len(lmlst) < len(imagelst):
                trash = list(set(imagelst) - set(lmlst))
            else:
                print('\nNo Landmarks detected and rejected!\n')
                input('Press any key to restart the script...')
                continue
            movepath = os.path.join(path, 'errFace')
            if not os.path.exists(movepath):
                os.makedirs(movepath)
            print('Quantity cleared: \033[33m%d\033[0m' % len(trash))
            for it in tqdm(trash, desc='workflow'):
                src = os.path.join(path, it)
                dst = os.path.join(movepath, it)
                try:
                    shutil.move(src, dst)
                except:
                    print('%s File move failed!', it)
            print('\nThe error avatar has been moved to %s\n' % movepath)
            input('Press any key to restart the script...')
        else:
            cpu = mp.cpu_count() - 1
            amount = len(imagelst) // cpu
            cpu = 1 if amount == 0 else cpu
            checklst = []
            for i in range(cpu):
                if i == cpu - 1:
                    checklst.append(imagelst[i * amount:])
                    break
                checklst.append(imagelst[i * amount:(i + 1) * amount])
            progresscnt = Value('L', 0)
            if func == 1:
                print('Script Function: Automatically Screen Wrong Faces')
                errfacelst = mp.Manager().list()
                mptask = [mp.Process(target=CF, args=(checklst[i], path, errfacelst, progresscnt, ), daemon=True) for i in range(cpu)]
                progress = tqdm(desc='detection process', total=len(imagelst))
            elif func == 2:
                print('Script Function: Outlining')
                LmDir = os.path.join(path, 'Landmarks')
                if not os.path.exists(LmDir):
                    os.mkdir(LmDir)
                mptask = [mp.Process(target=draw, args=(checklst[i], path, LmDir, progresscnt, ), daemon=True) for i in range(cpu)]
                progress = tqdm(desc='Painting process', total=len(imagelst))
            [t.start() for t in mptask]
            while progress.n < len(imagelst):
                progress.n = progresscnt.value
                progress.refresh()
            progress.close()
            progress = None
            [t.join() for t in mptask]
            if func == 2:
                input('\nJob completed, press any key to restart the script....\n')
                continue

            if len(errfacelst) < 1:
                print("\nThere doesn't seem to be a wrong face within the catalog!")
                input('\nJob completed, press any key to restart the script....\n')
                continue
            errfacepath = os.path.join(path, 'errFace')
            if not os.path.exists(errfacepath):
                os.mkdir(errfacepath)
            for it in tqdm(errfacelst, desc='workflow'):
                src = os.path.join(path, it)
                dst = os.path.join(errfacepath, it)
                try:
                    shutil.move(src, dst)
                except:
                    print('draw attention to sth.: %s The file was not moved successfully, please process manually.!' % it)
            print('\nI found \033[33m%d\033[0m the wrong avatar, and moved it. %s' % (len(errfacelst), errfacepath))
            input('\nJob completed, press any key to restart the script....\n')