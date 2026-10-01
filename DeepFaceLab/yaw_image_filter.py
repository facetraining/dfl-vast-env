#%% -*- coding:utf-8 -*-
"""
Author: yangala@dfldata.xyz
Date: 2021-08-31 16:51:22

"""
#=====================================================
__version__ = '0.0.1'

import sys
import os

import random

from pathlib import Path
import shutil

from DFLIMG import DFLIMG,DFLJPG
from facelib import LandmarksProcessor,FaceType

from core.interact import interact as io

def getargv():
    if sys.argv.__len__()<=1:
        return ""
    return sys.argv[1]


def check(f,cmd): # Returns True if the condition is met

    dflimg = DFLJPG.load(f)

    if dflimg is None or not dflimg.has_data():
        print(f"{f.name} is not a dfl image file")
        return False,0,0

    ft = dflimg.get_face_type()
    ft = FaceType.fromString(ft)
    # print(int(ft),type(ft))
    ft = int(ft)
    # HALF = 0
    # MID_FULL = 1
    # FULL = 2
    # FULL_NO_ALIGN = 3
    # WHOLE_FACE = 4
    # HEAD = 10
    # HEAD_NO_ALIGN = 20
    # MARK_ONLY = 100


    pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks(), size=dflimg.get_shape()[1])

    k=180./3.14159

    # pitch >40 head up <-40 looking at the ground
    # yaw >45 left <-45 right

    y, x, roll = pitch*k, yaw*k, roll*k
    r = random.random()

    # print(exec(cmd,{'x':x,'y':y}),cmd,y,x,f)
    # print(eval(cmd),cmd,y,x,f)

    # return abs(x)>10 and abs(y)>10
    return eval(cmd),x,y




sort_func_methods = {
    'Large angle': ("Large angle: left and right facing greater than 40°, top and bottom greater than 40° ", '(abs(x)>=40 or abs(y)>=40)'),
    'Upper and lower large angle': ("Upper and lower large angle: greater than 40° up and down ", '(abs(y)>=40)'),
    'Upper and Lower Medium Angle': ("Upper and Lower Medium Angle: upper and lower greater than 30° ", '(abs(y)>=30)'),
    'Left-Right Large Angle': ("Left-Right Large Angle: Left-Right > 40° ", '(abs(x)>=40)'),
    'Extract 20% of positive faces': ("Extract 20% of positive faces: avatars with angle less than 20° randomly extract 20% of them ", '( abs(x)<20 and abs(y)<20 and r<0.2 ))'),
    'Non-wf faces': ("Non-wf faces: filter by face type", '(ft ! = 4)'),
    'custom': ("custom: x for left and right, y for up and down,r for random values from 0-1,ft for face type ", ''),
}

if __name__ == '__main__':

    print('yaw_image_filter.py Start ......')
    a=getargv()

    if os.path.exists(a):
        pass
    else:
        # print('Directory where jpg does not exist:', a)
        a=input('Path is invalid, please enter the directory where the jpg is located:')

    # Directory where the jpg is located
    alignedpath=Path(a)
    if alignedpath.is_file():
        alignedpath = alignedpath.parent


    print(f"\r\n avatar is in the directory: {alignedpath}\r\n")

    # Menu
    key_list = list(sort_func_methods.keys())
    for i, key in enumerate(key_list):
        desc, func = sort_func_methods[key]
        io.log_info(f"[{i}] {desc+' -> '+func}")

    io.log_info("")
    id = io.input_int("", 0, valid_list=[*range(len(key_list))] )

    sort_by_method = key_list[id]

    cmd = sort_func_methods[sort_by_method][1]
    print(sort_by_method,cmd)

    if cmd == '':
        print('''
    x means left and right, greater than 0 means left, less than 0 means right.
    y is up and down, greater than 0 means head up, less than 0 means head down
    r is a random value from 0 to 1

    ft face type wf=4 f=2 head=10

    abs() function to take absolute value

    and and
    or or
    not

    Example:
    r<0.2 means that 20% of the avatars are randomly selected.
    x>20 and y<-20 means randomize avatars that are 20° to the left and 20° down.
    ft!=4 means randomize non-wf faces.
        ''')
        cmd = input('Please enter the basis for your judgment:')
    # if cmd == '':
    #     print('cmd is empty, exit')
    #     exit(0)

    # print(sort_by_method, cmd)


    # Target directory
    mubiao = 'aligned_'+sort_by_method
    a = input(f'Please enter the destination directory name, if you press enter directly, it will be {mubiao}.:')
    if a == '':
        a = mubiao
    filterpath = alignedpath.parent / (a)
    if not filterpath.exists():
        filterpath.mkdir(parents=True)

    # Copy or move
    a = input(f'Whether to copy or move the file, 1 copy, 2 move, press enter to default to 1 copy.')
    if a=='' or a=='1':
        cpimg = 'Copy:'
    else:
        cpimg = 'Moving:'

    #
    #
    #
    # cmd = '(abs(x)>=40 or abs(y)>=30) and r<0.1 '
    # # cmd = 'x>10.0'

    # Statistical angular distribution
    tongji = [ [ 0 for b in range(7)   ] for a in range(7) ]
    def xyidx(x):
        return min(6,max(0,int(x+90+14.99999)//30))

    cnt = 0
    cntcopy = 0
    for f in alignedpath.glob('*.*'):
        # print(f.name)
        if f.is_file():
            cnt += 1
            rst,x,y = check(f,cmd)
            tongji[xyidx(y)][xyidx(x)] += 1
            if rst:
                # print(f.name)
                print(cnt,cpimg,f.name)
                dst = filterpath / f.name
                if cpimg == 'Copy:':
                    shutil.copy(f,dst)
                else:
                    shutil.move(f,dst)
                cntcopy += 1
            else:
                print(cnt,f.name)

    print()
    print(f'Processing result: {filterpath}')
    print(f'Total files {cnt}, {cpimg} {cntcopy} {sort_by_method}')

    jiaodu = ['','<-75','-60 ','-30 ','0±15','30  ','60  ','>75 ']
    print()
    print(' '+'\t'.join(jiaodu))
    print('------------------------------------------------------------')
    for a in range(7):
        print(jiaodu[a+1] +'\t| '+ '\t| '.join( [str(b if b>0 else '  ') for b in tongji[a]]  ))
        print('------------------------------------------------------------')



