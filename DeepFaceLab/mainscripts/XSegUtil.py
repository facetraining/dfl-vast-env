import operator
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from DFLIMG import *
from facelib import FaceType, LandmarksProcessor, XSegNet
from samplelib import PackedFaceset


def is_packed(input_path):
    if PackedFaceset.path_contains(input_path):
        io.log_info (f'\n{input_path} contains packed faceset! Unpack it first.\n')
        return True

def apply_xseg(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if not model_path.exists():
        raise ValueError(f'{model_path} not found. Please ensure it exists.')

    if is_packed(input_path) : return
        
    face_type = None
    
    # Collect the names and last modification times of all model data files
    saved_models_names = []
    for filepath in pathex.get_file_paths(model_path):
        filepath_name = filepath.name
        if filepath_name.endswith(f'XSeg_data.dat'):
            # If the filename ends with a model class name, add the filename and last modification time to the list
            saved_models_names += [(filepath_name.split('_')[0], os.path.getmtime(filepath))]

    # Sorted in reverse chronological order of modification
    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True)
    saved_models_names = [x[0] for x in saved_models_names]

    # If there's a saved model
    if len(saved_models_names) == 1:
        model_name=saved_models_names[0]  #XSeg
        
    elif len(saved_models_names) > 1:

        io.log_info("Select a model")

        for i, model_name in enumerate(saved_models_names):
            s = f"[{i}] : {model_name} "
            if i == 0:
                s += "- Previous implementation"
            io.log_info(s)

        # User input of selected model index or action (rename or delete)
        inp = io.input_str(f"", "0", show_default_value=False)
        # Initialize variables
        model_idx = -1
        try:
            model_idx = np.clip(int(inp), 0, len(saved_models_names) - 1)
        except:
            pass

        if model_idx == -1:
            # means that the user input cannot be converted to a valid integer, or that the converted value is not within a legal index range
            model_name = inp
        else:
            # Set the current model name based on the index selected by the user
            model_name = saved_models_names[model_idx]

    else:
        # If there is no saved model, prompt the user to enter the name of a new model
        print("No XSeg model found, please download or train it.")


    if model_name == "XSeg":
        model_dat = model_path / ('XSeg_data.dat')
    else:
        model_dat = model_path / (model_name+'_XSeg_data.dat')
        
    if model_dat.exists():
        dat = pickle.loads( model_dat.read_bytes() )
        dat_options = dat.get('options', None)
        if dat_options is not None:
            face_type = dat_options.get('face_type', None)
            if model_name == "XSeg":
                full_name = "XSeg"
                resolution = 256
            else:
                resolution = dat_options.get('resolution', None)
                full_name = model_name+'_XSeg'

    if face_type is None:
        face_type = io.input_str ("XSeg model face type", 'same', ['h','mf','f','wf','head','same', 'custom'], help_message="Specify face type of trained XSeg model. For example if XSeg model trained as WF, but faceset is HEAD, specify WF to apply xseg only on WF part of HEAD. Default is 'same'").lower()
        if face_type == 'same':
            face_type = None
    
    if face_type is not None:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'custom' : FaceType.CUSTOM,
                     'head' : FaceType.HEAD}[face_type]
                     
    io.log_info(f'Applying trained XSeg model to {input_path.name}/ folder.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)

    xseg = XSegNet(name=full_name, 
                    resolution=resolution,
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    xseg_res = xseg.get_resolution()
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        
        img_face_type = FaceType.fromString( dflimg.get_face_type() )
        if face_type is not None and img_face_type != face_type or img_face_type == FaceType.CUSTOM: # custom always goes for eqvivalents
            lmrks = dflimg.get_source_landmarks()
            
            fmat = LandmarksProcessor.get_transform_mat(lmrks, w, face_type)
            imat = LandmarksProcessor.get_transform_mat(lmrks, w, img_face_type)
            
            g_p = LandmarksProcessor.transform_points (np.float32([(0,0),(w,0),(0,w) ]), fmat, True)
            g_p2 = LandmarksProcessor.transform_points (g_p, imat)
            
            mat = cv2.getAffineTransform( g_p2, np.float32([(0,0),(w,0),(0,w) ]) )
            
            img = cv2.warpAffine(img, mat, (w, w), cv2.INTER_LANCZOS4)
            img = cv2.resize(img, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        else:
            if w != xseg_res:
                img = cv2.resize( img, (xseg_res,xseg_res), interpolation=cv2.INTER_LANCZOS4 )    
                    
        if len(img.shape) == 2:
            img = img[...,None]            
    
        mask = xseg.extract(img)
        
        if face_type is not None and img_face_type != face_type or img_face_type == FaceType.CUSTOM:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine( mask, mat, (w,w), np.zeros( (h,w,c), dtype=np.float), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (xseg_res, xseg_res), interpolation=cv2.INTER_LANCZOS4)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        dflimg.set_xseg_mask(mask)
        dflimg.save()
        
def fetch_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if is_packed(input_path) : return
    
    output_path = input_path.parent / (input_path.name + '_xseg')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'Copying faces containing XSeg polygons to {output_path.name}/ folder.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        ie_polys = dflimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'Files copied: {len(files_copied)}')
    
    is_delete = io.input_bool (f"\r\nDelete original files?", True)
    if is_delete:
        for filepath in files_copied:
            Path(filepath).unlink()
               
def remove_xseg(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if is_packed(input_path) : return
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED XSEG MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue
        
        if dflimg.has_xseg_mask():
            dflimg.set_xseg_mask(None)
            dflimg.save()
            files_processed += 1
    io.log_info(f'Files processed: {files_processed}')
    
def remove_xseg_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if is_packed(input_path) : return
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED XSEG POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        dflimg = DFLIMG.load(filepath)
        if dflimg is None or not dflimg.has_data():
            io.log_info(f'{filepath} is not a DFLIMG')
            continue

        if dflimg.has_seg_ie_polys():
            dflimg.set_seg_ie_polys(None)
            dflimg.save()            
            files_processed += 1
            
    io.log_info(f'Files processed: {files_processed}')