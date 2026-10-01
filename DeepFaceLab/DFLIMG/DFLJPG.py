import pickle
import struct
import traceback

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.imagelib import SegIEPolys
from core.interact import interact as io
from core.structex import *
from facelib import FaceType


class DFLJPG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack ("BB", data[data_counter:data_counter+2])
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F

                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True

                #if is_unk_chunk:
                #    #raise ValueError(f"Unknown chunk {chunk_m_h} in {filename}")
                #    io.log_info(f"Unknown chunk {chunk_m_h} in {filename}")

                if chunk_size == None: #variable size
                    chunk_size, = struct.unpack (">H", data[data_counter:data_counter+2])
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and (data[c] != 0xFF or data[c+1] != 0xD9):
                        c += 1

                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append ({'name' : chunk_name,
                                'm_h' : chunk_m_h,
                                'data' : chunk_data,
                                'ex_data' : chunk_ex_data,
                                })
            inst.chunks = chunks

            return inst
        except Exception as e:
            raise Exception (f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLJPG.load_raw (filename, loader_func=loader_func)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack (d, c, "=4sB")

                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack (d, c, "=BBBHHBB")
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id) )
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack (d, c, ">BHH")
                    inst.shape = (height, width, 3)

                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            return inst
        except Exception as e:
            io.log_err (f'Exception occured while DFLJPG.load : {traceback.format_exc()}')
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( f'cannot save {self.filename}' )

    def dump(self):
        data = b""

        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate (self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i

        dflchunk = {'name' : 'APP15',
                    'm_h' : 0xEF,
                    'data' : pickle.dumps(dict_data),
                    'ex_data' : None,
                    }
        self.chunks.insert (last_app_chunk+1, dflchunk)


        for chunk in self.chunks:
            data += struct.pack ("BB", 0xFF, chunk['m_h'] )
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack (">H", len(chunk_data)+2 )
                data += chunk_data

            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                data += chunk_ex_data

        return data

    def get_img(self):
        if self.img is None:
            self.img = cv2_imread(self.filename)
        return self.img

    def get_shape(self):
        if self.shape is None:
            img = self.get_img()
            if img is not None:
                self.shape = img.shape
        return self.shape

    def get_height(self):
        for chunk in self.chunks:
            if type(chunk) == IHDR:
                return chunk.height
        return 0

    def get_dict(self):
        return self.dfl_dict

    def set_dict (self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):            
        return self.dfl_dict.get('face_type', FaceType.toString (FaceType.FULL) )
    
    def set_face_type(self, face_type): 
        self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):       
        data=self.dfl_dict.get('landmarks',None)
        if data is None:
            return None
        else:
            return np.array ( data)

    def set_landmarks(self, landmarks): 
        self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):                      
        return self.dfl_dict.get ('eyebrows_expand_mod', 1.0)
    
    def set_eyebrows_expand_mod(self, eyebrows_expand_mod): 
        self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):                  
        return self.dfl_dict.get ('source_filename', None)
    
    def set_source_filename(self, source_filename): 
        self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              
        return self.dfl_dict.get ('source_rect', None)
    
    def set_source_rect(self, source_rect): 
        self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     
        return np.array ( self.dfl_dict.get('source_landmarks', None) )
    
    def set_source_landmarks(self, source_landmarks):   
        self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None
    
    def set_image_to_face_mat(self, image_to_face_mat):   
        self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_seg_ie_polys(self):
        return self.dfl_dict.get('seg_ie_polys',None) is not None

    def get_seg_ie_polys(self):
        d = self.dfl_dict.get('seg_ie_polys',None)
        if d is not None:
            d = SegIEPolys.load(d)
        else:
            d = SegIEPolys()

        return d

    def set_seg_ie_polys(self, seg_ie_polys):
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.dfl_dict['seg_ie_polys'] = seg_ie_polys

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask',None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None
        return mask_buf
        
    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[...,None]

        return img.astype(np.float32) / 255.0


    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return
        mask_a = imagelib.normalize_channels(mask_a, 1)
        img_data = np.clip( mask_a*255, 0, 255 ).astype(np.uint8)
        data_max_len = 50000
        ret, buf = cv2.imencode('.png', img_data)
        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100,-1,-1):
                ret, buf = cv2.imencode( '.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] )
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")
        self.dfl_dict['xseg_mask'] = buf

    def get_mouth_mask(self):
        landmarks=self.get_landmarks()
        if landmarks is None:
            print(f"!!! no landmarks data({self.filename})")
            return
        n,p=landmarks.shape
        if n!=68:
            print(f"!!! landmarks point num is not 68 ({self.filename})")
            return
        mouth_idx=[48,49,50,51,52,53,54,55,56,57,58,59,48]
        pts=landmarks[mouth_idx].astype(np.int32)
        mask=np.zeros_like(self.get_img())
        cv2.fillPoly(mask, [pts], (1,1,1) )
        self.set_xseg_mask(mask)
        self.save()
        return self.get_img()*mask

    def create_mouth_ie_polygon(self):
        landmarks=self.get_landmarks()
        if landmarks is None:
            return
        n,p=landmarks.shape
        if n!=68:
            return
        ie_polygons=self.get_seg_ie_polys()
        #ie_polygons.polys.clear()
        poly=ie_polygons.add_poly(1)
        mouth_idx=[48,49,50,51,52,53,54,55,56,57,58,59,48]
        pts=landmarks[mouth_idx].astype(np.int32)
        for pt in pts:
            poly.add_pt(pt[0],pt[1])
        self.set_seg_ie_polys(ie_polygons)
        self.save()

    def create_teeth_ie_polygon(self):
        landmarks=self.get_landmarks()
        if landmarks is None:
            return
        n,p=landmarks.shape
        if n!=68:
            return
        ie_polygons=self.get_seg_ie_polys()
        #ie_polygons.polys.clear()
        poly=ie_polygons.add_poly(1)
        mouth_idx=[60,61,62,63,64,65,66,67,60]
        pts=landmarks[mouth_idx].astype(np.int32)
        for pt in pts:
            poly.add_pt(pt[0],pt[1])
        self.set_seg_ie_polys(ie_polygons)
        self.save()

    def create_eyes_ie_polygon(self):
        landmarks=self.get_landmarks()
        if landmarks is None:
            return
        n,p=landmarks.shape
        if n!=68:
            return
        
        left_eye_idx=[36,37,38,39,40,41,36]
        left_eye_pts=landmarks[left_eye_idx].astype(np.int32)
        right_eye_idx=[42,43,44,45,46,47,42]
        right_eye_pts=landmarks[right_eye_idx].astype(np.int32)

        ie_polygons=self.get_seg_ie_polys()
        #ie_polygons.polys.clear()
        left_eye_poly=ie_polygons.add_poly(1)
        for pt in left_eye_pts:
            left_eye_poly.add_pt(pt[0],pt[1])
        right_eye_poly=ie_polygons.add_poly(1)
        for pt in right_eye_pts:
            right_eye_poly.add_pt(pt[0],pt[1])
        
        self.set_seg_ie_polys(ie_polygons)
        self.save()

    def clear_ie_polygon(self):
        ie_polygons=self.get_seg_ie_polys()
        ie_polygons.polys.clear()
        self.set_seg_ie_polys(ie_polygons)
        self.save()


    def get_teeth_mask(self):
        landmarks=self.get_landmarks()
        if landmarks is None:
            print(f"!!! no landmarks data({self.filename})")
            return
        n,p=landmarks.shape
        if n!=68:
            print(f"!!! landmarks point num is not 68 ({self.filename})")
        mouth_idx=[60,61,62,63,64,65,66,67,60]
        pts=landmarks[mouth_idx].astype(np.int32)
        mask=np.zeros_like(self.get_img())
        cv2.fillPoly(mask, [pts], (1,1,1) )
        self.set_xseg_mask(mask)
        self.save()
        return self.get_img()*mask


    def get_eyes_mask(self):
        landmarks=self.get_landmarks()
        if landmarks is None:
            print(f"!!! no landmarks data({self.filename})")
            return
        n,p=landmarks.shape
        if n!=68:
            print(f"!!! landmarks point num is not 68 ({self.filename})")
        left_eye_idx=[36,37,38,39,40,41,36]
        left_eye_pts=landmarks[left_eye_idx].astype(np.int32)
        right_eye_idx=[42,43,44,45,46,47,42]
        right_eye_pts=landmarks[right_eye_idx].astype(np.int32)
        mask=np.zeros_like(self.get_img())
        cv2.fillPoly(mask, [left_eye_pts], (1,1,1) )
        cv2.fillPoly(mask, [right_eye_pts], (1,1,1) )
        self.set_xseg_mask(mask)
        self.save()
        return self.get_img()*mask

    def create_seg_polygon_from_landmarks(self,save=True):
        landmarks=self.get_landmarks()
        if landmarks is None:
            print(f"!!! no landmarks data({self.filename})")
            return 
        seg_ie_polys=self.get_seg_ie_polys()
        poly=seg_ie_polys.add_poly(1)
        face_out_landmarks=np.vstack([landmarks[0:17],landmarks[26:21:-1],landmarks[21:16:-1],landmarks[0]])
        poly.set_points(face_out_landmarks)
        self.set_seg_ie_polys(seg_ie_polys)
        if save:
            self.save()

    #--- Calculate seg_polygon from mask
    def create_seg_polygon_from_mask(self,save=True):
        xseg_mask=self.get_xseg_mask()
        img=self.get_img()
        if xseg_mask is None: 
            print(f"Create editable polygon error (image without xseg mask).{self.filename} no xseg mask")
        xseg_mask=cv2.resize(xseg_mask,(img.shape[1],img.shape[0]))

        if xseg_mask is None:
            return
        xseg_mask=255*xseg_mask.astype(np.uint8)    
        kernel = np.ones((4, 4), np.uint8)
        xseg_mask = cv2.dilate(xseg_mask, kernel, iterations = 1)
        contours,hier=cv2.findContours(xseg_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        seg_ie_polys=self.get_seg_ie_polys()
        if seg_ie_polys.has_polys():
            seg_ie_polys.polys.clear()
        for  i  in range(len(contours)):  
            contour=contours[i]
            contour=contour[0:-1:5]
            if len(contour)<5:
                continue
            parent_idx=hier[0][i][3]
            poly_type=1 if parent_idx<0 else 0
            poly=seg_ie_polys.add_poly(ie_poly_type=poly_type)
            for pt in contour:
                poly.add_pt(pt[0][0],pt[0][1])
        self.set_seg_ie_polys(seg_ie_polys)
        if save:
            self.save()

    def dilate_xseg(self,n=2):
        pass