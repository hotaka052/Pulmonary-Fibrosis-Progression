import numpy as np
import pydicom
from skimage.measure import label,regionprops
from skimage.segmentation import clear_border
from multiprocessing import Pool

"""
肺の大きさをカラムに追加する
https://www.kaggle.com/khyeh0719/lung-volume-calculus-with-trapezoidal-rule
"""

class Detector:
    def __call__(self, x):
        raise NotImplementedError('Abstract') 

class ThrDetector(Detector):
    def __init__(self, thr=-400):
        self.thr = thr
        
    def __call__(self, x):
        try:
            x = pydicom.dcmread(x)
            img = x.pixel_array
            img = (img + x.RescaleIntercept) / x.RescaleSlope
            img = img < self.thr
            
            img = clear_border(img)
            img = label(img)
            areas = [r.area for r in regionprops(img)]
            areas.sort()
            if len(areas) > 2:
                for region in regionprops(img):
                    if region.area < areas[-2]:
                        for coordinates in region.coords:                
                            img[coordinates[0], coordinates[1]] = 0
                            
            area = (img > 0).sum() * x.PixelSpacing[0] * x.PixelSpacing[1] # scale the detected lung area according the the pixel spacing value
            
        except:
            area = np.nan

        try:
            loc = x.ImagePositionPatient[2]
        except:
            loc = np.nan

        return area, loc

class Integral:
    def __init__(self, detector: Detector):
        self.detector = detector
    
    def __call__(self, xs):
        raise NotImplementedError('Abstract')
        
class AreaIntegral(Integral):
    def __call__(self, xs):
        
        with Pool(4) as p:
            areas, locs = map(list, zip(*p.map(self.detector, xs) ))
        
        filt = (~np.isnan(locs)) & (~np.isnan(areas))
        areas = np.array(areas)[filt]
        locs = np.array(locs)[filt]
        seq_idx = np.argsort(locs)

        return np.trapz(y=areas[seq_idx], x=locs[seq_idx])/1000