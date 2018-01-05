from lung_seg import Lung_Seg
from airway_seg import airway_seg
import SimpleITK as ST
import numpy as np

def airway_segmentation(dicom_dir):
    lung_img = Lung_Seg(dicom_dir)
    airway_mask = airway_seg(lung_img)
    return airway_mask

# if __name__ =="__main__":
#     dicom_dir = "./case06/original1"
#     airway_mask = airway_segmentation(dicom_dir)