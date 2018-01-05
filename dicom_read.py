# coding=utf-8
import SimpleITK as ST
import sys

def read_dicoms(input_directory):
    if len(sys.argv)<1:
        print "Usage: DicomSeriesReader <input_directory> <output_file>"
        sys.exit(1)

    print "Reading Dicom directory",input_directory
    reader=ST.ImageSeriesReader()

    dicom_names=reader.GetGDCMSeriesFileNames(input_directory)
    reader.SetFileNames(dicom_names)
    # print dicom_names

    image=reader.Execute()
    return image