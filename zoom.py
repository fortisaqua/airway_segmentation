# import cv
import math
import numpy as np
import scipy.io as sio
import time
import cPickle as pickle

# def JZoom(image, m, n):
#     H = int(image.height * m - m)
#     W = int(image.width * n - n)
#     size = (W, H)
#     iZoom = cv.CreateImage(size, image.depth, image.nChannels)
#     sum = [0, 0, 0]
#     for i in range(H):
#         for j in range(W):
#             x1 = int(math.floor((i + 1) / m - 1))
#             y1 = int(math.floor((j + 1) / n - 1))
#             p = (i + 0.0) / m - x1
#             q = (j + 0.0) / n - y1
#             for k in range(3):
#                 sum[k] = int(
#                     image[x1, y1][k] * (1 - p) * (1 - q) + image[x1 + 1, y1][k] * p * (1 - q) + image[x1, y1 + 1][k] * (
#                     1 - p) * q + image[x1 + 1, y1 + 1][k] * p * q)
#             iZoom[i, j] = (sum[0], sum[1], sum[2])
#     return iZoom

def Array_Zoom_in(image, m, n):
    shape=np.shape(image)
    H = int(shape[0] * m - m)
    W = int(shape[1] * n - n)
    iZoom = np.zeros((H,W,shape[2]),dtype=np.float32)
    for i in range(H):
        for j in range(W):
            x1 = int(math.floor((i + 1) / m - 1))
            y1 = int(math.floor((j + 1) / n - 1))
            p = (i + 0.0) / m - x1
            q = (j + 0.0) / n - y1
            for k in range(shape[2]):
                sum= int(
                    image[x1, y1, k] * (1 - p) * (1 - q) + image[x1 + 1, y1, k] * p * (1 - q) +
                    image[x1, y1 + 1, k] * (1 - p) * q + image[x1 + 1, y1 + 1,k] * p * q)
                iZoom[i, j, k] = sum
    return iZoom

def Array_Reduce(image,m,n):
    shape=np.shape(image)
    H = int(shape[0] * m)
    W = int(shape[1] * n)
    iJReduce = np.zeros((H, W, shape[2]), dtype=np.float32)
    for c in range(shape[2]):
        for i in range(H):
            for j in range(W):
                x1 = int(i/m)
                x2 = int((i+1)/m)
                y1 = int(j/n)
                y2 = int((j+1)/n)
                sum = 0
                for k in range(x1,x2):
                    for l in range(y1,y2):
                        sum = sum+image[k , l, c]
                num = (x2-x1)*(y2-y1)
                iJReduce[i , j, c] = sum/num
    return iJReduce

'''
returns an array [xmin,xmax,ymin,ymax]
'''
def get_valid_area(input_array):
    array_shape=np.shape(input_array)
    central_point=[(array_shape[0]-1)/2,(array_shape[1]-1)/2]
    xmin=0
    xmax=array_shape[0]-1
    ymin=0
    ymax=array_shape[1]-1
    tags=[0,0,0,0]
    for i in range(array_shape[1]/2):
        if np.max(input_array[central_point[0]-i,:,:])>0 and tags[0]==0:
            xmin=central_point[0]-i
        else:
            tags[0]=1
        if np.max(input_array[central_point[0]+i,:,:])>0 and tags[1]==0:
            xmax=central_point[0]+i
        else:
            tags[1]=1
    for j in range(array_shape[1]/2):
        if np.max(input_array[:,central_point[1]-j,:])>0 and tags[2]==0:
            ymin=central_point[1]-j
        else:
            tags[2]=1
        if np.max(input_array[:,central_point[1]+j,:])>0 and tags[3]==0:
            ymax=central_point[1]+j
        else:
            tags[3]=1
    return [xmin,xmax,ymin,ymax]


# image = cv.LoadImage('lena.jpg', 1)
# iZoom1 = JZoom(image, 2, 3)
# iZoom2 = JZoom(image, 2.5, 2.5)
# cv.ShowImage('image', image)
# cv.ShowImage('iZoom1', iZoom1)
# cv.ShowImage('iZoom2', iZoom2)

# img_array = np.array(iZoom2[:,:])
# print np.shape(img_array)
# cv.ShowImage('iZoom2', cv.fromarray(img_array[:,:,1]))
# cv.WaitKey(0)
# central = [(array_shape[0] - 1) / 2, (array_shape[1] - 1) / 2,(array_shape[2] - 1) / 2]
# test_input=np.transpose(test_input,(2,1,0))
# test_img=ST.GetImageFromArray(test_input)
# # NeighborhoodConnected(Image image1, VectorUIntList seedList, double lower=0, double upper=1, VectorUInt32 radius, double replaceValue=1)
# threshed_mask=ST.NeighborhoodConnected(test_img,[[central[0],central[1],central[2]]],-50,np.float64(np.max(test_input)),[1,1,1],1.0)
# threshed_array = ST.GetArrayFromImage(threshed_mask)
#
# img_array = ST.GetArrayFromImage(test_img)*threshed_array
# threshed_img = ST.GetImageFromArray(img_array)
# print img_array.dtype
# ST.WriteImage(threshed_img,'./threshed_img1.vtk')
# img_array = np.transpose(img_array,(2,1,0))
# reduced=Array_Reduce(test_input,0.5,0.5)
# print time.strftime('%Y-%m-%d %H:%M:%S'),np.shape(reduced)
# zoomed = Array_Zoom_in(test_input,1.377,1.377)
# print time.strftime('%Y-%m-%d %H:%M:%S'),np.shape(zoomed)
# print np.max(test_input[ranger[0]-1,:,:])
# print np.max(test_input[ranger[1]+1,:,:])
# print np.max(test_input[:,ranger[2]-1,:])
# print np.max(test_input[:,ranger[3]+1,:])
# test_input = sio.loadmat(
#     '/home/fortis/pycharmProjects/analyse_liver_data/out_put/3Dircadb1.1/PATIENT_DICOM/PATIENT_DICOM.mat')[
#     'original']
# test_mask = sio.loadmat(
#     '/home/fortis/pycharmProjects/analyse_liver_data/out_put/3Dircadb1.1/MASKS_DICOM/liver/liver.mat')[
#     'liver_mask']

def resize_image(test_input,resized_length):
    array_shape = np.shape(test_input)
    # print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',np.shape(test_input)
    ranger=get_valid_area(test_input)
    # print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',ranger

    sliced_img=test_input[ranger[0]:ranger[1]+1,ranger[2]:ranger[3]+1,:]
    size=np.array([ranger[1]-ranger[0]+1,ranger[3]-ranger[2]+1])
    maxsize=np.max(size)
    minsize=np.min(size)

    sliced_size=np.shape(sliced_img)
    processed_img=np.zeros((maxsize,maxsize,array_shape[2]),np.float32)
    padding_size=maxsize-minsize
    try:
        if size[0]>size[1]:
            processed_img[:,padding_size/2:maxsize-padding_size/2,:]=sliced_img[:,:,:]
        else:
            processed_img[padding_size / 2:maxsize - padding_size / 2, :, :] = sliced_img[:, :, :]
    except Exception,e:
        if size[0]>size[1]:
            processed_img[:,padding_size/2+1:maxsize-padding_size/2,:]=sliced_img[:,:,:]
        else:
            processed_img[padding_size / 2+1:maxsize - padding_size / 2, :, :] = sliced_img[:, :, :]

    resized_rate=float(resized_length)/float(maxsize)
    if resized_rate<1:
        resized_img=Array_Reduce(processed_img,resized_rate,resized_rate)
    else:
        resized_img=Array_Zoom_in(processed_img,resized_rate,resized_rate)
    return resized_img,ranger

def resize_mask(test_mask,ranger,resized_length):
    array_shape = np.shape(test_mask)

    sliced_img=test_mask[ranger[0]:ranger[1]+1,ranger[2]:ranger[3]+1,:]
    size=np.array([ranger[1]-ranger[0]+1,ranger[3]-ranger[2]+1])
    maxsize=np.max(size)
    minsize=np.min(size)

    sliced_size=np.shape(sliced_img)
    processed_img=np.zeros((maxsize,maxsize,array_shape[2]),np.float32)
    padding_size=maxsize-minsize
    try:
        if size[0]>size[1]:
            processed_img[:,padding_size/2:maxsize-padding_size/2,:]=sliced_img[:,:,:]
        else:
            processed_img[padding_size / 2:maxsize - padding_size / 2, :, :] = sliced_img[:, :, :]
    except Exception,e:
        if size[0]>size[1]:
            processed_img[:,padding_size/2+1:maxsize-padding_size/2,:]=sliced_img[:,:,:]
        else:
            processed_img[padding_size / 2+1:maxsize - padding_size / 2, :, :] = sliced_img[:, :, :]

    resized_rate=float(resized_length)/float(maxsize)
    if resized_rate<1:
        resized_img=Array_Reduce(processed_img,resized_rate,resized_rate)
    else:
        resized_img=Array_Zoom_in(processed_img,resized_rate,resized_rate)
    return resized_img

# print np.shape(resized_img)
def resize_data(resized_length):
    # resized_length=128
    reader1=open('/opt/analyse_liver_data/filelist.pkl','rb')
    reader2=open('/opt/analyse_liver_data/data_meta.pkl','rb')
    filelist=pickle.load(reader1)
    meta_data=pickle.load(reader2)
    successrul_list=[]
    failed_list=[]
    for number,dataset in meta_data['matrixes'].items():
        try:
            original_data=sio.loadmat(dataset['PATIENT_DICOM'])
            original_img_temp=original_data['original']
            temp_shape = np.shape(original_img_temp)
            liver_data=sio.loadmat(dataset['liver'])
            liver_mask_temp=liver_data['liver_mask']
            if temp_shape[0]<=resized_length and temp_shape[1]<=resized_length:
                # print "probably resized : \n",dataset['PATIENT_DICOM'],'\npassed this time'
                resized_original = original_img_temp
                resized_liver_mask = resized_liver_mask
                sio.savemat(dataset['PATIENT_DICOM'], {'original': original_img_temp,'original_resized':resized_original})
                sio.savemat(dataset['liver'], {'liver_mask': liver_mask_temp,'liver_mask_resized':resized_liver_mask})
                print time.strftime('%Y-%m-%d %H:%M:%S'), '  ', np.shape(resized_original)
                print time.strftime('%Y-%m-%d %H:%M:%S'), '  ', np.shape(resized_liver_mask)
                print "======================================================================================"
                successrul_list.append(dataset['PATIENT_DICOM'])
                successrul_list.append(dataset['liver'])
                continue
            print dataset['PATIENT_DICOM']
            print dataset['liver']
            print time.strftime('%Y-%m-%d %H:%M:%S'), '  ', np.shape(original_img_temp)
            print time.strftime('%Y-%m-%d %H:%M:%S'), '  ', np.shape(liver_mask_temp)
            resized_original,ranger_temp=resize_image(original_img_temp,resized_length)
            print 'valid area: ',ranger_temp
            resized_liver_mask=resize_mask(liver_mask_temp,ranger_temp,resized_length)
            resized_liver_mask=np.int8(resized_liver_mask>0)
            sio.savemat(dataset['PATIENT_DICOM'], {'original': original_img_temp,'original_resized':resized_original})
            sio.savemat(dataset['liver'], {'liver_mask': liver_mask_temp,'liver_mask_resized':resized_liver_mask})
            print time.strftime('%Y-%m-%d %H:%M:%S'), '  ', np.shape(resized_original)
            print time.strftime('%Y-%m-%d %H:%M:%S'), '  ', np.shape(resized_liver_mask)
            print "======================================================================================"
            successrul_list.append(dataset['PATIENT_DICOM'])
            successrul_list.append(dataset['liver'])
        except Exception,e:
            print e
            failed_list.append(dataset['PATIENT_DICOM'])
            failed_list.append(dataset['liver'])

    reader1.close()
    reader2.close()

    file1=open('./successful.txt','wb')
    file2=open('./failed.txt','wb')

    for item in successrul_list:
        file1.write(item+"\n")
    for item in failed_list:
        file2.write(item+"\n")

    file1.close()
    file2.close()

# print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',np.shape(test_input)
# resized,rangerer=resize_image(test_input,resized_length)
# print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',np.shape(resized)
# resized_mask = resize_mask(test_mask,rangerer,resized_length)
# print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',np.shape(resized_mask)

# resize_data(512)
