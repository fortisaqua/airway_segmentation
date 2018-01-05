import os
import shutil
import tensorflow as tf
import scipy.io
import tools
import numpy as np
from dicom_read import read_dicoms
import SimpleITK as ST
import time
import zoom
from zoom import Array_Zoom_in,Array_Reduce

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

def resize_image(test_input,resized_length):
    array_shape = np.shape(test_input)
    # print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',np.shape(test_input)
    # ranger=get_valid_area(test_input)
    # print time.strftime('%Y-%m-%d %H:%M:%S'),'  ',ranger
    ranger=[0,array_shape[0]-1,0,array_shape[1]-1]
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

def get_threshed_img(dicom_dir):
    img=read_dicoms(dicom_dir)
    space = img.GetSpacing()
    image_array = ST.GetArrayFromImage(img)
    # image_array = np.transpose(image_array,(2,1,0))
    print np.shape(image_array)

    array_shape = np.shape(image_array)
    central = [(array_shape[2] - 1) / 2, (array_shape[1] - 1) / 2, (array_shape[0] - 1) / 2]
    print central
    pointslist=[]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i!=0 or j!=0 or k!=0:
                    pointslist.append([central[0]+i,central[1]+j,central[2]+k])
                    pointslist.append([central[0]+i,central[1]+j,central[2]-k])
                    pointslist.append([central[0]+i,central[1]-j,central[2]+k])
                    pointslist.append([central[0]+i,central[1]-j,central[2]-k])
                    pointslist.append([central[0]-i,central[1]+j,central[2]+k])
                    pointslist.append([central[0]-i,central[1]+j,central[2]-k])
                    pointslist.append([central[0]-i,central[1]-j,central[2]+k])
                    pointslist.append([central[0]-i,central[1]-j,central[2]-k])
    threshed_mask = ST.NeighborhoodConnected(img, pointslist, -40,
                                             np.float64(np.max(image_array)), [1, 1, 1], 1.0)
    threshed_mask_array = ST.GetArrayFromImage(threshed_mask)

    threshed_array = image_array * threshed_mask_array
    # threshed_img = ST.GetImageFromArray(threshed_array)

    threshed_array = np.transpose(threshed_array, (2, 1, 0))
    # threshed_array = np.float32(threshed_array)
    # threshed_img = ST.GetImageFromArray(threshed_array)
    # blured_img = ST.CurvatureAnisotropicDiffusion(threshed_img,0.0625,3,1,3)
    # blured_array = ST.GetArrayFromImage(blured_img)
    return threshed_array,space

def get_organized_data(dicom_dir,resized_size):
    half_size = resized_size[2]/2
    time1=time.time()
    origin_array,space = get_threshed_img(dicom_dir)
    time2 = time.time()
    print "time for thresholding: ",time2-time1," s"
    if np.shape(origin_array)[0]==resized_size[0]:
        resized_array=origin_array
    else:
        resized_array,ranger = resize_image(origin_array,resized_size[0])
    # shape = np.shape(origin_array)
    # test_inputs = []
    # for i in range(half_size,shape[2]-half_size,half_size):
    #     test_inputs.append(resized_array[:,:,i-half_size:i+half_size])
    #     print i
    # print len(test_inputs)
    time3 = time.time()
    print "time for resizing: ", time3-time2," s"
    return space,resized_array

# def get_results(output_shape):
#     dicom_dir = "./3Dircadb1.2/PATIENT_DICOM"
#     input_shape = [384,384,4]
#     batch_size = 8
#     GPU0 = '0'
#     train_models_dir = './train_models/'
#     Net = DenseVoxNet.Network()
#     # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
#     X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
#     # Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
#     Y = tf.placeholder(shape=[batch_size, output_shape[0], output_shape[1], output_shape[2]], dtype=tf.float32)
#     training = tf.placeholder(tf.bool)
#     Y_pred, Y_pred_modi,Y_pred_nosig = Net.ae_u(X,training,batch_size)
#     input_datas = get_organized_data(dicom_dir,input_shape)
#     time1 = time.time()
#     results = []
#     saver = tf.train.Saver(max_to_keep=1)
#     config = tf.ConfigProto(allow_soft_placement=True)
#     config.gpu_options.visible_device_list = GPU0
#     with tf.Session(config=config) as sess:
#         print "restoring saved model"
#         saver.restore(sess, train_models_dir + 'model.cptk')
#         if os.path.exists(train_models_dir):
#             saver.restore(sess, train_models_dir + 'model.cptk')
#         for i in range(0,len(input_datas),4):
#             if i+batch_size < len(input_datas)-1:
#                 input_data = np.zeros([batch_size,input_shape[0], input_shape[1], input_shape[2]])
#                 for j in range(i,i+8):
#                     input_data[j-i,:,:,:]=input_datas[j][:,:,:]
#             partial_result = sess.run([Y_pred_modi],feed_dict={X:input_data,training:True})
#             results.append(partial_result)
#     time2 = time.time()
#     print "time for calculating: ",time2-time1," s"
#     print len(results)
#     return results
#
# def test_main():
#     output_shape = [256, 256, 4]
#     results = get_results(output_shape)
#     final_array = np.zeros([output_shape[0],output_shape[1],len(results)*4+4],np.float32)
#     for i in range(len(results)):
#         final_array[:,:,i*4:i*4+8]+=np.float32((results[i][:,:,:]-0.01)>0)
#     final_array = np.int8(final_array>0.5)
#     final_img = ST.GetImageFromArray(final_array)
#     ST.WriteImage(final_img,'./test_result.vtk')

# test_main()
