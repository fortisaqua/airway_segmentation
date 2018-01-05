import sys
import os
import numpy as np
import SimpleITK as ST
import dicom_read
import cPickle as pickle
import scipy.io as sio

def get_range(mask,type=0):
    begin_switch = 0
    begin = 0
    if type==0:
        end_switch = 0
        end = np.shape(mask)[2] - 1
        for i in range(np.shape(mask)[2]):
            if np.max(mask[:, :, i]) == 1 and begin_switch == 0:
                begin_switch = 1
                begin = i
            if np.max(mask[:, :, i]) == 0 and end_switch == 0 and begin_switch == 1:
                end = i
                end_switch = 1
            if end_switch:
                break
        return begin,end
    if type==1:
        end_switch = 0
        end = np.shape(mask)[1] - 1
        for i in range(np.shape(mask)[1]):
            if np.max(mask[:, i, :]) == 1 and begin_switch == 0:
                begin_switch = 1
                begin = i
            if np.max(mask[:, i, :]) == 0 and end_switch == 0 and begin_switch == 1:
                end = i
                end_switch = 1
            if end_switch:
                break
        return begin, end
    if type==2:
        end_switch = 0
        end = np.shape(mask)[0] - 1
        for i in range(np.shape(mask)[0]):
            if np.max(mask[i, :, :]) == 1 and begin_switch == 0:
                begin_switch = 1
                begin = i
            if np.max(mask[i, :, :]) == 0 and end_switch == 0 and begin_switch == 1:
                end = i
                end_switch = 1
            if end_switch:
                break
        return begin, end

def get_range_slices(mask,type):
    mask_shape=np.shape(mask)
    ret_begin=0
    ret_end=0
    max_length = 0
    if type==1:
        ret_begin=0
        ret_end=mask_shape[1]-1
        for i in range(mask_shape[2]):
            temp_slice=mask[:,:,i]
            begin_switch = 0
            end_switch = 0
            begin = 0
            for j in range(np.shape(temp_slice)[1]):
                if np.max(temp_slice[:, j]) == 1 and begin_switch == 0:
                    begin_switch = 1
                    begin = j
                if np.max(temp_slice[:,j]) == 0 and end_switch == 0 and begin_switch == 1:
                    end_switch = 1
                    end = j
                if end_switch:
                    break
            if (end-begin)>max_length:
                # print begin,end
                ret_begin = begin
                ret_end = end
    if type==2:
        ret_begin=0
        ret_end=mask_shape[0]-1
        for i in range(mask_shape[2]):
            temp_slice=mask[:,:,i]
            begin_switch = 0
            end_switch = 0
            begin = 0
            for j in range(np.shape(temp_slice)[1]):
                if np.max(temp_slice[j, :]) == 1 and begin_switch == 0:
                    begin_switch = 1
                    begin = j
                if np.max(temp_slice[j, :]) == 0 and end_switch == 0 and begin_switch == 1:
                    end_switch = 1
                    end = j
                if end_switch:
                    break
            if (end-begin)>max_length:
                # print begin,end
                ret_begin = begin
                ret_end = end
    return ret_begin,ret_end

def get_array(dicom_dir):
    img = dicom_read.read_dicoms(dicom_dir)
    ret_array = ST.GetArrayFromImage(img)
    ret_array = np.transpose(ret_array,[2,1,0])
    return  ret_array

def get_original_arrays(root_path,type):
    number = 0
    origin_datas = dict()
    for patient_dir in os.listdir(root_path):
        # read folder names
        origin_datas[number] = dict()
        origin_datas[number]['name'] = patient_dir
        dicom_dirs = root_path+'/'+patient_dir
        for sub_dir in os.listdir(dicom_dirs):
            dicom_dir = dicom_dirs+'/'+sub_dir
            if os.path.isdir(dicom_dir) and ('origin' in dicom_dir or type in dicom_dir):
                origin_datas[number][sub_dir]=get_array(dicom_dirs+'/'+sub_dir)
        number+=1
        origin_datas['mask_type']=type
    return origin_datas

def organize_data_pairs(origin_datas):
    ret_pairs=dict()
    number=0
    mask_type = origin_datas['mask_type']
    for data_num in origin_datas.keys():
        if not 'mask_type' == data_num:
            # Read each set of data
            print 'processing data: ',origin_datas[data_num]['name']
            # Each data has only one mask array and will be used to check if shapes are identical
            mask_array = origin_datas[data_num][mask_type]
            mask_shape = np.shape(mask_array)
            for data_name in origin_datas[data_num]:
                # check if this original data array has the same shape with the mask array
                if 'original' in data_name:
                    original_array = origin_datas[data_num][data_name]
                    original_shape = np.shape(original_array)
                    if mask_shape[0]==original_shape[0] and mask_shape[1]==original_shape[1] and mask_shape[2]==original_shape[2]:
                        ret_pairs[number]=dict()
                        ret_pairs[number]['original']=original_array
                        ret_pairs[number]['mask']=mask_array
                        ret_pairs[number]['name']=origin_datas[data_num]['name']
                        number+=1
    return ret_pairs

def get_range_each(data_pair):
    # print data_pair.keys()
    mask_array = data_pair['mask']
    ret=[]
    ending = 0
    z_length=90
    mask_shape = np.shape(mask_array)
    for j in range(mask_shape[2]):
        if np.sum(mask_array[:, :, mask_shape[2] - j - 1]) > 0:
            ending = mask_shape[2] - j - 1
            # print np.sum(mask_array[:,:,mask_shape[2]-j-1])
            break
    for i in range(3):
        if i==0:
            begin, end = get_range(mask_array[:, :, ending - z_length:ending], i)
            ret.append(begin+ending-z_length)
            ret.append(end+ending-z_length)
        else:
            begin, end = get_range_slices(mask_array[:, :, ending - z_length:ending],i)
            ret.append(begin)
            ret.append(end)
    return ret

def get_proper_range(data_pairs):
    maxs=[0,0,0]
    for number in data_pairs.keys():
        ranger=get_range_each(data_pairs[number])
        size=[]
        for i in range(3):
            size.append(ranger[i*2+1]-ranger[i*2])
        for i in range(3):
            if size[i]>maxs[i]:
                maxs[i]=size[i]
        print ranger,'---',size,"---",data_pairs[number]['name']
    print 'propel size of sliding window: ',maxs
    return maxs

def out_put_data_pairs(data_pairs):
    root_dir = '/opt/analyse_airway/'
    if not os.path.exists('./output'):
        os.makedirs('./output')
    data_meta = dict()
    for number in data_pairs:
        sio.savemat('./output/data'+str(number)+'.mat',{'original':data_pairs[number]['original'],
                                             'mask':data_pairs[number]['mask']})
        data_meta[number]=root_dir+'output/data'+str(number)+'.mat'
    pickle_writer = open('./data_meta.pkl','wb')
    pickle.dump(data_meta,pickle_writer)
    pickle_writer.close()