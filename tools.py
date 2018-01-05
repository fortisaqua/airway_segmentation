import numpy as np
import os
import re
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import random
import organize_data
from dicom_read import read_dicoms
import SimpleITK as ST

class Data_block:
    # single input data block
    def __init__(self,ranger,data_array):
        self.ranger=ranger
        self.data_array=data_array

    def get_range(self):
        return self.ranger

    def load_data(self):
        return self.data_array

class Test_data():
    # load data and translate to original array
    def __init__(self,data,block_shape,type):
        if type == 'dicom_data':
            self.img = read_dicoms(data)
        elif type == 'vtk_data':
            self.img = data
        self.space = self.img.GetSpacing()
        self.image_array = ST.GetArrayFromImage(self.img)
        self.image_array = np.transpose(self.image_array,[2,1,0])
        self.image_shape = np.shape(self.image_array)
        self.block_shape=block_shape
        self.blocks=dict()
        self.results=dict()

    # do the simple threshold function
    def threshold(self,low,high):
        mask_array=np.float32(np.float32(self.image_array<=high)*np.float32(self.image_array>=low))
        return np.float32(np.float32(self.image_array<=high)*np.float32(self.image_array>=low))

    def organize_blocks(self):
        block_num=0
        original_shape=np.shape(self.image_array)
        threshed_array = self.image_array*np.float32(self.image_array<=0)
        print 'data shape: ', original_shape
        for i in range(0,original_shape[0],self.block_shape[0]/2):
            for j in range(0,original_shape[1],self.block_shape[1]/2):
                for k in range(0,original_shape[2],self.block_shape[2]/2):
                    if i<original_shape[0] and j<original_shape[1] and k<original_shape[2]:
                        block_array = threshed_array[i:i+self.block_shape[0],j:j+self.block_shape[1],k:k+self.block_shape[2]]
                        block_shape = np.shape(block_array)
                        ranger=[i,i+block_shape[0],j,j+block_shape[1],k,k+block_shape[2]]
                        this_block=Data_block(ranger,threshed_array[i:i+self.block_shape[0],j:j+self.block_shape[1],k:k+self.block_shape[2]])
                        self.blocks[block_num]=this_block
                        block_num+=1

    def upload_result(self,block_num,result_array):
        ranger = self.blocks[block_num].get_range()
        partial_result = np.float32((result_array-0.01)>0)
        this_result = Data_block(ranger,partial_result)
        self.results[block_num]=this_result

    def get_result(self):
        ret=np.zeros(self.image_shape,np.float32)
        for number in self.results.keys():
            try:
                ranger=self.results[number].get_range()
                xmin=ranger[0]
                xmax=ranger[1]
                ymin=ranger[2]
                ymax=ranger[3]
                zmin=ranger[4]
                zmax=ranger[5]
                temp_result = self.results[number].load_data()[:,:,:,0]
                # temp_shape = np.shape(temp_result)
                ret[xmin:xmax,ymin:ymax,zmin:zmax]+=temp_result[:xmax-xmin,:ymax-ymin,:zmax-zmin]
            except Exception,e:
                print np.shape(self.results[number].load_data()[:,:,:,0]),self.results[number].get_range()
        return np.float32(ret>=2)

class Data:
    def __init__(self,config,epoch):
        self.config = config
        self.train_batch_index = 0
        self.test_seq_index = 0
        self.epoch = epoch
        self.resolution = config['resolution']
        self.batch_size = config['batch_size']
        
        self.train_names = config['train_names']
        self.test_names = config['test_names']
        self.data_size = config['data_size']
        # self.X_train_files, self.Y_train_files = self.load_X_Y_files_paths_all( self.train_names,label='train')
        # self.X_test_files, self.Y_test_files = self.load_X_Y_files_paths_all(self.test_names,label='test')
        # print "X_train_files:",len(self.X_train_files)
        # print "X_test_files:",len(self.X_test_files)

        self.train_numbers,self.test_numbers = self.load_X_Y_numbers_special(config['meta_path'],self.epoch)

        # self.total_train_batch_num = int(len(self.X_train_files) // self.batch_size) -1
        # self.total_test_seq_batch = int(len(self.X_test_files) // self.batch_size) -1
        print "train_numbers:",len(self.train_numbers),"---",self.train_numbers
        print "test_numbers:",len(self.test_numbers),"---",self.test_numbers
        self.total_train_batch_num,self.train_locs = self.load_X_Y_train_batch_num()
        self.total_test_seq_batch,self.test_locs = self.load_X_Y_test_batch_num()
        print "total_train_batch_num: ", self.total_train_batch_num
        print "total_test_seq_batch: ",self.total_test_seq_batch
        # self.check_data()
        self.shuffle_X_Y_pairs()
        # testing code
        # for i in range(0,3):
        #     X_train_voxels,Y_train_voxels=self.load_X_Y_voxel_train_next_batch()
        # X_test_voxels,Y_test_voxels=self.load_X_Y_voxel_test_next_batch()
        # print 123


    @staticmethod
    def plotFromVoxels(voxels,original):
        if len(voxels.shape)>3:
            x_d = voxels.shape[0]
            y_d = voxels.shape[1]
            z_d = voxels.shape[2]
            v = voxels[:,:,:,0]
            v = np.reshape(v,(x_d,y_d,z_d))
        else:
            v = voxels
        x, y, z = v.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        print "generated :",str(len(x))

        if len(original.shape)>3:
            x_d = original.shape[0]
            y_d = original.shape[1]
            z_d = original.shape[2]
            v_ori = original[:,:,:,0]
            v_ori = np.reshape(v_ori,(x_d,y_d,z_d))
        else:
            v_ori = original
        x, y, z = v_ori.nonzero()
        fig = plt.figure()
        ax_ori = fig.add_subplot(111, projection='3d')
        ax_ori.scatter(x, y, z, zdir='z', c='red')
        print "orign :", str(len(x))

        plt.show()

    def load_X_Y_files_paths_all(self, obj_names, label='train'):
        x_str=''
        y_str=''
        if label =='train':
            x_str='X_train_'
            y_str ='Y_train_'

        elif label == 'test':
            x_str = 'X_test_'
            y_str = 'Y_test_'

        else:
            print "label error!!"
            exit()

        X_data_files_all = []
        Y_data_files_all = []
        for name in obj_names:
            X_folder = self.config[x_str + name]
            Y_folder = self.config[y_str + name]
            X_data_files, Y_data_files = self.load_X_Y_files_paths(X_folder, Y_folder)

            for X_f, Y_f in zip(X_data_files, Y_data_files):
                if X_f[0:15] != Y_f[0:15]:
                    print "index inconsistent!!\n"
                    exit()
                X_data_files_all.append(X_folder + X_f)
                Y_data_files_all.append(Y_folder + Y_f)
        return X_data_files_all, Y_data_files_all

    def load_X_Y_files_paths(self,X_folder, Y_folder):
        X_data_files = [X_f for X_f in sorted(os.listdir(X_folder))]
        Y_data_files = [Y_f for Y_f in sorted(os.listdir(Y_folder))]

        return X_data_files, Y_data_files

    def voxel_grid_padding(self,a):
        x_d = a.shape[0]
        y_d = a.shape[1]
        z_d = a.shape[2]
        channel = a.shape[3]
        resolution = self.resolution
        size = [resolution, resolution, resolution,channel]
        b = np.zeros(size)

        bx_s = 0;bx_e = size[0];by_s = 0;by_e = size[1];bz_s = 0; bz_e = size[2]
        ax_s = 0;ax_e = x_d;ay_s = 0;ay_e = y_d;az_s = 0;az_e = z_d
        if x_d > size[0]:
            ax_s = int((x_d - size[0]) / 2)
            ax_e = int((x_d - size[0]) / 2) + size[0]
        else:
            bx_s = int((size[0] - x_d) / 2)
            bx_e = int((size[0] - x_d) / 2) + x_d

        if y_d > size[1]:
            ay_s = int((y_d - size[1]) / 2)
            ay_e = int((y_d - size[1]) / 2) + size[1]
        else:
            by_s = int((size[1] - y_d) / 2)
            by_e = int((size[1] - y_d) / 2) + y_d

        if z_d > size[2]:
            az_s = int((z_d - size[2]) / 2)
            az_e = int((z_d - size[2]) / 2) + size[2]
        else:
            bz_s = int((size[2] - z_d) / 2)
            bz_e = int((size[2] - z_d) / 2) + z_d
        b[bx_s:bx_e, by_s:by_e, bz_s:bz_e,:] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e, :]

        return b

    def load_single_voxel_grid(self,path):
        temp = re.split('_', path.split('.')[-2])
        x_d = int(temp[len(temp) - 3])
        y_d = int(temp[len(temp) - 2])
        z_d = int(temp[len(temp) - 1])

        a = np.loadtxt(path)
        if len(a)<=0:
            print " load_single_voxel_grid error: ", path
            exit()

        voxel_grid = np.zeros((x_d, y_d, z_d,1))
        for i in a:
            voxel_grid[int(i[0]), int(i[1]), int(i[2]),0] = 1 # occupied

        #Data.plotFromVoxels(voxel_grid)
        voxel_grid = self.voxel_grid_padding(voxel_grid)
        return voxel_grid

    def load_X_Y_voxel_grids(self,X_data_files, Y_data_files):
        if len(X_data_files) !=self.batch_size or len(Y_data_files)!=self.batch_size:
            print "load_X_Y_voxel_grids error:", X_data_files, Y_data_files
            exit()

        X_voxel_grids = []
        Y_voxel_grids = []
        index = -1
        for X_f, Y_f in zip(X_data_files, Y_data_files):
            index += 1
            X_voxel_grid = self.load_single_voxel_grid(X_f)
            X_voxel_grids.append(X_voxel_grid)

            Y_voxel_grid = self.load_single_voxel_grid(Y_f)
            Y_voxel_grids.append(Y_voxel_grid)

        X_voxel_grids = np.asarray(X_voxel_grids)
        Y_voxel_grids = np.asarray(Y_voxel_grids)
        return X_voxel_grids, Y_voxel_grids

    def load_X_Y_numbers_special(self,meta_path,epoch):
        self.dicom_origin,self.mask = organize_data.get_organized_data(meta_path,self.data_size,epoch)
        numbers=[]
        train_numbers=[]
        test_numbers=[]
        for number in self.mask.keys():
            if len(self.mask[number])>0:
                numbers.append(number)
        for i in range(1):
            test_numbers.append(numbers[random.randint(0,len(numbers)-1)])
        for number in numbers:
            if not number in test_numbers:
                train_numbers.append(number)
        return train_numbers,test_numbers

    def load_X_Y_train_batch_num(self):
        total_num=0
        locs=[]
        for number in self.train_numbers:
            for i in range(len(self.mask[number])):
                total_num=total_num+1
                locs.append([number,i])
        return int(total_num/self.batch_size),locs

    def load_X_Y_test_batch_num(self):
        total_num = 0
        locs=[]
        for number in self.test_numbers:
            for i in range(len(self.mask[number])):
                total_num = total_num + 1
                locs.append([number,i])
        return int(total_num / self.batch_size),locs

    def shuffle_X_Y_files(self, label='train'):
        X_new = []; Y_new = []
        if label == 'train':
            X = self.X_train_files; Y = self.Y_train_files
            self.train_batch_index = 0
            index = range(len(X))
            shuffle(index)
            for i in index:
                X_new.append(X[i])
                Y_new.append(Y[i])
            self.X_train_files = X_new
            self.Y_train_files = Y_new

        elif label == 'test':
            X = self.X_test_files; Y = self.Y_test_files
            self.test_seq_index = 0
            index = range(len(X))
            shuffle(index)
            for i in index:
                X_new.append(X[i])
                Y_new.append(Y[i])
            self.X_test_files = X_new
            self.Y_test_files = Y_new

        else:
            print "shuffle_X_Y_files error!\n"
            exit()

    def shuffle_X_Y_pairs(self):
        train_locs_new=[]
        test_locs_new=[]
        trains=self.train_locs
        tests=self.test_locs
        self.train_batch_index = 0
        train_index = range(len(trains))
        test_index = range(len(tests))
        shuffle(train_index)
        shuffle(test_index)
        for i in train_index:
            train_locs_new.append(trains[i])
        for j in test_index:
            test_locs_new.append(tests[j])
        self.train_locs=train_locs_new
        self.test_locs=test_locs_new

    ###################### voxel grids
    def load_X_Y_voxel_grids_train_next_batch(self):
        X_data_files = self.X_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        Y_data_files = self.Y_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        self.train_batch_index += 1
        # self.train_batch_index=0

        X_voxel_grids, Y_voxel_grids = self.load_X_Y_voxel_grids(X_data_files, Y_data_files)
        return X_voxel_grids, Y_voxel_grids

    def load_X_Y_voxel_train_next_batch(self):
        temp_locs=self.train_locs[self.batch_size*self.train_batch_index:self.batch_size*(self.train_batch_index+1)]
        X_data_voxels=[]
        Y_data_voxels=[]
        for pair in temp_locs:
            X_data_voxels.append(self.dicom_origin[pair[0]][pair[1]])
            Y_data_voxels.append(self.mask[pair[0]][pair[1]])
        self.train_batch_index += 1
        X_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        Y_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        '''
        X_voxel_grids = np.asarray(X_voxel_grids)
        Y_voxel_grids = np.asarray(Y_voxel_grids)
        X_data_voxels=np.asarray(X_data_voxels)
        Y_data_voxels=np.asarray(Y_data_voxels)
        '''
        for i in range(len(X_data_voxels)):
            temp_X = X_data_voxels[i][:,:,:]
            temp_y = Y_data_voxels[i][:,:,:]
            shape_X = np.shape(temp_X)
            shape_Y = np.shape(temp_y)
            X_data[i,:shape_X[0],:shape_X[1],:shape_X[2]] = X_data_voxels[i][:,:,:]
            Y_data[i,:shape_Y[0],:shape_Y[1],:shape_Y[2]] = Y_data_voxels[i][:,:,:]

        return X_data,Y_data

    def load_X_Y_voxel_grids_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(45)
        idx = random.sample(range(len(self.X_test_files)), self.batch_size)
        X_test_files_batch = []
        Y_test_files_batch = []
        for i in idx:
            X_test_files_batch.append(self.X_test_files[i])
            Y_test_files_batch.append(self.Y_test_files[i])

        X_test_batch, Y_test_batch = self.load_X_Y_voxel_grids(X_test_files_batch, Y_test_files_batch)
        return X_test_batch, Y_test_batch

    def load_X_Y_voxel_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(45)
        idx = random.sample(range(len(self.test_locs)), self.batch_size)
        X_test_voxels_batch=[]
        Y_test_voxels_batch=[]
        for i in idx:
            temp_pair=self.test_locs[i]
            X_test_voxels_batch.append(self.dicom_origin[temp_pair[0]][temp_pair[1]])
            Y_test_voxels_batch.append(self.mask[temp_pair[0]][temp_pair[1]])
        X_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        Y_data = np.zeros([self.batch_size,self.data_size[0],self.data_size[1],self.data_size[2]],np.float32)
        '''
        X_test_voxels_batch=np.asarray(X_test_voxels_batch)
        Y_test_voxels_batch=np.asarray(Y_test_voxels_batch)
        '''
        for i in range(len(X_test_voxels_batch)):
            temp_X = X_test_voxels_batch[i][:,:,:]
            temp_y = Y_test_voxels_batch[i][:,:,:]
            shape_X = np.shape(temp_X)
            shape_Y = np.shape(temp_y)
            X_data[i,:shape_X[0],:shape_X[1],:shape_X[2]] = X_test_voxels_batch[i][:,:,:]
            Y_data[i,:shape_Y[0],:shape_Y[1],:shape_Y[2]] = Y_test_voxels_batch[i][:,:,:]
        return X_data,Y_data

    ###################  check datas
    def check_data(self):
        fail_list=[]
        tag=True
        for pair in self.train_locs:
            shape1 = np.shape(self.dicom_origin[pair[0]][pair[1]])
            shape2 = np.shape(self.mask[pair[0]][pair[1]])
            if shape1[0]==shape2[0]==self.data_size[0] and shape1[1]==shape2[1]==self.data_size[1] and shape1[2]==shape2[2]==self.data_size[2]:
                tag=True
            else:
                tag=False
                fail_list.append(pair)
        for pair in self.test_locs:
            shape1 = np.shape(self.dicom_origin[pair[0]][pair[1]])
            shape2 = np.shape(self.mask[pair[0]][pair[1]])
            if shape1[0]==shape2[0]==self.data_size[0] and shape1[1]==shape2[1]==self.data_size[1] and shape1[2]==shape2[2]==self.data_size[2]:
                tag=True
            else:
                tag=False
                fail_list.append(pair)
                print shape1
                print shape2
                print "=============================================="
        if tag:
            print "checked!"
        else:
            print "some are failed"
            for item in fail_list:
                print item

class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,name='relu'):
        if name =='relu':
            return  Ops.relu(x)
        if name =='lrelu':
            return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            try:
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
            except Exception,e:
                print e

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[4]
        w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)

        stride = [1, str, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        bat, in_d1, in_d2, in_d3, in_c = [int(d) for d in x.get_shape()]
        w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
        stride = [1, str, str, str, 1]
        y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
        y = tf.nn.bias_add(y, b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999):
        '''Assume 2d [batch, values] tensor'''

        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[-1]
            x_shape = x.get_shape()
            axis = list(range(len(x_shape) - 1))
            scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
            offset = tf.get_variable('offset', [size])

            pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer, trainable=False)
            pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer, trainable=False)
            batch_mean, batch_var = tf.nn.moments(x, axis)

            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

            return tf.cond(training, batch_statistics, population_statistics)