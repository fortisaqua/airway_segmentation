import os
import shutil
import tensorflow as tf
import scipy.io
import tools
import numpy as np
import time
import test
import SimpleITK as ST
from dicom_read import read_dicoms
import gc

resolution = 64
batch_size = 4
lr_down = [0.001,0.0002,0.0001]
ori_lr = 0.001
power = 0.9
GPU0 = '0'
input_shape = [64,64,128]
output_shape = [64,64,128]
type_num = 0

###############################################################
config={}
config['train_names'] = ['chair']
for name in config['train_names']:
    config['X_train_'+name] = './Data/'+name+'/train_25d/voxel_grids_64/'
    config['Y_train_'+name] = './Data/'+name+'/train_3d/voxel_grids_64/'

config['test_names']=['chair']
for name in config['test_names']:
    config['X_test_'+name] = './Data/'+name+'/test_25d/voxel_grids_64/'
    config['Y_test_'+name] = './Data/'+name+'/test_3d/voxel_grids_64/'

config['resolution'] = resolution
config['batch_size'] = batch_size
config['meta_path'] = '/opt/analyse_airway/data_meta.pkl'
config['data_size'] = input_shape

################################################################

class Network:
    def __init__(self):
        self.train_models_dir = './airway_model/'
        # self.train_sum_dir = './train_sum/'
        # self.test_results_dir = './test_results/'
        # self.test_sum_dir = './test_sum/'

    def ae_u(self,X,training,batch_size,threshold):
        original=16
        growth=10
        dense_layer_num=12
        # input layer
        X=tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
        # image reduce layer
        conv_input=tools.Ops.conv3d(X,k=3,out_c=original,str=2,name='conv_input')
        with tf.device('/gpu:'+GPU0):
            ##### dense block 1
            c_e = []
            s_e = []
            layers_e=[]
            layers_e.append(conv_input)
            for i in range(dense_layer_num):
                c_e.append(original+growth*(i+1))
                s_e.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_e[-1], 'bn_dense_1_' + str(j), training=training)
                layer = tools.Ops.xxlu(layer, name='relu')
                layer = tools.Ops.conv3d(layer,k=3,out_c=growth,str=s_e[j],name='dense_1_'+str(j))
                next_input = tf.concat([layer,layers_e[-1]],axis=4)
                layers_e.append(next_input)

        # middle down sample
            mid_layer = tools.Ops.batch_norm(layers_e[-1], 'bn_mid', training=training)
            mid_layer = tools.Ops.xxlu(mid_layer,name='relu')
            mid_layer = tools.Ops.conv3d(mid_layer,k=1,out_c=original+growth*dense_layer_num,str=1,name='mid_conv')
            mid_layer_down = tools.Ops.maxpool3d(mid_layer,k=2,s=2,pad='SAME')

        ##### dense block
        with tf.device('/gpu:'+GPU0):
            c_d = []
            s_d = []
            layers_d = []
            layers_d.append(mid_layer_down)
            for i in range(dense_layer_num):
                c_d.append(original+growth*(dense_layer_num+i+1))
                s_d.append(1)
            for j in range(dense_layer_num):
                layer = tools.Ops.batch_norm(layers_d[-1],'bn_dense_2_'+str(j),training=training)
                layer = tools.Ops.xxlu(layer, name='relu')
                layer = tools.Ops.conv3d(layer,k=3,out_c=growth,str=s_d[j],name='dense_2_'+str(j))
                next_input = tf.concat([layer,layers_d[-1]],axis=4)
                layers_d.append(next_input)

            ##### final up-sampling
            bn_1 = tools.Ops.batch_norm(layers_d[-1],'bn_after_dense',training=training)
            relu_1 = tools.Ops.xxlu(bn_1 ,name='relu')
            conv_27 = tools.Ops.conv3d(relu_1,k=1,out_c=original+growth*dense_layer_num*2,str=1,name='conv_up_sample_1')
            deconv_1 = tools.Ops.deconv3d(conv_27,k=2,out_c=128,str=2,name='deconv_up_sample_1')
            concat_up = tf.concat([deconv_1,mid_layer],axis=4)
            deconv_2 = tools.Ops.deconv3d(concat_up,k=2,out_c=64,str=2,name='deconv_up_sample_2')

            predict_map = tools.Ops.conv3d(deconv_2,k=1,out_c=1,str=1,name='predict_map')

            vox_no_sig = predict_map
            # vox_no_sig = tools.Ops.xxlu(vox_no_sig,name='relu')
            vox_sig = tf.sigmoid(predict_map)
            vox_sig_modified = tf.maximum(vox_sig-threshold,0.01)
        return vox_sig, vox_sig_modified,vox_no_sig

    def dis(self, X, Y,training):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
            Y = tf.reshape(Y,[batch_size,output_shape[0],output_shape[1],output_shape[2],1])
            layer = tf.concat([X,Y],axis=4)
            c_d = [1,2,64,128,256,512]
            s_d = [0,2,2,2,2,2]
            layers_d =[]
            layers_d.append(layer)
            for i in range(1,6,1):
                layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d_1'+str(i))
                if i!=5:
                    layer = tools.Ops.xxlu(layer, name='lrelu')
                    # batch normal layer
                    layer = tools.Ops.batch_norm(layer, 'bn_up' + str(i), training=training)
                layers_d.append(layer)
            y = tf.reshape(layers_d[-1],[batch_size,-1])
            # for j in range(len(layers_d)-1):
            #     y = tf.concat([y,tf.reshape(layers_d[j],[batch_size,-1])],axis=1)
        return tf.nn.sigmoid(y)

    def test(self,dicom_dir):
        # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
        g_airway = tf.Graph()
        with g_airway.as_default():
            test_input_shape = input_shape
            test_batch_size = batch_size
            threshold = tf.placeholder(tf.float32)
            training = tf.placeholder(tf.bool)
            X = tf.placeholder(shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
                               dtype=tf.float32)
            with tf.variable_scope('ae',reuse=False):
                Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size, threshold)

            # print tools.Ops.variable_count()
            sum_merged = tf.summary.merge_all()
            saver = tf.train.Saver(max_to_keep=1)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.visible_device_list = GPU0
            with tf.Session(config=config) as sess:
                if os.path.exists(self.train_models_dir):
                    saver.restore(sess, self.train_models_dir + 'model.cptk')
                # sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, sess.graph)
                # sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

                if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                    print "restoring saved model"
                    saver.restore(sess, self.train_models_dir + 'model.cptk')
                else:
                    sess.run(tf.global_variables_initializer())
                test_data = tools.Test_data(dicom_dir, input_shape,'vtk_data')
                test_data.organize_blocks()
                block_numbers = test_data.blocks.keys()
                for i in range(0, len(block_numbers), test_batch_size):
                    batch_numbers = []
                    if i + test_batch_size < len(block_numbers):
                        temp_input = np.zeros(
                            [test_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                        for j in range(test_batch_size):
                            temp_num = block_numbers[i + j]
                            temp_block = test_data.blocks[temp_num]
                            batch_numbers.append(temp_num)
                            block_array = temp_block.load_data()
                            block_shape = np.shape(block_array)
                            temp_input[j, 0:block_shape[0], 0:block_shape[1], 0:block_shape[2]] += block_array
                        Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run([Y_pred, Y_pred_modi, Y_pred_nosig],
                                                                               feed_dict={X: temp_input,
                                                                                          training: False,
                                                                                          threshold: 0.8})
                        for j in range(test_batch_size):
                            test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
                    else:
                        temp_batch_size = len(block_numbers) - i
                        temp_input = np.zeros(
                            [temp_batch_size, input_shape[0], input_shape[1], input_shape[2]])
                        for j in range(temp_batch_size):
                            temp_num = block_numbers[i + j]
                            temp_block = test_data.blocks[temp_num]
                            batch_numbers.append(temp_num)
                            block_array = temp_block.load_data()
                            block_shape = np.shape(block_array)
                            temp_input[j, 0:block_shape[0], 0:block_shape[1], 0:block_shape[2]] += block_array
                        X_temp = tf.placeholder(
                            shape=[temp_batch_size, input_shape[0], input_shape[1], input_shape[2]],
                            dtype=tf.float32)
                        with tf.variable_scope('ae', reuse=True):
                            Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp = self.ae_u(X_temp, training,
                                                                                         temp_batch_size, threshold)
                        Y_temp_pred, Y_temp_modi, Y_temp_pred_nosig = sess.run(
                            [Y_pred_temp, Y_pred_modi_temp, Y_pred_nosig_temp],
                            feed_dict={X_temp: temp_input,
                                       training: False,
                                       threshold: 0.8})
                        for j in range(temp_batch_size):
                            test_data.upload_result(batch_numbers[j], Y_temp_modi[j, :, :, :])
                test_result_array = test_data.get_result()
                # print "result shape: ", np.shape(test_result_array)
                r_s = np.shape(test_result_array)  # result shape
                e_t = 10  # edge thickness
                to_be_transformed = np.zeros(r_s, np.float32)
                to_be_transformed[e_t:r_s[0] - e_t, e_t:r_s[1] - e_t, 0:r_s[2] - e_t] += test_result_array[
                                                                                           e_t:r_s[0] - e_t,
                                                                                           e_t:r_s[1] - e_t,
                                                                                           0:r_s[2] - e_t]
                # print np.max(to_be_transformed)
                # print np.min(to_be_transformed)
                final_img = ST.GetImageFromArray(np.transpose(to_be_transformed, [2, 1, 0]))
                final_img.SetSpacing(test_data.space)
                return final_img

def airway_seg(lung_img):
    time1 = time.time()
    net = Network()
    airway_mask = net.test(lung_img)
    time2 = time.time()
    del net
    gc.collect()
    print "Writing airway mask"
    ST.WriteImage(airway_mask,'./output/airway_mask.vtk')
    print "total time cost of airway segmentation: ",str(time2-time1),'s'
    return airway_mask