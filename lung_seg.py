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
input_shape = [512,512,4]
output_shape = [512,512,4]
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
config['meta_path'] = '/opt/analyse_lung/data_meta.pkl'
config['data_size'] = input_shape

################################################################

class Network:
    def __init__(self):
        self.train_models_dir = './lung_model/'

    def ae_u(self,X,training,batch_size,threshold):
        original=16
        growth=12
        dense_layer_num=6
        # input layer
        X=tf.reshape(X,[batch_size,input_shape[0],input_shape[1],input_shape[2],1])
        # image reduce layer
        # conv_input_1=tools.Ops.conv3d(X,k=3,out_c=2,str=2,name='conv_input_down')
        # conv_input_normed=tools.Ops.batch_norm(conv_input_1, 'bn_dense_0_0', training=training)
        # network start
        conv_input_1=tools.Ops.conv3d(X,k=3,out_c=original,str=1,name='conv_input_1')
        conv_input=tools.Ops.conv3d(conv_input_1,k=3,out_c=original,str=2,name='conv_input')
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
            # lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=d1 * d2 * d3 * cc, name='fc2'),name='relu')
            # lfc = tf.reshape(lfc, [bat, d1, d2, d3, cc])

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
            concat_up_1 = tf.concat([deconv_2, conv_input_1], axis=4)
            predict_map = tools.Ops.conv3d(concat_up_1,k=1,out_c=1,str=1,name='predict_map')

            # zoom in layer
            # predict_map_normed = tools.Ops.batch_norm(predict_map,'bn_after_dense_1',training=training)
            # predict_map_zoomed = tools.Ops.deconv3d(predict_map_normed,k=2,out_c=1,str=2,name='deconv_zoom_3')

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
            c_d = [1,2,64,128,2565]
            s_d = [0,2,2,2,2]
            layers_d =[]
            layers_d.append(layer)
            for i in range(1,5,1):
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
        g_lung = tf.Graph()
        with g_lung.as_default():
            # X = tf.placeholder(shape=[batch_size, input_shape[0], input_shape[1], input_shape[2]], dtype=tf.float32)
            test_input_shape = input_shape
            test_batch_size = batch_size
            threshold = tf.placeholder(tf.float32)
            training = tf.placeholder(tf.bool)
            X = tf.placeholder(shape=[test_batch_size, test_input_shape[0], test_input_shape[1], test_input_shape[2]],
                               dtype=tf.float32)
            with tf.variable_scope('lung_net'):
                Y_pred, Y_pred_modi, Y_pred_nosig = self.ae_u(X, training, test_batch_size, threshold)

            # print tools.Ops.variable_count()
            # sum_merged = tf.summary.merge_all()
            saver = tf.train.Saver(max_to_keep=1)
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.visible_device_list = GPU0
            # with tf.Session(config=config) as sess:
            sess = tf.Session(config=config)
            if os.path.exists(self.train_models_dir):
                saver.restore(sess, self.train_models_dir + 'model.cptk')

            if os.path.isfile(self.train_models_dir + 'model.cptk.data-00000-of-00001'):
                print "restoring saved model"
                saver.restore(sess, self.train_models_dir + 'model.cptk')
            else:
                sess.run(tf.global_variables_initializer())
            test_data = tools.Test_data(dicom_dir, input_shape, 'dicom_data')
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
                    with tf.variable_scope('lung_net', reuse=True):
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
            # print "writing final testing result"
            # ST.WriteImage(final_img, './lung_mask.vtk')
            return final_img

def post_process(img,dicom_dir):
    # print img.GetSize()
    original_img = read_dicoms(dicom_dir)
    img_array = np.transpose(ST.GetArrayFromImage(img),[2,1,0])
    img_shape = np.shape(img_array)
    # Get outer mask to ensure outer noise get excluded
    original_array = ST.GetArrayFromImage(original_img)
    min_val = np.min(original_array)
    outer_seeds = []
    inner_step = 2
    outer_seeds.append([inner_step, inner_step, img_shape[2] - inner_step])
    outer_seeds.append([inner_step, img_shape[1] - inner_step, inner_step])
    outer_seeds.append([img_shape[0] - inner_step, inner_step, inner_step])
    outer_seeds.append([inner_step, img_shape[1] - inner_step, img_shape[2] - inner_step])
    outer_seeds.append([img_shape[0] - inner_step, inner_step, img_shape[2] - inner_step])
    outer_seeds.append([img_shape[0] - inner_step, img_shape[1] - inner_step, inner_step])
    outer_seeds.append([img_shape[0] - inner_step, img_shape[1] - inner_step, img_shape[2] - inner_step])
    outer_space = ST.NeighborhoodConnected(original_img, outer_seeds, min_val * 1.0, -200, [1, 1, 0], 1.0)
    # ST.WriteImage(outer_space , './outer_space.vtk')
    outer_array = ST.GetArrayFromImage(outer_space)
    outer_array = np.transpose(outer_array, [2, 1, 0])
    # Take out outer noise
    inner_array = np.float32((img_array - outer_array) > 0)
    inner_img = ST.GetImageFromArray(np.transpose(inner_array,[2,1,0]))
    # ST.WriteImage(inner_img,'./inner_mask.vtk')

    median_filter = ST.MedianImageFilter()
    median_filter.SetRadius(1)
    midian_img = median_filter.Execute(inner_img)
    midian_array = ST.GetArrayFromImage(midian_img)
    midian_array = np.transpose(midian_array,[2,1,0])
    array_shape = np.shape(midian_array)

    seed = [0,0,0]
    max = 0
    for i in range(array_shape[0]):
        temp_max = np.sum(midian_array[i,:,:])
        if max < temp_max:
            max = temp_max
            seed[0]=i
    max = 0
    for i in range(array_shape[1]):
        temp_max = np.sum(midian_array[:,i,:])
        if max < temp_max:
            max = temp_max
            seed[1]=i
    max = 0
    for i in range(array_shape[2]):
        temp_max = np.sum(midian_array[:,:,i])
        if max < temp_max:
            max = temp_max
            seed[2]=i
    # print seed
    growed_img = ST.NeighborhoodConnected(img, [seed], 0.9,1, [1, 1, 1], 1.0)

    return img,growed_img

def Lung_Seg(dicom_dir):
    time1 = time.time()
    original_img = read_dicoms(dicom_dir)
    net = Network()
    final_img = net.test(dicom_dir)
    del net
    gc.collect()
    img_spacing = final_img.GetSpacing()
    time2 = time.time()
    print "time cost for lung sement: ",str(time2-time1),'s'
    time3 = time.time()
    final_img, growed_mask = post_process(final_img,dicom_dir)
    growed_mask.SetSpacing(img_spacing)
    print "Writing lung mask"
    # ST.WriteImage(growed_mask, './output/lung_mask.vtk')
    time4 = time.time()
    print "time cost for lung post_process: ",str(time4-time3),'s'
    final_array = ST.GetArrayFromImage(growed_mask)
    img_array = ST.GetArrayFromImage(original_img)
    lung_array = final_array*img_array
    # lung_array = lung_array + np.min(lung_array)*2*np.int8(lung_array==0)
    lung_img = ST.GetImageFromArray(lung_array)
    lung_img.SetSpacing(img_spacing)
    print "Writing lung image"
    # ST.WriteImage(lung_img,'./output/lung_img.vtk')
    return lung_img

# if __name__ =="__main__":
#     dicom_dir = "./WANG_REN/original1"
#     lung_img = Lung_Seg(dicom_dir)
