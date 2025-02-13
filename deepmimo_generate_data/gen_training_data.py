import numpy as np
import pandas as pd
import os
import scipy
import random
import DeepMIMO

os.makedirs('./data', exist_ok=True)
parameters = DeepMIMO.default_params()

# Change parameters for the setup
user_low_first = 100  # select by ourselves
user_low_last = 900
num_atten = 64

# Scenario O1_60 extracted at the dataset_folder
parameters['scenario'] = 'O1_28'
parameters['dataset_folder'] = r'scenarios'
parameters['num_paths'] = 1

# User rows 1-1000
parameters['user_row_first'] = user_low_first
parameters['user_row_last'] = user_low_last  # 2751

# Activate only the first basestation
parameters['active_BS'] = np.array([1])
parameters['OFDM']['bandwidth'] = 0.1  # 50 MHz
parameters['OFDM']['subcarriers'] = 256  # OFDM with 512 subcarriers
parameters['OFDM']['subcarriers_limit'] = 64  # Keep only first 64 subcarriers

parameters['ue_antenna']['shape'] = np.array([1, 1, 1])  # Single antenna
parameters['bs_antenna']['shape'] = np.array([1, num_atten, 1])  # ULA of 32 elements
parameters['BS2BS'] = 0

# Generate and inspect the dataset
dataset = DeepMIMO.generate_data(parameters)
sum=0
for j in range(len(dataset[0]['user']['LoS'])):
    if not dataset[0]['user']['LoS'][j]:
        sum+=1
        print(sum)
    #print(dataset[0]['user']['LoS'])
DoD_phi_matrices=[]

for print_index in range(len(dataset[0]['user']['paths'])):
    DoD_phi_matrices.append(dataset[0]['user']['paths'][print_index]['DoD_phi'][0])
    #print(dataset[0]['user']['paths'][print_index])
DoD_phi_matrices=np.array(DoD_phi_matrices).reshape(parameters['user_row_last'] - parameters['user_row_first'] + 1, 181)


bs_antenna_number = num_atten  # ULA of 32 elements
bs_beam_num = num_atten  # narrow beam number
sector_start = - np.pi
sector_end = np.pi

candidate_narrow_beam_angle = sector_start + (sector_end - sector_start) / bs_beam_num * np.arange(0.5, bs_beam_num + 0.5, 1)
bs_antenna_index = np.arange(bs_antenna_number).reshape(bs_antenna_number, 1)
candidate_narrow_beam = np.exp(- 1j * bs_antenna_index * (candidate_narrow_beam_angle.reshape(1, bs_beam_num))) / np.sqrt(bs_antenna_number)

# UE distribution
row_index = np.arange(user_low_first, user_low_last + 1)
MM_channel = dataset[0]['user']['channel'].reshape(parameters['user_row_last'] - parameters['user_row_first'] + 1, 181, num_atten, 64)
MM_channel = np.sum(MM_channel, axis=3)

# MM_ch = np.zeros((len(row_index), 181, 32), dtype=complex)
# number of considered rows, users per row, OFDM sub-carrier
MM_ch = np.matmul(MM_channel, candidate_narrow_beam)

file_num = 30  # generate 10 files, 40 for training and 10 for testing
file_size = 256  # sample number in each file
speeds = np.arange(5,25,5)
#speeds=[5]

beam_tracking_time = 1.6
# beam_training_duration = 0.16
beam_prediction_resolution = 0.016
t = int(beam_tracking_time / beam_prediction_resolution + 1)
# MM beam training received signal
MM_data = np.zeros((file_size, 2, t, bs_beam_num))
# MM optimal beam index
num_training_beam = 1
num_neighbor = 2

beam_label = np.zeros((file_size, t, num_training_beam))  # store the best num_training_beam beams
beam_true_label = np.zeros((file_size, t, 1))
beam_snr=np.zeros((file_size, t, num_training_beam+2*num_neighbor))
normal_gain = np.zeros((file_size, t, bs_antenna_number))  # Store the best three beams
locations = np.zeros((file_size, t, 2))  # store the user locations
DoD_phi=np.zeros((file_size, t))


t_fix = 10  # after t_fix time steps, the user change the directions
for speed in speeds:
    file_name = 'ODE_dataset_v_{}'.format(speed)
    os.makedirs('./data/' + file_name, exist_ok=True)
    for i in range(file_num):
        false_num = 0
        for j in range(file_size):
            # find UE trajectory within the predefined range

            flag = 0

            while flag == 0:
                user_speed = speed
                location = np.zeros((t, 2))  # the user location
                initial_x = round(200 + random.random() * 600)  # left some space
                initial_y = round(random.random() * 180)  # the range is selected by ourselves
                initial = np.array([initial_x, initial_y], dtype=np.float64)
                location[0, :] = initial
                dir_angel = random.random() * 2 * np.pi
                direction = np.array([np.cos(dir_angel), np.sin(dir_angel)], dtype=np.float64)

                for t_step in range(t - 1):

                    # if t_step % t_fix == 0:
                    #      dir_angel = (random.random() - 0.5) * np.pi * 2/3 + dir_angel
                    #      direction = np.array([np.cos(dir_angel), np.sin(dir_angel)], dtype=np.float64)

                    a = random.uniform(0, 1.0) * 0.2 * speed * beam_prediction_resolution
                    user_speed += a
                    vary_distance = user_speed * 5.0 * beam_prediction_resolution
                    update_location = np.round(initial + vary_distance * direction)
                    location[t_step + 1, :] = update_location
                    # update initial_x and initial_y
                    initial = update_location

                if np.min(location[:, 0]) >= user_low_first and np.max(location[:, 0]) < user_low_last and np.min(location[:, 1]) >= 0 and np.max(location[:, 1]) <= 180:
                    flag = 1
                    locations[j, :, :] = location
                else:
                    false_num+=1
                    # print("speed",user_speed)
                    # print("false_num", false_num)

            # save corresponding data
            # MM_data: sequence of mmWave beam training received signal vector
            # beam label: sequence of mmWave optimal beam

            index_last = 0  # store the last chosen beam
            for t_step in range(t):
                DoD_phi[j,t_step]=DoD_phi_matrices[int(location[t_step, 0] - user_low_first + 1), int(location[t_step, 1])]
                MM_chs = np.squeeze(
                    np.abs(MM_ch[int(location[t_step, 0] - user_low_first + 1), int(location[t_step, 1]), :]))
                if t_step == 0:
                    arr_len = len(MM_chs)
                    values = []
                    # 获取左侧的邻居
                    for left in range(num_neighbor, 0, -1):
                        values.append(MM_chs[(index_last - left) % arr_len])
                    values.append(MM_chs[index_last])
                    # 获取右侧的邻居
                    for right in range(1, num_neighbor + 1):
                        values.append(MM_chs[(index_last + right) % arr_len])
                    values = np.array(values)
                    beam_snr[j, t_step, :] = values

                    top_3_indices = np.argsort(MM_chs)[-3:][::-1]
                    index_last = top_3_indices[0]  # set as the initial best beam
                    beam_label[j, t_step, :] = index_last  # top_3_indices
                    beam_true_label[j, t_step, :] = index_last
                else:
                    arr_len = len(MM_chs)
                    values = []
                    # 获取左侧的邻居
                    for left in range(num_neighbor, 0, -1):
                        values.append(MM_chs[(index_last - left) % arr_len])
                    values.append(MM_chs[index_last])
                    # 获取右侧的邻居
                    for right in range(1, num_neighbor + 1):
                        values.append(MM_chs[(index_last + right) % arr_len])
                    values = np.array(values)
                    beam_snr[j, t_step, :]=values
                    # left = MM_chs[index_last - 1] if index_last > 0 else MM_chs[-1]
                    # current = MM_chs[index_last]
                    # right = MM_chs[index_last + 1] if index_last < bs_antenna_number - 1 else MM_chs[0]
                    # values =np.vstack((left,current,right)).squeeze()

                    sorted_indices = np.argsort(values)[::-1]
                    sorted_indices = (sorted_indices + index_last - num_neighbor) % bs_antenna_number
                    # sorted_indices = sorted_indices[:, np.newaxis]
                    index_last = sorted_indices[0]
                    beam_label[j, t_step, :] = index_last  # np.squeeze(sorted_indices)

                    top_3_indices = np.argsort(MM_chs)[-3:][::-1]
                    index_true = top_3_indices[0]
                    beam_true_label[j, t_step, :] = index_true

                MM_chs_2 = np.power(MM_chs, 2)
                MM_chs_max = np.max(MM_chs_2)
                MM_gain_normal = MM_chs_2 / MM_chs_max
                normal_gain[j, t_step, :] = MM_gain_normal
                # MM_data[j, 0, t_step, :] = np.real(MM_ch[int(location[t_step, 0] - user_low_first + 1),
                # int(location[t_step, 1]), :])
                # MM_data[j, 1, t_step, :] = np.imag(MM_ch[int(location[t_step, 0] - user_low_first + 1),
                # int(location[t_step, 1]), :])

            for t_step in range(1, t):  # prevent the sudden change
                threshold_left = beam_label[j, t_step - 1, :] - bs_antenna_number / 2
                threshold_right = beam_label[j, t_step - 1, :] + bs_antenna_number / 2
                beam_label_im = beam_label[j, t_step, :]
                beam_label_im[beam_label_im > threshold_right] -= bs_antenna_number
                beam_label_im[beam_label_im < threshold_left] += bs_antenna_number
                beam_label[j, t_step, :] = beam_label_im

                # additive white gaussian noise?
        beam_label_store = beam_label.transpose([0, 2, 1])
        #print(normal_gain.shape)
        normal_gain_store = normal_gain.reshape(file_size, -1)

        beam_snr_store=beam_snr.transpose([0, 2, 1])
        df_snr=pd.DataFrame(beam_snr_store.reshape(-1, t))
        df_snr.to_csv('./data/' + file_name + '/beam_snr_a{}'.format(bs_antenna_number) + '_v{}'.format(speed)
                  + '_{}.csv'.format(i), index=False)

        df_phi = pd.DataFrame(DoD_phi)
        df_phi.to_csv('./data/' + file_name + '/beam_phi_a{}'.format(bs_antenna_number) + '_v{}'.format(speed)
                      + '_{}.csv'.format(i), index=False)

        df = pd.DataFrame(beam_label_store.reshape(file_size * num_training_beam, t))
        df.to_csv('./data/' + file_name + '/beam_label_a{}'.format(bs_antenna_number) + '_v{}'.format(speed)
                  + '_{}.csv'.format(i), index=False)
        if i in range(27, 30):
            df_1 = pd.DataFrame(normal_gain_store)
            df_1.to_csv('./data/' + file_name + '/normal_gain_a{}'.format(bs_antenna_number) + '_v{}'.format(speed)
                        + '_{}.csv'.format(i), index=False)

        # beam_true_label_store = beam_true_label.transpose([0, 2, 1])
        # df_true = pd.DataFrame(beam_true_label_store.reshape(file_size, t))
        # df_true.to_csv('./data/' + file_name + '/beam_true_label_a{}'.format(bs_antenna_number) + '_v{}'.format(
        # speed) + '_{}.csv'.format(i), index=False)
        # scipy.io.savemat('./data/' + file_name + '/beam_label_a{}'.format(bs_antenna_number) + '_v{}'.format(speed)
        # + '_{}.mat'.format(i), {'location': locations})




















