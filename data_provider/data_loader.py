import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_BP(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None,speeds=16,num_antenna=32,configs=None):
        if size == None:
            self.seq_len = 32
            self.label_len = 16
            self.pred_len = 16
            self.total_pred_len = 16
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            self.total_pred_len =size[3]
            # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.speeds=speeds
        self.num_antenna=num_antenna
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.configs=configs
        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler= StandardScaler()
        speeds =self.speeds
        bs_attenna_numbers=self.num_antenna
        #speeds = np.arange(10, 30, 5)
        #bs_attenna_numbers = [32, 64, 128]
        self.lenspeed=len(speeds)
        self.lenatten=len(bs_attenna_numbers)
        file_range=30
        matrices=[]
        snr_matrices=[]
        att_matrices=[]
        phi_matrices=[]
        gain_matrices=[]
        for i in range(file_range):
            for bs_attenna_number in bs_attenna_numbers:
                for speed in speeds:
                    self.data_path=f'ODE_dataset_v_{speed}/beam_label_a{bs_attenna_number}_v{speed}_{i}.csv'
                    whole_path=os.path.join(self.root_path,self.data_path)
                    df_raw=pd.read_csv(whole_path)
                    df_raw=df_raw.to_numpy()
                    len_data = df_raw.shape[-1]
                    df_raw=df_raw/bs_attenna_number
                    att_data=np.ones_like(df_raw)*bs_attenna_number
                    att_matrices.append(att_data)
                    matrices.append(df_raw)


                    self.data_path_snr = f'ODE_dataset_v_{speed}/beam_snr_a{bs_attenna_number}_v{speed}_{i}.csv'
                    whole_path_snr = os.path.join(self.root_path, self.data_path_snr)
                    df_raw_snr = pd.read_csv(whole_path_snr)
                    df_raw_snr = df_raw_snr.to_numpy()
                    df_raw_snr=df_raw_snr.reshape(256,5,len_data)
                    df_raw_snr=df_raw_snr.transpose(0,2,1)
                    snr_matrices.append(df_raw_snr)
                    if self.set_type==2 and (27 <= i < 30):
                        #print(i)
                        self.data_path_gain = f'ODE_dataset_v_{speed}/normal_gain_a{bs_attenna_number}_v{speed}_{i}.csv'
                        whole_path_gain = os.path.join(self.root_path, self.data_path_gain)
                        df_raw_gain = pd.read_csv(whole_path_gain)
                        df_raw_gain = df_raw_gain.to_numpy()
                        df_raw_gain = df_raw_gain.reshape(256, len_data, -1)
                        gain_matrices.append(df_raw_gain)

                    self.data_path_phi = f'ODE_dataset_v_{speed}/beam_phi_a{bs_attenna_number}_v{speed}_{i}.csv'
                    whole_path_phi = os.path.join(self.root_path, self.data_path_phi)
                    df_raw_phi = pd.read_csv(whole_path_phi)
                    df_raw_phi = df_raw_phi.to_numpy()
                    df_raw_phi = np.sin((df_raw_phi/180)*np.pi)
                    df_raw_phi = df_raw_phi.reshape(256, 1, len_data)
                    df_raw_phi = df_raw_phi.transpose(0, 2, 1)
                    phi_matrices.append(df_raw_phi)

        if matrices:
            big_matrix= np.vstack(matrices)
            big_att_matrix=np.vstack(att_matrices)
            beam_snr= np.vstack(snr_matrices)
            phi=np.vstack(phi_matrices)
            if self.set_type == 2:
                gain=np.vstack(gain_matrices)
            beam_snr=np.concatenate([beam_snr,phi],axis=-1)

        border1s=np.array([0,256*8*len(speeds)*len(bs_attenna_numbers),256*9*len(speeds)*len(bs_attenna_numbers)])*(file_range//10)
        border2s = np.array(
            [256*8*len(speeds)*len(bs_attenna_numbers), 256 * 9 * len(speeds) * len(bs_attenna_numbers), 256 * 10 * len(speeds) * len(bs_attenna_numbers)]) * (
                               file_range // 10)
        border1= border1s[self.set_type]
        border2=border2s[self.set_type]
        if self.scale:
            if self.configs.M_phi:
                train_data = phi[border1s[0]:border2s[0]]#(-1,51)
                train_data= train_data.reshape(-1,1)#(-1,1)
                self.scaler.fit(train_data)#(-1,1)
                data = self.scaler.transform(phi.reshape(-1,1))#(-1，1)
                data = data.reshape(-1,len_data)
            else:
                train_data = big_matrix[border1s[0]:border2s[0]]  # (-1,51)
                train_data = train_data.reshape(-1, 1)  # (-1,1)
                self.scaler.fit(train_data)  # (-1,1)
                data = self.scaler.transform(big_matrix.reshape(-1, 1))  # (-1，1)
                data = data.reshape(-1, len_data)
        else:
            data = big_matrix


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        #print(self.data_y[0,40:50])
        if self.set_type == 2:
            self.gain= gain
        self.data_stamp = data[border1:border2]
        self.att_number = big_att_matrix[border1:border2]
        self.beam_snr= beam_snr [border1:border2]

    def transform(self,data):

        data_transform = self.scaler.transform(data.reshape(-1, 1))
        data_transform=data_transform.reshape(data.shape[0],data.shape[1])
        return data_transform

    def __getitem__(self, index):
        if index<0 or index>len(self.data_x):
            raise IndexError('Index out of range')
        seq_x=self.data_x[index,0:self.seq_len]

        att_num_x=self.att_number[index,0:self.seq_len]
        if self.set_type==2 and self.total_pred_len!=10:
            seq_y = self.data_y[index, self.seq_len:self.seq_len + self.total_pred_len]
            att_num_y = self.att_number[index, self.seq_len:self.seq_len + self.pred_len]
            beam_snr_x = self.beam_snr[index, 0:self.seq_len+self.total_pred_len, :]
        else:
            seq_y = self.data_y[index, self.seq_len:self.seq_len+self.pred_len]
            att_num_y = self.att_number[index, self.seq_len:self.seq_len+self.pred_len]
            beam_snr_x = self.beam_snr[index, 0:self.seq_len, :]

        seq_x = seq_x[:, np.newaxis]
        seq_y = seq_y[:, np.newaxis]
        seq_x_mark=seq_x
        seq_y_mark=seq_y
        if self.set_type == 2:
            seq_y_gain=self.gain[index, self.seq_len:self.seq_len+self.total_pred_len,:]
            seq_y_dict = {'data': seq_y, 'att_num': att_num_y, 'gain': seq_y_gain}
        else:
            seq_y_dict = {'data': seq_y, 'att_num': att_num_y}
        seq_x_dict={'data':seq_x,'att_num':att_num_x,'snr':beam_snr_x}



        return seq_x_dict, seq_y_dict, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)
        #return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

