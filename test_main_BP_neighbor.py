import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import pandas as pd

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content,get_near

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='Time-LLM',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=False, default='BP', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/BP_dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
parser.add_argument('--label_len', type=int, default=16, help='start token length')
parser.add_argument('--pred_len', type=int, default=16, help='prediction sequence length')
parser.add_argument('--total_pred_len', type=int, default=10, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
#parser.add_argument('--speeds', type=list, nargs='+')
parser.add_argument('--speeds', type=int, nargs='+', default=16,help='speeds')
parser.add_argument('--num_antenna', type=int, nargs='+', default=32,help='speeds')


# model define
parser.add_argument('--M_phi', type=int, default=0, help='phi as input')
parser.add_argument('--add_snr', type=int, default=0, help='load model')
parser.add_argument('--load_model', type=int, default=0, help='load model')
parser.add_argument('--islora', type=int, default=0, help='lora')
parser.add_argument('--num_tokens', type=int, default=100, help='encoder input size')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--num_neighbor', type=int, default=3, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization

parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_lr{}_ba{}_tok{}_pa{}_st{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        args.learning_rate,
        args.batch_size,
        args.num_tokens,
        args.patch_len,
        args.stride)
    if args.data=='BP':
        args.label_len=args.pred_len

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    train_steps=len(train_loader)

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()


    if args.islora:
        path = os.path.join(args.checkpoints,
                            'lora'+setting + '-' + args.model_comment)  # unique checkpoint saving path
    else:
        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    accelerator.print('test_num',len(test_loader))

    if args.load_model:
        accelerator.print('loading model')
        best_model_path=path+'/'+'checkpoint'
        accelerator.wait_for_everyone()
        unwrapped_model=accelerator.unwrap_model(model)
        #torch.cuda.syschronize()
        torch.cuda.empty_cache()
        unwrapped_model.load_state_dict(torch.load(best_model_path,map_location=lambda storage, loc:storage))

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()



    total_loss=[]
    total_mae_loss=[]
    model.eval()
    pred_index_matrices=[]
    true_index_matrices=[]
    with torch.no_grad():
        for i, (batch_x_dict, batch_y_dict, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            if args.data == 'BP':
                batch_x_snr = batch_x_dict['snr'].float().to(accelerator.device)
                batch_x = batch_x_dict['data'].float().to(accelerator.device)
                batch_y = batch_y_dict['data'].float()
                batch_y_gain = batch_y_dict['gain'].float()
                batch_y_att = batch_y_dict['att_num']
            else:
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            # encoder - decoder
            repeat_times = args.total_pred_len // args.pred_len
            f_dim = -1 if args.features == 'MS' else 0

            batch_y = batch_y[:, -args.total_pred_len:, f_dim:].to(accelerator.device)
            batch_y_original = test_data.inverse_transform(batch_y.detach().squeeze().cpu().numpy())
            batch_y_att = batch_y_att.detach().cpu().numpy()
            #print(batch_y_original,batch_y_att)
            batch_y_att_tile=np.tile(batch_y_att,(1,repeat_times))
            true_att_index = np.multiply(batch_y_original, batch_y_att_tile)
            true_att_index = np.round(true_att_index)





            outputs_list = []
            pred_index_list=[]
            beam_past = batch_x
            for repeat in range(repeat_times):
                #print(beam_past[0,:])
                #print(true_att_index[0,:])
                phi_past = batch_x_snr[:, repeat * args.pred_len: args.seq_len + repeat * args.pred_len, :]
                gain_now=batch_y_gain[:, repeat * args.pred_len:(1 + repeat)* args.pred_len, :]
                outputs_temp = model(beam_past, phi_past, dec_inp, batch_y_mark)
                outputs_temp = outputs_temp[:, -args.pred_len:, f_dim:]
                #print(outputs_temp[0,:,:])
                outputs_original = test_data.inverse_transform(outputs_temp.detach().squeeze().cpu().numpy())
                pred_att_index=np.multiply(outputs_original,batch_y_att)
                pred_att_index = np.round(pred_att_index)


                near_true_index = get_near(gain_now, pred_att_index,args.num_neighbor)/batch_y_att[0,0] # here gain_now is timesteps = 10.

                pred_index_list.append(pred_att_index)
                outputs_list.append(outputs_temp)
                pred_transform= test_data.transform(near_true_index.detach().cpu().numpy())
                #print(pred_transform[0,:])
                pred_transform=torch.from_numpy(pred_transform).to(accelerator.device)
                beam_past = torch.concat([beam_past,pred_transform.unsqueeze(-1)], dim=1)
                beam_past = beam_past[:,-args.seq_len:,:]

            pred_att_indexs = np.concatenate(pred_index_list, axis=1)
            outputs=torch.concat(outputs_list,dim=1)

            pred_index_matrices.append(pred_att_indexs)


            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    accelerator.print(
        " Test Loss: {0:.7f} MAE Loss: {1:.7f}".format( total_loss, total_mae_loss))

    pred_index=np.vstack(pred_index_matrices)


    # speeds=[10]
    # bs_attenna_numbers=[32]
    speeds = args.speeds
    bs_attenna_numbers = args.num_antenna

    gain_matrices=[]
    index_file=0
    #i=27
    for i in range(27,30):
        for bs_attenna_number in bs_attenna_numbers:
            for speed in speeds:
                gain_data_path = f'ODE_dataset_v_{speed}/normal_gain_a{bs_attenna_number}_v{speed}_{i}.csv'
                gain_whole_path = os.path.join(args.root_path, gain_data_path)
                df_raw_gain = pd.read_csv(gain_whole_path)
                df_raw_gain = df_raw_gain.to_numpy()
                df_raw_gain = df_raw_gain.reshape(256,101,bs_attenna_number)
                gain_pred = df_raw_gain[:,args.seq_len:args.seq_len+args.total_pred_len,:]


                normal_gain= np.zeros([gain_pred.shape[0],gain_pred.shape[1]])
                for m in range(gain_pred.shape[0]):
                    for n in range(gain_pred.shape[1]):
                        #index_pred = raw_index[m, n] % bs_attenna_number
                        index_pred = pred_index[int(256*index_file+m),n]%bs_attenna_number
                        #if m==0 :
                            #accelerator.print(index_pred)
                            #accelerator.print(gain_pred[m,n,:])
                        normal_gain[m,n]=gain_pred[m,n,int(index_pred)]
                gain_matrices.append(normal_gain)
                index_file+=1




    normal_gain_big=np.vstack(gain_matrices)
    normal_gain_average=np.average(normal_gain_big,axis=0).reshape(1,-1)

    normal_gain_average = np.average(normal_gain_average.reshape(-1,10),axis=0).reshape(1,-1)


    result_path=os.path.join(args.root_path,'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    df_pred=pd.DataFrame(pred_index)
    df_pred.to_csv(result_path+'/beam_label_pred_v{}_a{}'.format(speeds,bs_attenna_numbers)+setting+'.csv',index=False)
    df_gain = pd.DataFrame(normal_gain_average)
    df_gain.to_csv(result_path + '/beam_label_gain_v{}_a{}_total{}_num{}'.format(speeds,bs_attenna_numbers,args.total_pred_len,args.num_neighbor) + setting + '.csv', index=False)

accelerator.wait_for_everyone()
#
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete checkpoints')