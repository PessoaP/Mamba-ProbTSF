import random

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate##, visual
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from mamba_ssm import Mamba

warnings.filterwarnings('ignore')

class Sigma_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Sigma_NN, self).__init__()
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.GELU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc_input(x)
        x = self.relu(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.fc_out(x)
        x = self.softplus(x) + 1e-8
        return x
    
softplus = nn.Softplus()

def gaussian_negloglike(obs,mus,sigs):
    z = (obs-mus)/sigs
    return torch.pow(z,2)/2 + torch.log(sigs)



from model import S_Mamba
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S_Mamba': S_Mamba,
        }
        self.device = self._acquire_device()
        self.model = [mdl.to(self.device) for mdl in self._build_model()]

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class Exp_Probabilistic_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Probabilistic_Forecast, self).__init__(args)

    def _build_model(self):
        model_mus = self.model_dict[self.args.model].Model(self.args).float()
    
        if self.args.sigma_network == 'Linear':
            model_sig = Sigma_NN(self.args.seq_len,
                                512,
                                self.args.pred_len).float().to(torch.device('cuda:{}'.format(self.args.gpu))) 
            if self.args.sigma_method == 'Compound':
                self.get_sigma = lambda model, batch_x, batch_x_mark, dec_inp, batch_y_mark: torch.sqrt(torch.pow(model(batch_x.permute(0, 2, 1)),2).cumsum(axis=-1)).permute(0, 2, 1)  
            else:
                self.get_sigma = lambda model, batch_x, batch_x_mark, dec_inp, batch_y_mark: model(batch_x.permute(0, 2, 1)).permute(0, 2, 1)
        
        else: #elif S-Mamba
            model_sig = self.model_dict[self.args.model].Model(self.args).float()
            if self.args.sigma_method == 'Compound':
                self.get_sigma = lambda model, batch_x, batch_x_mark, dec_inp, batch_y_mark: torch.sqrt(torch.pow(softplus(model(batch_x,batch_x_mark, dec_inp, batch_y_mark)),2).cumsum(axis=1))  + 1e-8
            else:
                self.get_sigma = lambda model, batch_x, batch_x_mark, dec_inp, batch_y_mark: softplus(model(batch_x,batch_x_mark, dec_inp, batch_y_mark)) + 1e-8


        return model_mus, model_sig

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(list(self.model[0].parameters())+list(self.model[1].parameters()), 
                                 lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def train(self, setting,prob=True):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            
            iter_count = 0
            train_loss = []

            [mdl.train() for mdl in self.model]
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_mus = self.model[0](batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs_mus = outputs_mus[:, -self.args.pred_len:, f_dim:]

                if prob:
                    # outputs_sig = self.model[1](batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # #outputs_sig = self.model[1](batch_x)
                    outputs_sig = self.get_sigma(self.model[1],batch_x,batch_x_mark, dec_inp, batch_y_mark)
                    # outputs_sig = softplus(outputs_sig[:, -self.args.pred_len:, f_dim:])
                    outputs_sig = outputs_sig[:, -self.args.pred_len:, f_dim:]   

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)                    
                    loss =  gaussian_negloglike(batch_y,outputs_mus,outputs_sig).mean()
                
                else: 
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = (torch.pow( (batch_y - outputs_mus) , 2)).mean()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count

                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                else:
                    loss.backward()
                    model_optim.step()
                #del loss


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion,prob=prob)
            test_loss = self.vali(test_data, test_loader, criterion,prob=prob)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model[0], path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model[0].load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali(self, vali_data, vali_loader, criterion,prob=True):
        total_loss = []
        [mdl.eval() for mdl in self.model]
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs_mus = self.model[0](batch_x, batch_x_mark, dec_inp, batch_y_mark)
                               
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_mus = outputs_mus[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs_mus.detach().cpu()
                true = batch_y.detach().cpu()

                if prob:
                    # outputs_sig = self.model[1](batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # #outputs_sig = self.model[1](batch_x)
                    # outputs_sig = softplus(outputs_sig)
                    outputs_sig = self.get_sigma(self.model[1],batch_x,batch_x_mark, dec_inp, batch_y_mark)
                    sigs = outputs_sig.detach().cpu()            

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)                    
                    loss =  gaussian_negloglike(true,pred,sigs).mean() #(torch.pow( ((true-pred)/sigs) , 2)/2 + torch.log(sigs)).mean()
                else: 
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = (torch.pow( (true-pred) , 2)).mean()


                total_loss.append(loss)
        total_loss = np.average(total_loss)
        [mdl.train() for mdl in self.model]
        return total_loss


    def test(self, setting, test=0,prob = True):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model[0].load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        inputs= []
        preds = []
        trues = []
        sigs = []
        folder_path = './test_results_prob/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        [mdl.eval() for mdl in self.model]
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs_mus = self.model[0](batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_mus = outputs_mus[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs_mus = outputs_mus.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                input = batch_x.detach().cpu().numpy()
                pred = outputs_mus
                true = batch_y

                inputs.append(input)
                preds.append(pred)
                trues.append(true)

                if prob:

                    outputs_sig = self.get_sigma(self.model[1],batch_x,batch_x_mark, dec_inp, batch_y_mark)
                    outputs_sig = outputs_sig[:, -self.args.pred_len:, f_dim:] 

                    sig = outputs_sig.detach().cpu()    
                    sigs.append(sig)      
                


        inputs = np.array(inputs)
        preds = np.array(preds)
        trues = np.array(trues)
        if prob:
            sigs = np.array(sigs)    


        print('test shape:', inputs.shape, preds.shape, trues.shape)  

        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        if prob:
            #sigs = np.array(sigs)      
            sigs = sigs.reshape(-1, sigs.shape[-2], sigs.shape[-1])
            print('test shape:', inputs.shape, preds.shape, trues.shape,sigs.shape)
        else:
            print('test shape:', inputs.shape, preds.shape, trues.shape)  

        # result save
        folder_path = './results_prob/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        #Will 'de-standardize data'
        if test_data.scale and self.args.inverse:
            inputs = test_data.scaler.mean_ + test_data.scaler.scale_*inputs
            preds = test_data.scaler.mean_ + test_data.scaler.scale_*preds
            trues = test_data.scaler.mean_ + test_data.scaler.scale_*trues
            if prob:
                sigs = test_data.scaler.scale_*sigs
        

        np.save(folder_path + 'input.npy',inputs)
        np.save(folder_path + 'trues.npy',trues)

        if prob:
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy',preds)
            np.save(folder_path + 'sigs.npy',sigs)
        else:
            np.save(folder_path + 'det_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'det_pred.npy', preds)
        
        return
    
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results_prob/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model[0].load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     [mdl.eval() for mdl in self.model]
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             outputs = self.model[0](batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             outputs = outputs.detach().cpu().numpy()
    #             if pred_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
    #             preds.append(outputs)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results_prob/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'real_prediction.npy', preds)

    #     return