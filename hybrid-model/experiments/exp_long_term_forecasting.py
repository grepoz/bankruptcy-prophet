import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_metrics
from utils.metrics import metric, calculate_accuracy, calculate_f1, calculate_precision, calculate_specificity
from model import IT_HBERT

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast:
    def __init__(self, hybrid_model_args, iTransformer_args, HBERT_args):
        self.hybrid_model_args = hybrid_model_args
        self.iTransformer_args = iTransformer_args
        self.HBERT_args = HBERT_args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.hybrid_model_args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.hybrid_model_args.gpu) if not self.hybrid_model_args.use_multi_gpu else self.hybrid_model_args.devices
            device = torch.device('cuda:{}'.format(self.hybrid_model_args.gpu))
            print('Use GPU: cuda:{}'.format(self.hybrid_model_args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = IT_HBERT.Model(self.iTransformer_args, self.HBERT_args).float()

        if self.hybrid_model_args.use_multi_gpu and self.hybrid_model_args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.hybrid_model_args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.hybrid_model_args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.hybrid_model_args.learning_rate)

    def _select_criterion(self):
        return nn.BCELoss(reduction='none')

    def calculate_loss(self, outputs, batch_y_numerical):
        criterion = self._select_criterion()

        onehot_encoded_truth = nn.functional.one_hot(batch_y_numerical.to(torch.int64), 2).squeeze(1).float()

        loss = criterion(outputs, onehot_encoded_truth)

        weights = torch.tensor([1.0, 49.0])

        weighted_loss = loss * weights
        final_loss = weighted_loss.mean()

        return final_loss

    def train(self, setting):
        numeric_train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print('loaded data - training model')

        train_metrics_directory = './train_results/metrics'

        if not os.path.exists(train_metrics_directory):
            os.makedirs(train_metrics_directory)

        path = os.path.join(self.hybrid_model_args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.hybrid_model_args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_losses = []
        train_accuracies = []
        vali_losses = []
        vali_accuracies = []
        test_losses = []
        test_accuracies = []

        for epoch in tqdm(range(self.hybrid_model_args.train_epochs)):
            epoch_loss = []
            epoch_accuracy = []

            self.model.train()

            for i, (batch_x_numerical, batch_y_numerical, batch_x_textual) in (enumerate(pbar := tqdm(train_loader, position=0))):
                pbar.set_description(f"Epoch: {epoch+1}/{self.hybrid_model_args.train_epochs}")

                model_optim.zero_grad()

                batch_x_numerical = batch_x_numerical[0].float().to(self.device)
                batch_y_numerical = batch_y_numerical[0].float().to(self.device)
                batch_x_textual = batch_x_textual[0].float().to(self.device)

                outputs = self.model(batch_x_numerical, batch_x_textual).float().to(self.device)

                loss = self.calculate_loss(outputs, batch_y_numerical)

                epoch_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                batch_y_numerical = batch_y_numerical.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                outputs = np.argmax(outputs, axis=1)

                accuracy = calculate_accuracy(outputs, batch_y_numerical)
                epoch_accuracy.append(accuracy)

            epoch_loss = np.average(epoch_loss)
            train_losses.append(epoch_loss)
            train_accuracies.append(np.average(epoch_accuracy))

            vali_loss, vali_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            vali_losses.append(vali_loss)
            vali_accuracies.append(vali_accuracy)

            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("\nEarly stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.hybrid_model_args)

        best_model_path = f'{path}/checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        torch.save({
            'epoch': self.hybrid_model_args.train_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'loss': epoch_loss,
        }, best_model_path)

        save_metrics(train_accuracies, train_losses, f'{train_metrics_directory}/train.png')
        save_metrics(vali_accuracies, vali_losses, f'{train_metrics_directory}/vali.png')
        save_metrics(test_accuracies, test_losses, f'{train_metrics_directory}/test.png')

        return self.model

    def vali(self, vali_data, vali_loader, criterion):

        total_loss = []
        total_accuracy = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_numerical, batch_y_numerical, batch_x_textual) in enumerate(vali_loader):

                batch_x_numerical = batch_x_numerical[0].float().to(self.device)
                batch_y_numerical = batch_y_numerical[0].float().to(self.device)
                batch_x_textual = batch_x_textual[0].float().to(self.device)

                outputs = self.model(batch_x_numerical, batch_x_textual).float().to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y_numerical.detach().cpu()

                loss = self.calculate_loss(pred, true)

                total_loss.append(loss)

                batch_y_numerical = batch_y_numerical.detach().cpu().numpy()
                pred = np.argmax(pred, axis=1).numpy()

                accuracy = calculate_accuracy(pred, batch_y_numerical)
                total_accuracy.append(accuracy)

        total_loss = np.average(total_loss)
        total_accuracy = np.average(total_accuracy)

        self.model.train()

        return total_loss, total_accuracy

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_numerical, batch_y_numerical, batch_x_textual) in enumerate(test_loader):

                batch_x_numerical = batch_x_numerical[0].float().to(self.device)
                batch_y_numerical = batch_y_numerical[0].float().to(self.device)
                batch_x_textual = batch_x_textual[0].float().to(self.device)

                outputs = self.model(batch_x_numerical, batch_x_textual).float().to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y_numerical.detach().cpu().numpy()

                if test_data.scale and self.iTransformer_args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = np.argmax(outputs, axis=1)

                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds).reshape(-1)
        trues = np.array(trues).reshape(-1)

        accuracy = calculate_accuracy(preds, trues)
        precision = calculate_precision(preds, trues)
        f1 = calculate_f1(preds, trues)
        specificity = calculate_specificity(preds, trues)

        print(f'Test accuracy: {accuracy}, precision: {precision}, specificity: {specificity}, f1: {f1}')

        # result save
        folder_path = './results/' + setting + '/'
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

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.hybrid_model_args.checkpoints, setting)
            best_model_path = f'{path}/checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, None)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.iTransformer_args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return