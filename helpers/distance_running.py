import pandas as pd
import numpy as np
from IPython.display import clear_output
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
from time import time, sleep
import pickle
import copy
import json
from helpers.my_plotting import my_interactable_xyline, my_traintestpredictions, my_xyline

INPUT_COLS = ["distance (m)", "elevation gain (m)", "average heart rate (bpm)", "time ago (s)"]

DATA = pd.read_csv("cleaned_data.csv", index_col='index')

class athlete_data():

    def __init__(self, athletes:list, race_idx = None, device = None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.athlete = athletes
        self.race_idx = race_idx

        # extract the rows for the chosen athlete
        if race_idx is None:
            self.X = DATA[DATA["athlete"].isin(athletes)][INPUT_COLS+["pace (min/km)"]]
            self.y = DATA[DATA["athlete"].isin(athletes)]["elapsed time (s)"]
        else:
            self.X = DATA[DATA["athlete"].isin(athletes)][INPUT_COLS+["pace (min/km)"]].drop(self.race_idx)
            self.y = DATA[DATA["athlete"].isin(athletes)]["elapsed time (s)"].drop(self.race_idx)

        # split the train/test data
        self.x_train_, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state=30)

        self.x_train = self.x_train_[INPUT_COLS]

        # initialise dataframes to store train/test data
        self.x_train_transformed = pd.DataFrame()
        self.y_train_transformed = pd.DataFrame()
        self.x_test_transformed = pd.DataFrame()
        self.y_test_transformed = pd.DataFrame()

        # transform the data, store the transformer objects
        self.yjpt_pace, self.pace_tranformed = yj_transform(self.x_train_["pace (min/km)"])
        self.pace_scaler = MinMaxScaler(feature_range=(0,1))
        self.pace_scaled = self.pace_scaler.fit_transform(self.pace_tranformed.reshape(-1,1))

        self.yjpt_distance, self.x_train_transformed["distance"], self.x_test_transformed["distance"] = yj_transform(self.x_train["distance (m)"], self.x_test["distance (m)"])

        self.yjpt_elevation, self.x_train_transformed["elevation"], self.x_test_transformed["elevation"] = yj_transform(self.x_train["elevation gain (m)"], self.x_test["elevation gain (m)"])
        
        self.x_train_transformed["hr"] = pd.Series(self.x_train["average heart rate (bpm)"].values)
        self.x_test_transformed["hr"] = pd.Series(self.x_test["average heart rate (bpm)"].values)

        self.yjpt_timeago, self.x_train_transformed["timeago"], self.x_test_transformed["timeago"] = yj_transform(self.x_train["time ago (s)"], self.x_test["time ago (s)"])

        # scale the data
        self.x_scaler = MinMaxScaler(feature_range=(0,1))
        self.x_scaler = self.x_scaler.fit(self.x_train_transformed)
        self.x_train_scaled = self.x_scaler.transform(self.x_train_transformed)
        self.x_test_scaled = self.x_scaler.transform(self.x_test_transformed)

        self.yjpt_time, self.y_train_transformed["time"], self.y_test_transformed["time"] = yj_transform(self.y_train, self.y_test)

        self.y_scaler = MinMaxScaler(feature_range=(0,1))
        self.y_scaler = self.y_scaler.fit(self.y_train_transformed.values)
        self.y_train_scaled = self.y_scaler.transform(self.y_train_transformed.values)
        self.y_test_scaled = self.y_scaler.transform(self.y_test_transformed.values)

        # convert from dataframe to np array (output from MinMaxScaler is always an array)
        self.x_train_vals = self.x_train_scaled.copy()
        self.x_test_vals = self.x_test_scaled.copy()
        self.y_train_vals = self.y_train_scaled.copy()
        self.y_test_vals = self.y_test_scaled.copy()

        # convert from np array to tensor
        self.x_train_tensor = torch.tensor(self.x_train_vals).float().to(self.device)
        self.x_test_tensor = torch.tensor(self.x_test_vals).float().to(self.device)
        self.y_train_tensor = torch.tensor(self.y_train_vals).float().to(self.device)
        self.y_test_tensor = torch.tensor(self.y_test_vals).float().to(self.device)

        if self.race_idx is not None:
            
            self.y_race = DATA.loc[self.race_idx]["elapsed time (s)"]
            self.x_race = DATA.loc[self.race_idx][INPUT_COLS].values

            # transform and scale the race row
            self.x_race_transformed = pd.DataFrame()
            self.x_race_transformed["distance"]=pd.Series(self.yjpt_distance.transform(self.x_race[0].reshape(-1,1))[0][0])
            self.x_race_transformed["elevation"]=pd.Series(self.yjpt_elevation.transform(self.x_race[1].reshape(-1,1))[0][0])
            self.x_race_transformed["hr"]=pd.Series(self.x_race[2])
            self.x_race_transformed["timeago"]=pd.Series(self.yjpt_timeago.transform(self.x_race[3].reshape(-1,1))[0][0])

            self.x_race_scaled = self.x_scaler.transform(self.x_race_transformed)
            self.x_race_tensor = torch.tensor([self.x_race_scaled]).float().to(self.device)

            self.y_race_transformed = self.yjpt_time.transform(self.y_race.reshape(-1,1))[0][0]
            self.y_race_scaled = self.y_scaler.transform(self.y_race_transformed.reshape(1,-1))

    def y2minutes(self, y_pred):
        '''
        converts the predicted y-value by reverse scaling, then reverse transforming, to get the time in seconds, then converts to minutes
        '''

        if str(type(y_pred)) == "<class 'torch.Tensor'>":
            y_pred = y_pred.detach().numpy()

        y_pred = self.y_scaler.inverse_transform(y_pred)

        # detransform
        if len(y_pred) == 1:
            return self.yjpt_time.inverse_transform(y_pred.reshape(-1,1))[0][0]/60
        else:
            return self.yjpt_time.inverse_transform(y_pred.reshape(-1,1))/60


class model_handler():

    def __init__(self, model, dataobject):

        self.model = model

        self.device = dataobject.device
        self.model.to(self.device)

        self.dataobject = dataobject

        self.epochs = 1
        self.batch_size = 1
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        self.L1_lambda = 0.001

        self.train_dataset = None
        self.train_loader = None
        
        self.criterion = nn.MSELoss()

        # trackers
        self.training_losses = []
        self.testing_losses = []

        self.training_losses_mape = []
        self.testing_losses_mape = []

        self.total_epochs = 0
        self.best_mape_test = 100.0
        self.best_epoch_test = 0

        self.result = pd.DataFrame()

    def save_handler(self, save_path):

        sleep_time = 120

        # save the handler
        for _ in range(sleep_time):
            try:
                with open(save_path, 'wb') as output:
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
                break
            except PermissionError:
                print('File is busy. Trying again in 1 second...')
                sleep(1)


    def train(self, 
              epochs = None, 
              batch_size = None,
              learning_rate = None,
              weight_decay = None,
              x_train_tensor = None,
              y_train_tensor = None,
              x_test_tensor = None,
              y_test_tensor = None,
              seed = 42,
              save_best = False,
              save_path = 'model',
              EarlyStopping_Patience = None,
              factor_duplications = 0.0):
        
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if learning_rate is None:
            learning_rate = self.learning_rate
        if weight_decay is None:
            weight_decay = self.weight_decay
        if x_train_tensor is None:
            x_train_tensor = self.dataobject.x_train_tensor
        if y_train_tensor is None:
            y_train_tensor = self.dataobject.y_train_tensor
        if x_test_tensor is None:
            x_test_tensor = self.dataobject.x_train_tensor
        if y_test_tensor is None:
            y_test_tensor = self.dataobject.y_train_tensor

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_test_tensor = x_test_tensor
        self.y_test_tensor = y_test_tensor

        # increase data density at high and low distance
        if factor_duplications > 0.0:
            # get the indices of the rows
            indices = (self.x_train_tensor[:, 0] < 0.2*self.x_train_tensor[:, 0].max()) | (self.x_train_tensor[:, 0] > 0.8*self.x_train_tensor[:, 0].max())

            if torch.any(indices).item():
                # get the rows that will be duplicated
                x_rows_to_duplicate = self.x_train_tensor[indices]
                y_rows_to_duplicate = self.y_train_tensor[indices]

                # create tensors of duplicate rows
                n_duplicates = int(factor_duplications*len(x_train_tensor))
                x_duplicated_rows = torch.cat([x_rows_to_duplicate] * n_duplicates, dim=0)
                y_duplicated_rows = torch.cat([y_rows_to_duplicate] * n_duplicates, dim=0)

                # Concatenate the original tensor and the duplicate rows
                self.x_train_tensor = torch.cat((self.x_train_tensor, x_duplicated_rows), dim=0)
                self.y_train_tensor = torch.cat((self.y_train_tensor, y_duplicated_rows), dim=0)

        # Create a DataLoader for batching
        self.train_dataset = TensorDataset(self.x_train_tensor, self.y_train_tensor)
        torch.manual_seed(seed) # set Torch's random seed for batching
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # tracker for EarlyStopping
        epochs_unimproved = 0

        for epoch in range(epochs):  # loop over the dataset multiple times

            self.model.train()

            training_loss = 0 # sum of all losses of all batches (reset each epoch)
            training_loss_mape = 0

            self.total_epochs += 1

            # train on every batch
            torch.manual_seed(42)
            for inputs, labels in self.train_loader:

                # Move data to GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # start forward + backward propogation:
                outputs = self.model(inputs)

                # compute losses
                loss = self.criterion(outputs, labels) 

                # implement L1 (Lasso) regularization
                L1_norm = sum(p.abs().sum() for p in self.model.parameters())

                loss = loss + (self.L1_lambda*L1_norm)

                # compute loss function gradients
                loss.backward() 

                # update weights and biases
                self.optimizer.step() 

                training_loss += loss.item() # average training loss in that batch
                training_loss_mape += mape_actualtime(outputs, labels, self.dataobject).item()

            self.model.eval()
            with torch.no_grad(): # prevents update of optimizer  
                y_pred = self.model(self.dataobject.x_test_tensor)
                test_loss = self.criterion(y_pred, self.dataobject.y_test_tensor).item()
                test_loss_mape = mape_actualtime(y_pred, self.dataobject.y_test_tensor, self.dataobject).item()

                self.training_losses.append(training_loss/len(self.train_loader))
                self.testing_losses.append(test_loss)

                self.training_losses_mape.append(training_loss_mape/len(self.train_loader))
                self.testing_losses_mape.append(test_loss_mape)

                if test_loss_mape < self.best_mape_test:

                    # reset tracker
                    epochs_unimproved = 0

                    # log the best model and scores
                    self.best_model = copy.deepcopy(self.model)
                    self.best_mape_test = test_loss_mape
                    self.best_epoch_test = self.total_epochs

                    # replace the saved version of the file
                    if save_best:
                        self.save_handler(save_path)
                else: 
                    epochs_unimproved += 1

                # stop the training loop if unimproved some number of epochs in a row
                if epochs_unimproved == EarlyStopping_Patience:
                    self.model = copy.deepcopy(self.best_model)
                    self.total_epochs = self.total_epochs - epochs_unimproved
                    break

            print(f'Epoch {str(epoch + 1).zfill(3)} | Average training loss: {training_loss/len(self.train_loader):.5f} | Average testing loss: {test_loss:.5f}')

    def plot_training_results(self):
        
        my_interactable_xyline(y1 = self.training_losses,
                               y1_label = 'Training',
                               y2 = self.testing_losses,
                               y2_label = 'Validation',
                               y_label = 'MSE Loss',
                               x_label = 'Epochs')
        my_interactable_xyline(y1 = np.array(self.training_losses_mape)*100,
                               y1_label = 'Training',
                               y2 = np.array(self.testing_losses_mape)*100,
                               y2_label = 'Validation',
                               y_label = 'MAPE Loss (%)',
                               x_label = 'Epochs')

    def plot_traintestpredictions(self):
        for i in range(4):
            my_traintestpredictions(x_train=self.dataobject.x_train_tensor[:,i],
                                    y_train=self.dataobject.y_train_tensor,
                                    y_train_pred=self.model(self.dataobject.x_train_tensor.unsqueeze(1)).detach().numpy(),
                                    x_test=self.dataobject.x_test_tensor[:,i],
                                    y_test=self.dataobject.y_test_tensor,
                                    y_test_pred=self.model(self.dataobject.x_test_tensor.unsqueeze(1)).detach().numpy(),
                                    x_name=INPUT_COLS[i]+' Normalised',
                                    y_name='Normalised Time')

        for i in range(4):
            my_traintestpredictions(x_train=self.dataobject.x_train.values[:,i],
                                    y_train=self.dataobject.y2minutes(self.dataobject.y_train_tensor),
                                    y_train_pred=self.dataobject.y2minutes(self.model(self.dataobject.x_train_tensor.detach())),
                                    x_test=self.dataobject.x_test.values[:,i],
                                    y_test=self.dataobject.y2minutes(self.dataobject.y_test_tensor),
                                    y_test_pred=self.dataobject.y2minutes(self.model(self.dataobject.x_test_tensor.detach())),
                                    x_name=INPUT_COLS[i],
                                    y_name='Time (minutes)')

    def plot_relationships(self):
        # view the relationship with each variable
        for i in range(4):
            x = np.full(shape=(100,4), fill_value=0.5)

            x[:,i] = np.linspace(start=0.0, stop=1.0, num=len(x))
            
            pred, descaled = self.predict(x, scaled=True, return_minutes=True)

            if i == 3:
                descaled[:,i] = descaled[:,i]/3600/24
                my_xyline(x1=descaled[:,i],y1=pred, x_label="Time Ago (days)", y_label = "Time (minutes)")
            elif i == 0:
                descaled[:,i] = descaled[:,i]/1000
                my_xyline(x1=descaled[:,i],y1=pred, x_label="distance (km)", y_label = "Time (minutes)")
            else:
                my_xyline(x1=descaled[:,i],y1=pred, x_label=INPUT_COLS[i], y_label = "Time (minutes)")        
                
    def predict(self, x_raw, scaled=False, return_minutes=True):

        # transform and scale
        # expected order of x: distance, elevation, hr, timeago
        
        # convert x_raw to np.ndarray
        if isinstance(x_raw, pd.DataFrame):
            x = x_raw.values
        elif torch.is_tensor(x_raw):
            x = x_raw.numpy()
        if isinstance(x_raw, list) or isinstance(x_raw, tuple):
            if len(x_raw) != 4:
                raise ValueError("If providing a list or tuple, it must be length 4")
            else:
                x = np.array(list(x_raw)).reshape(1,4)
        elif isinstance(x_raw, np.ndarray):
            if len(x_raw.shape) == 1:
                x = x_raw.copy().reshape(1,-1)
            else:
                x = x_raw.copy()
        
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Input data type must be either (1) np.ndarray (2) pd.Dataframe (3) torch.tensor (4) list (5) tuple.")
        elif x.shape[1] != 4:
            raise ValueError("x must be shape (N,4) where N is the number of points")

        if not scaled:
            # detransform each column with the yjpt scaler
            x[:,0] = self.dataobject.yjpt_distance.transform(x[:,0].reshape(-1,1)).reshape(-1)
            x[:,1] = self.dataobject.yjpt_elevation.transform(x[:,1].reshape(-1,1)).reshape(-1)
            x[:,3] = self.dataobject.yjpt_timeago.transform(x[:,3].reshape(-1,1)).reshape(-1)

            # descale each column with the scaler
            x = self.dataobject.x_scaler.transform(pd.DataFrame(x, columns = self.dataobject.x_test_transformed.columns[:4]))

        else:
            x_descaled = x.copy()

            # scale each column with the scaler
            x_descaled = self.dataobject.x_scaler.inverse_transform(pd.DataFrame(x_descaled, columns = self.dataobject.x_test_transformed.columns))

            # transform each column with the yjpt scaler
            x_descaled[:,0] = self.dataobject.yjpt_distance.inverse_transform(x_descaled[:,0].reshape(-1,1)).reshape(-1)
            x_descaled[:,1] = self.dataobject.yjpt_elevation.inverse_transform(x_descaled[:,1].reshape(-1,1)).reshape(-1)
            x_descaled[:,3] = self.dataobject.yjpt_timeago.inverse_transform(x_descaled[:,3].reshape(-1,1)).reshape(-1)
            
        # convert to tensor
        x_tensor = torch.tensor(x).float()

        y_tensor = self.model(x_tensor)

        if return_minutes:
            if scaled:
                return self.dataobject.y2minutes(y_tensor), x_descaled
            else:
                return self.dataobject.y2minutes(y_tensor)
        else:
            if scaled:
                return y_tensor.detach().numpy(), x_descaled
            else:
                return y_tensor.detach().numpy()


class LeakyReLU_NN(nn.Module):
    def __init__(self, hidden_nodes, seed=12):

        torch.manual_seed(seed)

        super().__init__()
        self.hidden1 = nn.Linear(4,hidden_nodes)
        self.actv1 = nn.LeakyReLU()
        self.hidden2 = nn.Linear(hidden_nodes,hidden_nodes)
        self.actv2 = nn.LeakyReLU()
        self.output = nn.Linear(hidden_nodes,1)
        self.actv_output = nn.LeakyReLU()

    def forward(self, x):
        x = self.actv1(self.hidden1(x))
        x = self.actv2(self.hidden2(x))
        x = self.actv_output(self.output(x))
        
        return x


class ELU_NN(nn.Module):
    def __init__(self, hidden_nodes, seed=12):

        torch.manual_seed(seed)

        super().__init__()
        self.hidden1 = nn.Linear(4,hidden_nodes)
        self.actv1 = nn.ELU()
        self.hidden2 = nn.Linear(hidden_nodes,hidden_nodes)
        self.actv2 = nn.ELU()
        self.output = nn.Linear(hidden_nodes,1)
        self.actv_output = nn.ELU()

    def forward(self, x):
        x = self.actv1(self.hidden1(x))
        x = self.actv2(self.hidden2(x))
        x = self.actv_output(self.output(x))
        
        return x

    
class find_hn_epochs():
    def __init__(self, model_class, start_hn, stop_hn, dataobject):

        self.model_class = model_class

        self.handler = None

        self.dataobject = dataobject

        self.hidden_nodes = np.arange(start=start_hn,stop=stop_hn,step=1).tolist()
        self.mape_bestlosses = []
        self.mape_bestepochs = []

        self.results = pd.DataFrame()

        self.best_testing_losses_mape = 1.0

    def find(self, save_best = False, target_folder = None):

        # make a folder to save the progress in
        if target_folder is not None:
            sleep_time = 120
            if os.path.exists(target_folder):
                for _ in range(sleep_time):
                    try:
                        shutil.rmtree(target_folder)
                        break
                    except PermissionError:
                        print('Folder is busy. Trying again in 1 second...')
                        sleep(1)
            os.mkdir(target_folder)
        else:
            target_folder = ''

        count = 0
        st = time()
        for hn in self.hidden_nodes:
            self.handler = model_handler(model = self.model_class(hn, seed = 42), dataobject=self.dataobject)

            save_path = target_folder + f"/handler_{hn}hn.pkl"

            self.handler.train(
                epochs = 512, 
                batch_size=20, 
                learning_rate=0.001, 
                save_best=save_best,
                save_path=save_path,
                EarlyStopping_Patience=25,
                factor_duplications=0.1
                )

            self.mape_bestlosses.append(min(self.handler.testing_losses_mape))
            self.mape_bestepochs.append(np.argmin(self.handler.testing_losses_mape))

            clear_output(wait=True)
            count += 1
            et = time()-st
            pc = count/len(self.hidden_nodes)
            etr = et/pc * (1-pc)
            print(f'{pc*100:.1f}% complete | Estimated time remaining = {etr/60:.2f} minutes')

        self.results = pd.DataFrame(columns=["hn","epochs","loss"])
        self.results["hn"] = np.array(self.hidden_nodes)
        self.results["epochs"] = np.array(self.mape_bestepochs)
        self.results["loss"] = np.array(self.mape_bestlosses)
        self.results = self.results.sort_values("hn")
    
    def plot_result(self):
        my_xyline(x1=self.results["hn"],y1=self.results["loss"],x_label='Number of Hidden Nodes', y_label='Best Testing Loss (MAPE)')


def yj_transform(train_set, test_set=None):
    '''
    Uses Yeo-Johnson to scale the train and test data sets according to the distribution of the train dataset only. 
    
    Returns the power transformer fitted for the train set, transformed train set, and transformed test set if a test set was provided
    '''

    yjpt = PowerTransformer(method='yeo-johnson')

    train_set_transformed = yjpt.fit_transform(train_set.values.reshape(-1,1))
    train_set_transformed = train_set_transformed.flatten()

    if test_set is not None:
        test_set_transformed = yjpt.transform(test_set.values.reshape(-1,1))

        test_set_transformed = test_set_transformed.flatten()

        return yjpt, train_set_transformed, test_set_transformed
    
    else:

        return yjpt, train_set_transformed      

def mape_actualtime(pred_tensor, true_tensor, dataobject):
    '''
    Uses MAPE on the descaled and detransformed time values
    '''

    try: # CPU
        pred_seconds = torch.tensor(dataobject.y2minutes(pred_tensor), requires_grad=True).float()
        true_seconds = torch.tensor(dataobject.y2minutes(true_tensor), requires_grad=True).float()
    except TypeError: # GPU
        pred_seconds = torch.tensor(dataobject.y2minutes(pred_tensor.cpu()), requires_grad=True).float()
        true_seconds = torch.tensor(dataobject.y2minutes(true_tensor.cpu()), requires_grad=True).float()

    # return torch.sum(torch.abs((pred_seconds-true_seconds)/(true_seconds)))/true_seconds.size(0)
    return torch.sum(torch.abs((pred_seconds-true_seconds)/(true_seconds)))/true_seconds.numel()

def prepare_new_data(base_dataobject, new_dataobject):
    '''
    Shifts the timeago of the new_dataobject so that the newest run (or race) of the new_dataobject has the same timeago as the newest run of the base_dataobject.

    The fitted transforms and scalers of the base_dataobject will be used to transform and scale the data of the new_dataobject
    '''

    delta_timeago = new_dataobject.X["time ago (s)"].min() - base_dataobject.X["time ago (s)"].min()

    prepared_new_X = new_dataobject.X[['distance (m)','elevation gain (m)','average heart rate (bpm)','time ago (s)']]

    # offset the timeago
    prepared_new_X.loc[:, "time ago (s)"] = prepared_new_X["time ago (s)"] - delta_timeago


    # transform the x data
    new_X_transformed = pd.DataFrame()
    new_X_transformed["distance"] = base_dataobject.yjpt_distance.transform(prepared_new_X["distance (m)"].values.reshape(-1,1)).reshape(-1)
    new_X_transformed["elevation"] = base_dataobject.yjpt_elevation.transform(prepared_new_X["elevation gain (m)"].values.reshape(-1,1)).reshape(-1)
    new_X_transformed["hr"] = prepared_new_X["average heart rate (bpm)"].values
    new_X_transformed["timeago"] = base_dataobject.yjpt_timeago.transform(prepared_new_X["time ago (s)"].values.reshape(-1,1)).reshape(-1)

    # scale the x data
    new_X_scaled = base_dataobject.x_scaler.transform(new_X_transformed) # np.ndarray

    # transform and scale the y data
    new_y_transformed = base_dataobject.yjpt_time.transform(new_dataobject.y.values.reshape(-1,1))
    new_y_scaled = base_dataobject.y_scaler.transform(new_y_transformed) # np.ndarray

    # split the data
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_X_scaled, new_y_scaled, test_size = 0.1, random_state=30)

    # convert to tensors
    new_x_train_tensor = torch.tensor(new_x_train).float().to(base_dataobject.device)
    new_x_test_tensor = torch.tensor(new_x_test).float().to(base_dataobject.device)
    new_y_train_tensor = torch.tensor(new_y_train).float().to(base_dataobject.device)
    new_y_test_tensor = torch.tensor(new_y_test).float().to(base_dataobject.device)

    return new_x_train_tensor, new_x_test_tensor, new_y_train_tensor, new_y_test_tensor


def update_handler_tracker(target_suffix: str):
    '''
    Stores the parameters of the handlers in a JSON
    '''

    target_folder = f'base_models_{target_suffix}'
    target_json = f'{target_folder}/handler_tracker_{target_suffix}.json'
    target_csv = f'{target_folder}/handler_tracker_{target_suffix}.csv'

    # load the tracker json as a dict
    with open(target_json, 'rb') as input:
        handler_tracker = json.load(input)

    # write a csv tracker 
    df = pd.DataFrame(columns = ['model', 'athletes', 'train_mse', 'test_mse', 'mape_train', 'mape_test', 'total_epochs'])

    # initialise a list of the athletes included in the dataset
    athletes_completed = []

    # go through each base model
    for filename in os.listdir(target_folder):
        if filename.endswith('.pkl'):
            with open(f'{target_folder}/{filename}', 'rb') as input:
                temp_handler = pickle.load(input)

            try:
                handler_tracker[filename]
            except KeyError:
                # create a new field if it's not in the tracker
                handler_tracker[filename] = dict()

            # log which athlete this model includes
            handler_tracker[filename]['athlete'] = str(temp_handler.dataobject.athlete)
            handler_tracker[filename]['best_mape_test'] = temp_handler.best_mape_test
            handler_tracker[filename]['total_epochs'] = temp_handler.total_epochs

            # add the athlete to the list
            athletes_completed += temp_handler.dataobject.athlete

            # update csv tracker
            row = pd.DataFrame({
                'model':[filename], 
                'athletes': [str(temp_handler.dataobject.athlete)[1:-1]], 
                'train_mse': [temp_handler.training_losses[-1]], 
                'test_mse': [temp_handler.testing_losses[-1]], 
                'mape_train': [temp_handler.training_losses_mape[-1]], 
                'mape_test': [temp_handler.testing_losses_mape[-1]], 
                'total_epochs': [temp_handler.total_epochs]
                })

            df = pd.concat([df, row], ignore_index=True)
            
    # add the list as a new field
    handler_tracker['athletes_completed'] = str(athletes_completed)

    handler_tracker = dict(sorted(handler_tracker.items()))

    # save the dict as a json
    with open(target_json, 'w') as output:
        json.dump(handler_tracker, output, indent=4)


    # save the df as csv
    df.to_csv(target_csv, index=False)

    return handler_tracker

def read_strava_csv(filepath=r'..\my_strava_data\activities.csv'):

    user_df = pd.read_csv(filepath)

    # extract the relevant columns
    user_df = user_df[user_df['Activity Type']=='Run'][['Moving Time', 'Distance', 'Elevation Gain', 'Average Heart Rate', 'Activity Date']]

    # rename the columns
    new_column_names = {
        "Moving Time": "elapsed time (s)",
        "Distance": "distance (m)",
        "Elevation Gain": "elevation gain (m)",
        "Average Heart Rate": "average heart rate (bpm)",
        "Activity Date": "timestamp"
    }
    user_df = user_df.rename(columns=new_column_names)

    # drop activities with missing heart rate
    user_df = user_df.dropna(subset=['average heart rate (bpm)'])

    # convert from km to m
    user_df["distance (m)"] = user_df["distance (m)"]*1000

    # convert dates to datetime objects
    user_df['timestamp'] = pd.to_datetime(user_df['timestamp'])

    return user_df

def time_before_race(user_df, race_info):
    '''
    Creates the time ago column using the race date as the datum
    '''

    race_date = pd.to_datetime(race_info['timestamp'])[0]

    user_df['time ago (s)'] = race_date - user_df["timestamp"]

    user_df['time ago (s)'] = user_df['time ago (s)'].apply(lambda x: x.total_seconds())

    user_df = user_df[['elapsed time (s)', 'distance (m)','elevation gain (m)','average heart rate (bpm)','time ago (s)']]

    return user_df

def prepare_user_data(base_dataobject, user_df, race_info):
    '''
    Shifts the timeago of the user data so that the user's race has the same timeago as the newest run of the base_dataobject.

    The fitted transforms and scalers of the base_dataobject will be used to transform and scale the user data
    '''

    # create the time ago column using the race date as the datum
    user_df = time_before_race(user_df, race_info)

    # assign a time ago value to the race info row (time ago is 0)
    race_info_ta = race_info.copy()
    race_info_ta['time ago (s)'] = 0.0

    delta_timeago = 0.0 - base_dataobject.X["time ago (s)"].min()

    x_offset = user_df[['distance (m)','elevation gain (m)','average heart rate (bpm)','time ago (s)']]

    x_offset = pd.concat([x_offset, race_info_ta[['distance (m)','elevation gain (m)','average heart rate (bpm)', 'time ago (s)']]])

    # offset the timeago
    x_offset.loc[:, "time ago (s)"] = x_offset["time ago (s)"] - delta_timeago

    # transform the x data
    x_transformed = pd.DataFrame()
    x_transformed["distance"] = base_dataobject.yjpt_distance.transform(x_offset["distance (m)"].values.reshape(-1,1)).reshape(-1)
    x_transformed["elevation"] = base_dataobject.yjpt_elevation.transform(x_offset["elevation gain (m)"].values.reshape(-1,1)).reshape(-1)
    x_transformed["hr"] = x_offset["average heart rate (bpm)"].values
    x_transformed["timeago"] = base_dataobject.yjpt_timeago.transform(x_offset["time ago (s)"].values.reshape(-1,1)).reshape(-1)

    # scale the x data
    x_scaled = base_dataobject.x_scaler.transform(x_transformed) # np.ndarray

    # transform and scale the y data
    y_transformed = base_dataobject.yjpt_time.transform(user_df["elapsed time (s)"].values.reshape(-1,1))
    user_y_scaled = base_dataobject.y_scaler.transform(y_transformed) # np.ndarray

    # extract the race row
    x_race_tensor = torch.tensor(x_scaled[-1,:].reshape(1,-1)).float().to(base_dataobject.device)
    y_race_tensor = torch.tensor(user_y_scaled[-1,:].reshape(1,-1)).float().to(base_dataobject.device)

    # drop the race row
    x_scaled = x_scaled[:-1,:]

    # split the data
    user_x_train, user_x_test, user_y_train, user_y_test = train_test_split(x_scaled, user_y_scaled, test_size = 0.1, random_state=30)

    # convert to tensors
    x_train_tensor = torch.tensor(user_x_train).float().to(base_dataobject.device)
    x_test_tensor = torch.tensor(user_x_test).float().to(base_dataobject.device)
    y_train_tensor = torch.tensor(user_y_train).float().to(base_dataobject.device)
    y_test_tensor = torch.tensor(user_y_test).float().to(base_dataobject.device)

    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, x_race_tensor, y_race_tensor

def get_best_model(user_df, race_info, model_folder = 'base_models_LeakyReLU'):
    '''
    Returns 
    1. The model handler with the best MAPE loss on the test dataset.
    2. The train and test tensors, transformed and scaled by the best model handler.
    3. MAPE results across all models.
    '''

    result = pd.DataFrame({
        "handler": [],
        "mape": [],
        "x_train_tensor": [],
        "x_test_tensor": [],
        "y_train_tensor": [],
        "y_test_tensor": [],
        "x_race_tensor": [],
        "y_race_tensor": []
    })

    for filename in os.listdir(model_folder):

        if filename.endswith('.pkl'):
            with open(f'{model_folder}/{filename}', 'rb') as input:
                base_handler = pickle.load(input)

        x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, x_race_tensor, y_race_tensor = prepare_user_data(base_handler.dataobject, user_df, race_info)


        # test the MAPE on the test data
        mape = mape_actualtime(pred_tensor = base_handler.model(x_test_tensor),
                               true_tensor = y_test_tensor,
                               dataobject = base_handler.dataobject).detach().numpy().item()

        result = pd.concat([result,pd.DataFrame({
            "handler": [base_handler], 
            "mape": [mape],
            "x_train_tensor": [x_train_tensor],
            "x_test_tensor": [x_test_tensor],
            "y_train_tensor": [y_train_tensor],
            "y_test_tensor": [y_test_tensor],
            "x_race_tensor": [x_race_tensor],
            "y_race_tensor": [y_race_tensor]      
            })])

    best_idx = result["mape"].argmin()
    handler = result.iloc[best_idx]["handler"]
    x_train_tensor = result.iloc[best_idx]["x_train_tensor"]
    x_test_tensor = result.iloc[best_idx]["x_test_tensor"]
    y_train_tensor = result.iloc[best_idx]["y_train_tensor"]
    y_test_tensor = result.iloc[best_idx]["y_test_tensor"]
    x_race_tensor = result.iloc[best_idx]["x_race_tensor"]
    y_race_tensor = result.iloc[best_idx]["y_race_tensor"]

    return handler, x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, x_race_tensor, y_race_tensor, result


def user_finetune(base_handler, 
                  x_train_tensor, 
                  x_test_tensor, 
                  y_train_tensor, 
                  y_test_tensor,
                  epochs = 512,
                  batch_size = 20):

    # copy the base model
    ft_model = copy.deepcopy(base_handler.model)

    # # freeze model parameters to prevent backpropagation 
    for param in ft_model.parameters():
        param.requires_grad = False

    # # Unfreeze the last layer
    for param in ft_model.output.parameters():
        param.requires_grad = True
    # for param in ft_model.actv2.parameters():
    #     param.requires_grad = True
    # for param in ft_model.actv1.parameters():
    #     param.requires_grad = True

    # # reset the last layer
    # ft_model.output = nn.Linear(
    #     ft_model.output.in_features,
    #     ft_model.output.out_features
    #     )
    
    ft_handler = model_handler(model = ft_model, 
                               dataobject = base_handler.dataobject)
    
    ft_handler.train(epochs = epochs,
                     batch_size = batch_size,
                     x_train_tensor = x_train_tensor,
                     y_train_tensor = y_train_tensor,
                     x_test_tensor  = x_test_tensor,
                     y_test_tensor  = y_test_tensor,
                     EarlyStopping_Patience=100,
                     factor_duplications=0.1)

    return ft_handler
