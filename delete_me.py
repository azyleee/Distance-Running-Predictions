# transform the variables wrt athlete 19023831 only
from sklearn.model_selection import train_test_split
import torch

input_cols = ["distance (m)", "elevation gain (m)", "average heart rate (bpm)", "time ago (s)"]
# input_cols = ["distance (m)", "elevation gain (m)", "average heart rate (bpm)"]


class athlete_data():

    def __init__(self, athlete, race_idx):

        self.athlete = athlete
        self.race_idx = race_idx

        self.y_race = data.loc[self.race_idx]["elapsed time (s)"]
        self.x_race = data.loc[self.race_idx][input_cols].values

        # extract the rows for the chosen athlete
        self.X = data[data["athlete"]==ath_mostruns][input_cols+["pace (min/km)"]].drop(self.race_idx)
        self.y = data[data["athlete"]==ath_mostruns]["elapsed time (s)"].drop(self.race_idx)

        # split the train/test data
        self.x_train_, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state=30)

        self.x_train = self.x_train_[input_cols]

        # initialise dataframes to store train/test data
        self.x_train_scaled = pd.DataFrame()
        self.y_train_scaled = pd.DataFrame()
        self.x_test_scaled = pd.DataFrame()
        self.y_test_scaled = pd.DataFrame()
        self.x_train_transformed = pd.DataFrame()
        self.y_train_transformed = pd.DataFrame()

        # transform+scale the data, store the transformer objects
        self.yjpt_pace, self.pace_scaled, self.pace_tranformed = yj_transform(self.x_train_["pace (min/km)"])

        self.yjpt_distance, self.x_train_scaled["distance"], self.x_test_scaled["distance"], self.x_train_transformed["distance"], _ = yj_transform(self.x_train["distance (m)"], self.x_test["distance (m)"])

        self.yjpt_elevation, self.x_train_scaled["elevation"], self.x_test_scaled["elevation"], self.x_train_transformed["elevation"], _ = yj_transform(self.x_train["elevation gain (m)"], self.x_test["elevation gain (m)"])
        
        hr_train, hr_test = minmax_scale(self.x_train["average heart rate (bpm)"], self.x_test["average heart rate (bpm)"])
        self.x_train_transformed["hr"] = pd.Series(self.x_train["average heart rate (bpm)"].values)
        self.x_train_scaled["hr"], self.x_test_scaled["hr"] = hr_train.values, hr_test.values

        self.yjpt_timeago, self.x_train_scaled["timeago"], self.x_test_scaled["timeago"], self.x_train_transformed["timeago"], _ = yj_transform(self.x_train["time ago (s)"], self.x_test["time ago (s)"])

        self.yjpt_time, self.y_train_scaled["time"], self.y_test_scaled["time"], self.y_train_transformed["time"], _ = yj_transform(self.y_train, self.y_test)


        # convert from dataframe to np array
        self.x_train_vals = self.x_train_scaled.values
        self.x_test_vals = self.x_test_scaled.values
        self.y_train_vals = self.y_train_scaled.values
        self.y_test_vals = self.y_test_scaled.values

        # convert from np array to tensor
        self.x_train_tensor = torch.tensor(self.x_train_vals).float()
        self.x_test_tensor = torch.tensor(self.x_test_vals).float()
        self.y_train_tensor = torch.tensor(self.y_train_vals).float()
        self.y_test_tensor = torch.tensor(self.y_test_vals).float()

        # transform and scale the race row
        self.x_race_transformed = pd.DataFrame()
        self.x_race_transformed["distance"]=pd.Series(self.yjpt_distance.transform(self.x_race[0].reshape(-1,1))[0][0])
        self.x_race_transformed["elevation"]=pd.Series(self.yjpt_elevation.transform(self.x_race[1].reshape(-1,1))[0][0])
        self.x_race_transformed["hr"]=pd.Series(self.x_race[2])
        self.x_race_transformed["timeago"]=pd.Series(self.yjpt_timeago.transform(self.x_race[2].reshape(-1,1))[0][0])

        # if self.x_race_transformed["hr"]
        # self.x_race_transformed = np.array([
        #     self.yjpt_distance.transform(self.x_race[0].reshape(-1,1))[0][0],
        #     self.yjpt_elevation.transform(self.x_race[1].reshape(-1,1))[0][0],
        #     self.x_race[2],
        #     self.yjpt_timeago.transform(self.x_race[2].reshape(-1,1))[0][0],
        #     ])
        _, self.x_race_scaled = minmax_scale(self.x_train_transformed, self.x_race_transformed)
        self.x_race_tensor = torch.tensor([self.x_race_scaled.values]).float()

        self.y_race_transformed = self.yjpt_time.transform(self.y_race.reshape(-1,1))[0][0]
        _, self.y_race_scaled = minmax_scale(self.y_train, self.y_race_transformed)

    def y2minutes(self, y_pred):
        '''
        converts the predicted y-value by reverse scaling, then reverse transforming, to get the time in seconds, then converts to minutes
        '''

        if str(type(y_pred)) == "<class 'torch.Tensor'>":
            y_pred = y_pred.detach().numpy()

        # if str(type(y_pred)) != "numpy.ndarray":
        #     y_pred = np.array([y_pred])

        # descale
        y_pred = y_pred * (self.y_train_transformed.max().values[0] - self.y_train_transformed.min().values[0]) + self.y_train_transformed.min().values[0]

        # detransform
        if len(y_pred) == 1:
            return self.yjpt_time.inverse_transform(y_pred.reshape(-1,1))[0][0]/60
        else:
            return self.yjpt_time.inverse_transform(y_pred.reshape(-1,1))/60
        

data19023831 = athlete_data(athlete = ath_mostruns, race_idx = race_idx_19023831)