import numpy as np
from matplotlib import pyplot as pl
import pandas as pd
import os


class RunImport():
    def __init__(self, speed_outlier, slope_outlier, time_period, segment_length, average_speed_th):
        self.speed_outlier = speed_outlier
        self.slope_outlier = slope_outlier
        self.time_period = time_period
        self.segment_length = segment_length
        self.average_speed_th = average_speed_th
    
    def import_path(self, path):
        # Dataset creation as a list of dataframe (race)
        tab_files = [f for f in os.listdir(path) if f[-3:] == 'tab']
        
        if len(tab_files) > 0:
            self._data_removed = 0
            race_number = 0
            races_removed = 0

            dataset = None

            for file_name in tab_files:
                print('processing', file_name)
                data = pd.read_table(os.path.join(path, file_name), header=0)
                data = self._initialize_data(data)

                #add the race number
                new_col = [race_number]*data.shape[0]
                data = data.assign(race=new_col)

                if dataset is None:
                    dataset = data
                else:
                    average_speed = data['distance'].max() / data['time'].max()
                    if average_speed > self.average_speed_th:
                        dataset = dataset.append(data)
                        race_number += 1
                    else:
                        print('\nRace', race_number, 'ignored. Average speed:', average_speed)
                        races_removed += 1
            
            
            print(len(tab_files), 'files read\n')
            print('Dataset shape:', dataset.shape, '\n')
            print('Data removed from dataset (outliers):', self._data_removed, '\n')
            print('Races removed (average speed):', races_removed, '\n')
            print('Dataset statistics:')
            display(dataset.describe())
            print('\nDataset sample:')
            display(dataset.head().append(dataset.tail()))
            
            return dataset
        else:
            print('Invalid path. Missing .tab files')
            return None
    
    # initialize de dataset (call of each utility fonctions)
    def _initialize_data(self, data):
        #dont change the order !
        #remove NA values and reset index
        new_data = data.dropna(axis=0, how='any')
        new_data = self._calculate_slope(new_data)
        new_data = self._filter_first_zeros(new_data)
        new_data = self._convert_date(new_data)
        new_data = self._filter_outlier(new_data)
        #new_data = self._average_over_period(new_data)
        new_data = self._average_over_segment(new_data)
        
        return new_data.reset_index(drop=True)
        
    # Calculate slope from delta(elevation) / delta(distance) *100
    def _calculate_slope(self, dataset):
        slope_array = [0.0] #first slope value is 0
        for i in range(1, dataset.shape[0]):
            delta_e = dataset['elevation'].iloc[i] - dataset['elevation'].iloc[i-1]
            delta_d = dataset['distance'].iloc[i] - dataset['distance'].iloc[i-1]
            if (delta_d == 0):
                # set slope to 0 if distance is 0
                slope_array.append(0.0)
            else:
                slope_array.append((delta_e / delta_d) * 100)
        
        return dataset.assign(slope=slope_array)

    #remove all the first values when speed and distance = 0 except one (consecutive 0s means the race hasn't started yet)
    def _filter_first_zeros(self, dataset):
        zeros = dataset.loc[(dataset['speed'] == 0) &
                            (dataset['distance'] == 0)]
        self._data_removed += len(zeros.index[:-1])
        
        return dataset.drop(index=zeros.index[:-1])

    # Take time colunm as input and convert date into seconds (first date = 0s)
    def _convert_date(self, dataset):
        dates = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

        delta = dates.iloc[0] #first date = 0s
        f = lambda x: (x - delta).seconds #convert pandas Timedelta to seconds
        new_dates = dates.map(f) #map function on every elements
        new_dates = pd.DataFrame({'time': new_dates}) #convert Serie to Dataframe

        dataset.update(new_dates) #update the values
        
        return dataset

    # Filter the outliers on the dataset (ex: impossible speed, slope, etc.)
    def _filter_outlier(self, data):

        # we define outlier as speed > 30km/h or slope > +-80%
        outliers = data.loc[(data['speed'] > self.speed_outlier) | 
                            (data['slope'] > self.slope_outlier) |
                            (data['slope'] < -self.slope_outlier)]
        self._data_removed += len(outliers.index)
        
        return data.drop(index=outliers.index)

    # average value over a period of time
    def _average_over_period(self, data):
        warning = True
        series_list = [] #list containing the pd.Series containing the mean value
        
        last_time = int(np.ceil(data['time'].iloc[-1] / self.time_period))
        for i in range(last_time):
            #extract all the values between the period
            tmp = data.loc[(data['time'] >= i*self.time_period) & 
                           (data['time'] < (i+1)*self.time_period)]
            
            if tmp.empty:
                if warning:
                    print('WARNING: time gap present in file !')
                    warning = False
                continue
                
            # average columns values
            serie = tmp.mean(axis=0)
            #replace the average time by the last time (from this period)
            serie[0] = tmp['time'].iloc[-1]
            series_list.append(serie)

        return pd.DataFrame(series_list)

    # average value over a segment
    def _average_over_segment(self, data):
        warning = True
        series_list = [] #list containing the pd.Series containing the mean value
        
        n_segments = int(np.ceil(data['distance'].iloc[-1] / self.segment_length))
        for i in range(n_segments):
            #extract all the values in the segment
            tmp = data.loc[(data['distance'] >= i*self.segment_length) & 
                           (data['distance'] < (i+1)*self.segment_length)]
            
            if tmp.empty:
                if warning:
                    print('WARNING: gap present in file !')
                    warning = False
                continue
                
            # average columns values
            serie = tmp.mean(axis=0)
            #replace the average time by the last time (from this segment)
            serie['time'] = tmp['time'].iloc[-1]
            #replace the average distance by the last distance (from this segment)
            serie['distance'] = tmp['distance'].iloc[-1]
            series_list.append(serie)

        return pd.DataFrame(series_list)
      

def plot_race(dataset, race_number):
    race = dataset.loc[dataset['race'] == race_number]
    
    fig, axarr = pl.subplots(2, sharex=True, figsize=(16,9))
    #first subplot
    #first axe
    axarr[0].plot(race['time'].values, race['speed'].values, 'darkgreen', label='speed')
    axarr[0].set_ylabel('Running speed [m/s]', color='darkgreen', fontsize=12)
    #second axe
    ax2 = axarr[0].twinx() #duplicate axe 
    ax2.plot(race['time'].values, race['slope'].values, 'darkblue', label='slope')
    ax2.set_ylabel('Land slope [%]', color='darkblue', fontsize=12)
    
    axarr[0].set_title('Speed and slope')
    h1, l1 = axarr[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axarr[0].legend(h1+h2, l1+l2, loc='upper left', shadow=True)

    #second subplot
    #first axe
    axarr[1].plot(race['time'].values, race['elevation'].values, 'darkmagenta', label='elevation')
    axarr[1].set_ylabel('Elevation [m]', color='darkmagenta', fontsize=12)
    #second axe
    ax2 = axarr[1].twinx()
    ax2.plot(race['time'].values, race['distance'].values, 'teal', label='distance')
    ax2.set_ylabel('Distance traveled [m]', color='teal', fontsize=12)
    
    axarr[1].set_title('Elevation and distance')
    h1, l1 = axarr[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axarr[1].legend(h1+h2, l1+l2, loc='upper left', shadow=True)
    
    ax2.set_xlabel('Time [s]')
    axarr[1].set_xlabel('Time [s]')
    
    pl.suptitle('Features from race n°' + str(race_number), fontsize=16)
    #pl.xlabel('Time [s]')
    pl.xlim(xmin=0, xmax=max(race['time'].values))
    pl.xticks(range(0, int(max(race['time'].values)), 5*60)) #1 tick every 5 minutes
    #pl.show()
