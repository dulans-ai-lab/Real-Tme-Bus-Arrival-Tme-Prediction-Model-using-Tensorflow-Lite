import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopy.distance
import re

dataset = pd.read_csv('../Dataset/GPS SHEET - my_locations.csv')
dataset['date_time'] = dataset['Date'].str.cat(dataset['Time'], sep=" ")
# dataset['distance'] = dataset['Date'].str.cat(dataset['Time'], sep=" ")

len = dataset.shape[0]
hours = np.zeros((len), dtype=float)
hours = np.array(hours)
minitues = np.zeros((len), dtype=float)
minitues = np.array(minitues)
seconds = np.zeros((len), dtype=float)
seconds = np.array(seconds)

for row in range(len):
    time = dataset.iloc[row, 2]
    temp = time.split(":")
    hours[row] = temp[0]
    minitues[row] = temp[1]
    seconds[row] = temp[2]

dataset['Hours'] = hours
dataset['Minitues'] = minitues
dataset['Seconds'] = seconds

dataset = dataset.drop(['Date', 'Time', 'Start/End'], axis=1)

dataset.set_index("date_time", inplace=True)

# for correct missing data
# dataset.replace('?', np.nan, inplace=True)

# to set all data to float64
# dataset=dataset.astype('float')

# no_of_nan=np.isnan(dataset).sum()
'''
def fill_missing_data(data):
#     location dekaka middle point eka denna oona.. eka balanna wenooo poddak
'''

# dataset = np.array(dataset)
len = dataset.shape[0]
distance = np.zeros((len), dtype=float)
distance = np.array(distance)

#
# for row in range(len - 1):
#     lon = dataset.iloc[row, 8]
#     lat = dataset.iloc[row, 9]
#     # if lon!='nan' & lat!='nan':
#     coords_1 = (lon, lat)
#     # if lon != 'nan' & lat != 'nan':
#     lon2 = dataset.iloc[row + 1, 8]
#     lat2 = dataset.iloc[row + 1, 9]
#     coords_2 = (lon2, lat2)
#
#     dis_km = geopy.distance.vincenty(coords_1, coords_2)
#     dis_m = dis_km * 1000.0
#     dis_m = dis_m.__str__()
#     dis_m_without_km = re.sub(r'[a-z]+', '', dis_m)
#     dis_m_without_km = float(dis_m_without_km)
#     distance[row] = float(dis_m_without_km)
#     # distance = distance.append(dis_m)
#     np.append(distance, [dis_m])
#     print(geopy.distance.vincenty(coords_1, coords_2) * 1000)

dataset = pd.DataFrame(dataset)
# dataset['Distance'] = distance
#
# coords_1 = (8.354721958, 80.50154775)
# coords_2 = (8.354866522, 80.50206427)
# print(geopy.distance.vincenty(coords_1, coords_2) * 1000)
#
# coords_1 = (8.35347, 80.42224)
# coords_2 = (8.35546, 80.42512)
# print(geopy.distance.vincenty(coords_1, coords_2) * 1000)

dataset.to_csv('../Dataset/GPS Database Cleaned Data-GPS SHEET.csv')
