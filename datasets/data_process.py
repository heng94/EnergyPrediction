import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# holiday2num = {
#     'NoHoliday': 0,
#     'TombSweepingDay': 1,
#     'InternationalLabourDays': 2,
#     'MidAutumnFestival': 3,
#     'DragonBoatFestival': 4,
#     'SpringFestival': 5, 
#     'NationalDay': 6, 
#     'NewYearsDay': 7, 
# }

# seaon2num = {
#     'Spring': 0,
#     'Summer': 1,
#     'Autumn': 2,
#     'Winter': 3,
# }

# workday2num = {
#     'WorkingDay': 0,
#     'NonWorkingDay': 1,
# }   

# str_data = [
#     'NoHoliday', 'TombSweepingDay', 'InternationalLabourDays', 'MidAutumnFestival', 'DragonBoatFestival', 
#     'SpringFestival', 'NationalDay', 'NewYearsDay', 'Spring', 'Summer', 'Autumn', 'Winter', 
#     'WorkingDay', 'NonWorkingDay'
# ]

# label_encoder = LabelEncoder()
# encoded_data = label_encoder.fit_transform(str_data)

# mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# # print(mapping)
# df = pd.read_csv('../data/area_0.csv')

# for idx in range(df.shape[0]):
#     df.loc[idx, 'holiday'] = mapping[df.loc[idx, 'holiday']]
#     df.loc[idx, 'season'] = mapping[df.loc[idx, 'season']]
#     df.loc[idx, 'workday'] = mapping[df.loc[idx, 'workday']]
# df.to_csv('../data/area_0_v3.csv', index=False)
# print('1')

# check correlation
# feature: 
# data,day,hour,day_of_year,week,month,days_in_month,day_of_week,season,holiday,workday,
# sea_pressure,ground_pressure,air_temp,air_max,air_min,air_mean,rain,snow,ground_temp,
# ground_max,ground_min,ground_mean,body_temp,body_max,body_min,body_mean,damp,
# evaporation,prevaporation,north_wind,south_wind,low_cloud,middle_cloud,high_cloud,
# total_cloud,sun_intensity,total_suninensity,uv_intensity

df = pd.read_csv('../data/area_0_v3.csv')
time_features = [
    'day','hour','day_of_year','week','month','days_in_month','day_of_week','season','holiday','workday',
]

weather_features = [
    'sea_pressure','ground_pressure','air_temp','air_max','air_min','air_mean','rain','snow','ground_temp',
    'ground_max','ground_min','ground_mean','body_temp','body_max','body_min','body_mean','damp',
    'evaporation','prevaporation','north_wind','south_wind','low_cloud','middle_cloud','high_cloud',
    'total_cloud','sun_intensity','total_suninensity','uv_intensity'
]

tmp_df = df[weather_features + ['data']]

correlation = tmp_df.corr(method='pearson')
keeping_features = correlation[np.abs(correlation['data']) > 0.1].index.tolist()
new_df = df[time_features + keeping_features]
column_pop = new_df.pop('data')
new_df.insert(0, 'data', column_pop)
new_df.to_csv('../data/area_0_cor.csv', index=False)