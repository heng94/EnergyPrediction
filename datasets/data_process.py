import pandas as pd


holiday2num = {
    'NoHoliday': 0,
    'TombSweepingDay': 1,
    'InternationalLabourDays': 2,
    'MidAutumnFestival': 3,
    'DragonBoatFestival': 4,
    'SpringFestival': 5, 
    'NationalDay': 6, 
    'NewYearsDay': 7, 
}

seaon2num = {
    'Spring': 0,
    'Summer': 1,
    'Autumn': 2,
    'Winter': 3,
}

workday2num = {
    'WorkingDay': 0,
    'NonWorkingDay': 1,
}   

df = pd.read_csv('../data/area_0.csv')

for idx in range(df.shape[0]):
    df.loc[idx, 'holiday'] = holiday2num[df.loc[idx, 'holiday']]
    df.loc[idx, 'season'] = seaon2num[df.loc[idx, 'season']]
    df.loc[idx, 'workday'] = workday2num[df.loc[idx, 'workday']]
# df.to_csv('../data/area_0_v1.csv', index=False)
print('1')