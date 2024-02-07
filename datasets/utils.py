import numpy as np


time_feature_weight = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
])

time_features = [
    'day','hour','day_of_year','week','month','days_in_month','day_of_week','season','holiday','workday',
]

weather_features = [
    'sea_pressure','ground_pressure','air_temp','air_max','air_min','air_mean','rain','snow','ground_temp',
    'ground_max','ground_min','ground_mean','body_temp','body_max','body_min','body_mean','damp',
    'evaporation','prevaporation','north_wind','south_wind','low_cloud','middle_cloud','high_cloud',
    'total_cloud','sun_intensity','total_suninensity','uv_intensity'
]
    
    
    
    
    
    