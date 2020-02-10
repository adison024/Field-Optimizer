import pvlib
import pandas as pd

year = 2020
if 2019 % 4 == 0:
    time = pd.date_range(start=str(year) + '-01-01 00:30', freq='1h', periods=8784).tz_localize('Asia/Kolkata')
    time = time.drop(time.index[1416:(1416+24)])
    c = 1
else:
    time = pd.date_range(start=str(year) + '-01-01 00:30', freq='1h', periods=8760).tz_localize('Asia/Kolkata')
    c = 2

df = pvlib.solarposition.spa_python(time, 23.171954, 70.180572, altitude=0, pressure=101325, temperature=12,
                                    delta_t=67.0, atmos_refract=None, how='numpy')
df.drop(['apparent_zenith', 'zenith', 'elevation', 'equation_of_time'], axis=1, inplace=True)
print(time)

'''
df['azimuth'] = 180 - df['azimuth']
for i in range(0, 8760):
    if df.iloc[i, 0] < 0:
        df.iloc[i, 0] = df.iloc[i, 1] = 0
'''

print(df)
