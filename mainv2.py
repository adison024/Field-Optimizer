import pandas as pd
import tkinter as tk

import pvlib
from tkinter import filedialog
import datetime

# Opening Input File
root = tk.Tk()
root.withdraw()
xl_location = "C:\\Users\\ahson\Google Drive\Testing Ground\Steam Generation\Sample Master File.xlsm"
# xl_location = filedialog.askopenfilename(title="Select Excel File from which data is to be taken ... ")
xl_pdata = "Project Data"
xl_cdata = "Calculated Data"
xl_ndata = "NREL Raw"
print("Loading Data from the source file ...")
srcdf = pd.read_excel(io=xl_location, sheet_name=xl_pdata, usecols=[1])
# Initializing Solar Field Layout Parameters
C_NSX = srcdf.iloc[35, 0]
C_NSY = 0
C_EWX = 0
C_EWY = srcdf.iloc[36, 0]
NSC = srcdf.iloc[33, 0]
EWC = srcdf.iloc[34, 0]
latitude = srcdf.iloc[4, 0]
longitude = srcdf.iloc[5, 0]
year = srcdf.iloc[7, 0]
srcdf = pd.read_excel(io=xl_location, sheet_name=xl_ndata, usecols=[7, 9, 10])
h_counter = 2
max_dni = 0
best_day = 0
if year%4 == 0:
    year = year -1

for i in range(0, 365):
    dnisum = 0
    for j in range(0, 24):
        dnisum = dnisum + srcdf.iloc[h_counter, 0]
        h_counter = h_counter + 1
    if dnisum > max_dni:
        max_dni = dnisum
        best_day = i + 1

print("Best Day =", best_day)

srcdf = srcdf[(best_day - 1) * 24 + 2:best_day * 24 + 2]
srcdf = srcdf.rename(columns={"Time Zone": "DNI (W/m2)", "Local Time Zone": "Pressure (kPa)",
                        "Clearsky DHI Units": "Temperature (DegC)"})
srcdf["Pressure (kPa)"] = srcdf["Pressure (kPa)"] * 10

date2use = pd.to_datetime(datetime.datetime.strptime(str(int(year) - 2000)+str(best_day), '%y%j').date()) + pd.Timedelta(minutes=30)
print(date2use)
time = pd.date_range(date2use, freq='1h', periods=24).tz_localize('Asia/Kolkata')

spa = pvlib.solarposition.spa_python(time, latitude, longitude, pressure=srcdf["Pressure (kPa)"], temperature=srcdf["Temperature (DegC)"],
                                     delta_t=67.0, atmos_refract=None, how='numpy')

spa.drop(['apparent_zenith', 'zenith', 'elevation', 'equation_of_time'], axis=1, inplace=True)
print(time)

spa['azimuth'] = 180 - spa['azimuth']
for i in range(0, 24):
    if spa.iloc[i, 0] < 0:
        spa.iloc[i, 0] = spa.iloc[i, 1] = 0


print("Loading Data for best day ...")
