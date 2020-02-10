import pandas as pd
import tkinter as tk
import pvlib
from tkinter import filedialog
import datetime


def getdate(n, y):
    if n <= 31:
        month = 1
        day = n
    elif n <= 59:
        month = 2
        day = n - 31
    elif n <= 90:
        month = 3
        day = n - 59
    elif n <= 120:
        month = 4
        day = n - 90
    elif n <= 151:
        month = 5
        day = n - 120
    elif n <= 181:
        month = 6
        day = n - 151
    elif n <= 212:
        month = 7
        day = n - 181
    elif n <= 243:
        month = 8
        day = n - 212
    elif n <= 273:
        month = 9
        day = n - 243
    elif n <= 304:
        month = 10
        day = n - 273
    elif n <= 334:
        month = 11
        day = n - 304
    else:
        month = 12
        day = n - 334
    date = pd.Timestamp(year=y, month=month, day=day, minute=30)
    return date


# Opening Input File
root = tk.Tk()
root.withdraw()
xl_location = "C:\\Users\\ahson\Google Drive\Testing Ground\Steam Generation\Sample Master File.xlsm"
# xl_location = filedialog.askopenfilename(title="Select Excel File from which data is to be taken ... ")
xl_pdata = "Project Data"
xl_cdata = "Calculated Data"
xl_ndata = "NREL Raw"
print("Loading Data from the source file ...")
df = pd.read_excel(io=xl_location, sheet_name=xl_pdata, usecols=[1])
# Initializing Solar Field Layout Parameters
C_NSX = df.iloc[35, 0]
C_NSY = 0
C_EWX = 0
C_EWY = df.iloc[36, 0]
NSC = df.iloc[33, 0]
EWC = df.iloc[34, 0]
latitude = df.iloc[4, 0]
longitude = df.iloc[5, 0]
year = df.iloc[7, 0]
df = pd.read_excel(io=xl_location, sheet_name=xl_ndata, usecols=[7, 9, 10])
h_counter = 2
max_dni = 0
best_day = 0
if year%4 == 0:
    year = year -1

for i in range(0, 365):
    dnisum = 0
    for j in range(0, 24):
        dnisum = dnisum + df.iloc[h_counter, 0]
        h_counter = h_counter + 1
    if dnisum > max_dni:
        max_dni = dnisum
        best_day = i + 1

print("Best Day =", best_day)

df = df[(best_day - 1) * 24 + 2:best_day * 24 + 2]
df = df.rename(columns={"Time Zone": "DNI (W/m2)", "Local Time Zone": "Pressure (kPa)",
                        "Clearsky DHI Units": "Temperature (DegC)"})
df["Pressure (kPa)"] = df["Pressure (kPa)"] * 10

date2use = pd.to_datetime(datetime.datetime.strptime(str(int(year) - 2000)+str(best_day), '%y%j').date()) + pd.Timedelta(minutes=30)
print(date2use)
time = pd.date_range(date2use, freq='1h', periods=24).tz_localize('Asia/Kolkata')

print("Loading Data for best day ...")
