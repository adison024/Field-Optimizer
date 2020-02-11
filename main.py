import math
import numpy as np
import csv
import cv2
import time
import pandas as pd
import tkinter as tk
import pvlib
from tkinter import filedialog
import datetime

side_multiplier = math.sqrt(485 / 380)


def showimage(img):
    cv2.imshow('Window', img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def cp(y_center, x_center):
    # Parameters passed are (x,y) but read are (y,x). This is to compensate the way images are processed
    res = rf * side_multiplier
    coordinates = np.zeros((50, 2), np.int32)
    for i in range(0, 5):
        # Initializing +ve X Coordinates (loop)
        coordinates[i * 2, 0] = coordinates[i * 2 + 1, 0] = coordinates[20 - i * 2, 0] = coordinates[
            21 - i * 2, 0] = x_center + (5.5 + i) * res - 1

        # Initializing -ve X Coordinates (loop)
        coordinates[22 + i * 2, 0] = coordinates[23 + i * 2, 0] = coordinates[43 - i * 2, 0] = coordinates[
            42 - i * 2, 0] = x_center - (5.5 + i) * res

        # Initializing -ve Y Coordinates (loop)
        coordinates[i * 2 + 1, 1] = coordinates[(i + 1) * 2, 1] = coordinates[42 - i * 2, 1] = coordinates[
            41 - i * 2, 1] = y_center - (9.5 - i) * res

        # Initializing +ve Y Coordinates (loop)
        coordinates[11 + i * 2, 1] = coordinates[12 + i * 2, 1] = coordinates[31 - i * 2, 1] = coordinates[
            32 - i * 2, 1] = y_center + (5.5 + i) * res - 1

    # Initializing +ve X Coordinates (non-loop)
    coordinates[10, 0] = coordinates[11, 0] = x_center + 10.5 * res - 1
    coordinates[45, 0] = coordinates[48, 0] = coordinates[44, 0] = coordinates[49, 0] = x_center + 0.5 * res

    # Initializing -ve X Coordinates (non-loop)
    coordinates[46, 0] = coordinates[47, 0] = x_center - 0.5 * res - 1
    coordinates[32, 0] = coordinates[33, 0] = x_center - 10.5 * res

    # Initializing +ve Y Coordinates (non-loop)
    coordinates[21, 1] = coordinates[22, 1] = y_center + 10.5 * res - 1
    coordinates[45, 1] = coordinates[46, 1] = y_center + 0.5 * res

    # Initializing -ve Y Coordinates (non-loop)
    coordinates[47, 1] = coordinates[48, 1] = y_center - 0.5 * res - 1
    coordinates[43, 1] = coordinates[44, 1] = coordinates[49, 1] = coordinates[0, 1] = y_center - 10.5 * res

    return coordinates


def overlaparea(nsx, nsy, ewx, ewy, nsc, ewc):
    xtrm_ox = ewx * (ewc - 1)
    xtrm_nsx = nsx * (nsc - 1)
    xtrm_ewx = ewx * (ewc - 1) + nsx * (nsc - 1)

    xtrm_oy = ewy * (ewc - 1)
    xtrm_nsy = nsy * (nsc - 1)
    xtrm_ewy = ewy * (ewc - 1) + nsy * (nsc - 1)

    x_max = max(xtrm_ox, xtrm_nsx, xtrm_ewx, 0)
    y_max = max(xtrm_oy, xtrm_nsy, xtrm_ewy, 0)
    x_min = min(xtrm_ox, xtrm_nsx, xtrm_ewx, 0)
    y_min = min(xtrm_oy, xtrm_nsy, xtrm_ewy, 0)

    xi = 10.5 * side_multiplier * rf - x_min
    yi = 10.5 * side_multiplier * rf - y_min
    array_x_size = math.ceil(x_max - x_min + 21 * side_multiplier * rf)
    array_y_size = math.ceil(y_max - y_min + 21 * side_multiplier * rf)

    sunviewimg = np.zeros((array_x_size, array_y_size), dtype=np.uint8)

    for row in range(0, nsc):
        for col in range(0, ewc):
            x = xi + ewx * col
            y = yi + ewy * col
            # print(x,y)
            dishpoints = cp(x, y)
            cv2.fillPoly(sunviewimg, [dishpoints], 255)
            # showimage(sunviewimg)
        x = xi + nsx
        y = yi + nsy
        xi = x
        yi = y

    # previewpercet = 1280/array_x_size*100
    # previewimg = cv2.resize(sunviewimg, (int(array_y_size*1280/array_x_size),1280), interpolation = cv2.INTER_AREA)
    # showimage(previewimg)
    # asd = cv2.countNonZero(sunviewimg)
    return cv2.countNonZero(sunviewimg)


def conv_c2s(zenith, azimuth, nsx, ewy):
    ewx = 0
    nsy = 0
    ewx = ((ewx + ewy * math.tan(azimuth)) / (1 + math.tan(azimuth) * math.tan(azimuth))) / math.cos(
        azimuth) * math.sin(zenith)
    nsx = ((nsx + nsy * math.tan(azimuth)) / (1 + math.tan(azimuth) * math.tan(azimuth))) / math.cos(
        azimuth) * math.sin(zenith)
    ewy = ewy * math.cos(azimuth) - ewx * math.sin(azimuth)
    nsy = nsy * math.cos(azimuth) - nsx * math.sin(azimuth)

    # Finding intermediate distance between dishes to select the calculation method.
    osn = math.sqrt(nsx ** 2 + nsy ** 2)
    oew = math.sqrt(ewx ** 2 + ewy ** 2)
    snew = math.sqrt(((ewx - nsx) * (ewx - nsx)) + ((ewy - nsy) * (ewy - nsy)))
    od4 = math.sqrt(2 * (osn ** 2 + oew ** 2) - snew ** 2)
    int_angle = math.atan(math.sqrt((ewx / ewy) ** 2)) + math.atan(math.sqrt((nsx / nsy) ** 2))
    nsline_space = osn * math.sin(int_angle)
    ewline_space = oew * math.sin(int_angle)
    # sfcase = 0

    if (od4 >= max_dia) and (snew >= max_dia) and (osn >= max_dia) and (oew >= max_dia):
        # All dishes are separately visible
        fraction = 1
        # sfcase = 1

    elif (oew < max_dia) and (nsline_space >= max_dia):  # EW Lines are overlapping
        fraction = overlaparea(0, 0, ewx, ewy, 1, EWC) / (rf ** 2) * NSC / MaxSFDishArea
        # sfcase = 2

    elif (osn < max_dia) and (ewline_space >= max_dia):  # NS Lines are overlapping
        fraction = overlaparea(nsx, nsy, 0, 0, NSC, 1) / (rf ** 2) * EWC / MaxSFDishArea
        # sfcase = 3
    else:
        fraction = overlaparea(nsx, nsy, ewx, ewy, NSC, EWC) / (rf ** 2) / MaxSFDishArea
        # sfcase = 4

    if fraction > 1:
        return 1
    else:
        return round(fraction, 4)


def power_calc(start, end, step):
    k = start
    sf = np.zeros(24, dtype='float32')  # solar fraction
    # max_NS = 26  # Optimized NS Spacing
    max_EW = 26  # Optimized EW Spacing
    max_energy = 0  # Optimized Energy Generation
    while k <= end:
        C_EWY = k
        C_NSX = math.floor(10 * ((max_area / ((EWC - 1) * C_EWY + 25)) - 25) / (NSC - 1)) / 10
        if C_NSX < 26:
            break
        area = ((NSC - 1) * C_NSX + 25) * ((EWC - 1) * C_EWY + 25) / 4046.85642
        C_EWY = rf * C_EWY
        C_NSX = rf * C_NSX
        for hr in range(0, 24):
            sf[hr] = 0
        for hr in range(0, 24):
            if spa.iloc[hr, 0] * srcdf.iloc[hr, 0] > 0:  # preforming area calculations only if sun is above horizon
                sf[hr] = sf[hr] + srcdf.iloc[hr, 0] * conv_c2s(math.radians(spa.iloc[hr, 0]),
                                                               math.radians(spa.iloc[hr, 1]), C_NSX, C_EWY)
        if sum(sf) > max_energy:
            max_energy = sum(sf)
            max_EW = C_EWY
            # max_NS = C_NSX
        print("NSX = ", round(C_NSX) / 100, "\tEWY =", round(C_EWY) / 100, "\tThermal Power =",
              round(sum(sf) * 100) / 100, "\tArea =", round(area * 100) / 100)
        # print(k)
        k = k + step
    return max_EW / rf


''' *** Main program starts *** '''
start_time = pd.Timestamp.now()  # For Time of calculation

# Initial assignment of variables
rf = 100  # Resolution Factor for scaling up the image for better accuracy
max_dia = rf * 2 * math.sqrt(10.5 ** 2 + 5.5 ** 2) * side_multiplier
h_counter = 2  # Hour counter
max_dni = 0  # Maximum cumulative DNI of a whole day, used as a check parameter while finding best day
best_day = 0  # Best Day based on which the optimisation will be performed
max_area_acre = 20  # Area Constraint in Acres
max_area = max_area_acre * 4046.85642  # Area Constraint in sq. mtrs
SingleDishArea = 485  # overlaparea(rf, 0, 0, 0, 0, 1, 1) / (rf ** 2)  #area of single Dish, Should be 485

# Opening Input File
# root = tk.Tk()
# root.withdraw()
# xl_location = filedialog.askopenfilename(title="Select Excel File from which data is to be taken ... ")

#xl_location = "C:\\Users\\ahson\\Google Drive\\Testing Ground\\Steam Generation\\Sample Master File.xlsm"
xl_location = "C:\\Users\\ahson\\Google Drive\\Testing Ground\\PY Files\\Field Optimizer\\Field-Optimizer\\Sample Master File.xlsm"
xl_pdata = "Project Data"
xl_ndata = "NREL Raw"
print("Loading Data from the source file ...")
srcdf = pd.read_excel(io=xl_location, sheet_name=xl_pdata, usecols=[1])

print('File read successfully...')

# Initializing Solar Field Layout Parameters
NSC = int(srcdf.iloc[33, 0])  # Number of NS Dishes
EWC = int(srcdf.iloc[34, 0])  # Number of EW Dishes
latitude = srcdf.iloc[4, 0]  # Latitude of site
longitude = srcdf.iloc[5, 0]  # Longitude of site
year = srcdf.iloc[7, 0]  # Year in selection
if year % 4 == 0:  # This version compensates for leap years simply by going one year back
    year = year - 1
MaxSFDishArea = SingleDishArea * NSC * EWC  # area of solar field with 0 shadows

# Loading NREL Raw Data from the excel File
srcdf = pd.read_excel(io=xl_location, sheet_name=xl_ndata, usecols=[7, 9, 10])

# Finding the best day for optimisation. This is the day with maximum solar Insolation (DNI)
for i in range(0, 365):
    dnisum = 0
    for j in range(0, 24):
        dnisum = dnisum + srcdf.iloc[h_counter, 0]
        h_counter = h_counter + 1
    if dnisum > max_dni:
        max_dni = dnisum
        best_day = i + 1

print("Best Day =", best_day, "Performing Optimisation taking this day...")

# Trimming the source data frame so that only the best day data remains, with colums of DNI, pressure and temp. only
srcdf = srcdf[(best_day - 1) * 24 + 2:best_day * 24 + 2]
srcdf = srcdf.rename(columns={"Time Zone": "DNI (W/m2)", "Local Time Zone": "Pressure (kPa)",
                              "Clearsky DHI Units": "Temperature (DegC)"})
srcdf["Pressure (kPa)"] = srcdf["Pressure (kPa)"] * 10  # Convert pressure from mbar to kPa

# Creating time dataframe for the best day
date2use = pd.to_datetime(
    datetime.datetime.strptime(str(int(year) - 2000) + str(best_day), '%y%j').date()) + pd.Timedelta(minutes=30)

time = pd.date_range(date2use, freq='1h', periods=24).tz_localize('Asia/Kolkata')

# Creating Alpha and Gamma angles
spa = pvlib.solarposition.spa_python(time, latitude, longitude, pressure=srcdf["Pressure (kPa)"],
                                     temperature=srcdf["Temperature (DegC)"],
                                     delta_t=67.0, atmos_refract=None, how='numpy')

spa.drop(['apparent_zenith', 'zenith', 'elevation', 'equation_of_time'], axis=1, inplace=True)


spa['azimuth'] = 180 - spa['azimuth']
for i in range(0, 24):
    if spa.iloc[i, 0] < 0:
        spa.iloc[i, 0] = spa.iloc[i, 1] = 0

print("Pre-Optimisation tasks completed in ", (pd.Timestamp.now() - start_time)/np.timedelta64(1,'s'),"seconds")

# Starting Optimisation
start_time = pd.Timestamp.now()  # For Time of calculation
print('Parameters Initialized, Calculating Areas...\n\nCalculating First Step...')
opt_EW = power_calc(26, 82, 4)

print("First step optimised at EW =", opt_EW, "\n\nCalculating Second Step...")
opt_EW = power_calc(opt_EW - 3, opt_EW + 3, 1)

print("First step optimised at EW =", opt_EW, "\n\nCalculating Third Step...")
opt_EW = power_calc(opt_EW - 0.9, opt_EW + 1, 0.1)
opt_NS = math.floor(10 * ((max_area / ((EWC - 1) * opt_EW + 25)) - 25) / (NSC - 1)) / 10
area = ((NSC - 1) * opt_NS + 25) * ((EWC - 1) * opt_EW + 25) / 4046.85642

print("\nOptimum NS = ", opt_NS, "\t\tOptimum EW = ", opt_EW)

print("Optimisation took", (pd.Timestamp.now() - start_time)/np.timedelta64(1,'s'),"seconds")