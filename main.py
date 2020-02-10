import math
import numpy as np
import csv
import cv2
import time

side_multiplier = math.sqrt(485 / 380)


def showimage(img):
    cv2.imshow('Window', img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def power_calc(start, end, step):
    k = start
    while k < end:
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
            if alpha[hr] * dni[hr] > 0:  # preforming area calculations only if sun is above horizon
                sf[hr] = sf[hr] + dni[hr] * conv_c2s(math.radians(alpha[hr]), math.radians(gamma[hr]))
        if sum(sf) > max_energy:
            max_energy = sum(sf)
            max_EW = C_EWY
            max_NS = C_NSX
        print(counter, "of 42", "\tNSX = ", round(C_NSX) / 100, "\tEWY =", round(C_EWY) / 100, "\tThermal Power =",
              round(sum(sf) * 100) / 100, "\tArea =", round(area * 100) / 100)
        k = k + step


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


def conv_c2s(zenith, azimuth):
    ewx = ((C_EWX + C_EWY * math.tan(azimuth)) / (1 + math.tan(azimuth) * math.tan(azimuth))) / math.cos(
        azimuth) * math.sin(zenith)
    nsx = ((C_NSX + C_NSY * math.tan(azimuth)) / (1 + math.tan(azimuth) * math.tan(azimuth))) / math.cos(
        azimuth) * math.sin(zenith)
    ewy = C_EWY * math.cos(azimuth) - C_EWX * math.sin(azimuth)
    nsy = C_NSY * math.cos(azimuth) - C_NSX * math.sin(azimuth)

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


# Main program starts
start_time = time.process_time()  # For Time of calculation
alpha = np.zeros(24, dtype='float32')  # Zenith Angle of Sun
gamma = np.zeros(24, dtype='float32')  # Azimuth Angle of Sun
dni = np.zeros(24, dtype='float32')  # Azimuth Angle of Sun
sf = np.zeros(24, dtype='float32')  # solar fraction
counter = 0
rf = 100  # Resolution Factor
max_dia = rf * 2 * math.sqrt(10.5 ** 2 + 5.5 ** 2) * side_multiplier

# Opening Input File
with open("C:\\Users\\ahson\Google Drive\Testing Ground\PY Files\Field-Optimizer\CSV_1_Day_Data.csv",
          'r') as csv_file_in:
    csv_reader = csv.reader(csv_file_in)
    for line in csv_reader:
        dni[counter] = (float(line[0]))  # DNI Initialized
        alpha[counter] = (float(line[1]))  # Alpha Angle Initialized
        gamma[counter] = (float(line[2]))  # Gamma Angle Initialized
        if counter == 23:
            break
        counter = counter + 1
csv_file_in.close()

print('File read successfully...')

# Initializing Solar Field Layout Parameters
C_NSY = 0  # NS Spacing in Y Direction
C_EWX = 0  # EW Spacing in X Direction
NSC = 10  # Number of dishes in NS Direction
EWC = 5  # Number of dishes in EW Direction
max_NS = 26  # Optimized NS Spacing
max_EW = 26  # Optimized EW Spacing
max_energy = 0  # Optimized Energy Generation
max_area_acre = 20  # Area Constraint in Acres
max_area = max_area_acre * 4046.85642  # Area Constraint in sq. mtrs
counter = 1
SingleDishArea = 485  # overlaparea(rf, 0, 0, 0, 0, 1, 1) / (rf ** 2)  #area of single Dish, Should be 485
MaxSFDishArea = SingleDishArea * NSC * EWC  # area of solar field with 0 shadows

print('Parameters Initialized, Calculating Areas...')

# First 15 Step Process
print('Performing First Step of Optimisation...')
for i in range(0, 15):
    C_EWY = 26 + 4 * i
    C_NSX = math.floor(10 * ((max_area / ((EWC - 1) * C_EWY + 25)) - 25) / (NSC - 1)) / 10
    if C_NSX < 26:
        break
    area = ((NSC - 1) * C_NSX + 25) * ((EWC - 1) * C_EWY + 25) / 4046.85642
    C_EWY = rf * C_EWY
    C_NSX = rf * C_NSX
    for hr in range(0, 24):
        sf[hr] = 0
    for hr in range(0, 24):
        if alpha[hr] * dni[hr] > 0:  # preforming area calculations only if sun is above horizon
            sf[hr] = sf[hr] + dni[hr] * conv_c2s(math.radians(alpha[hr]), math.radians(gamma[hr]))
    if sum(sf) > max_energy:
        max_energy = sum(sf)
        max_EW = C_EWY
        max_NS = C_NSX
    print(counter, "of 42", "\tNSX = ", round(C_NSX) / 100, "\tEWY =", round(C_EWY) / 100, "\tThermal Power =",
          round(sum(sf) * 100) / 100, "\tArea =", round(area * 100) / 100)
    counter = counter + 1

# Second 8 Step Process
print('Performing Second Step of Optimisation...')
for i in range(math.floor(max_EW / rf) - 4, math.ceil(max_EW / rf) + 5):
    C_EWY = i
    C_NSX = math.floor(10 * ((max_area / ((EWC - 1) * C_EWY + 25)) - 25) / (NSC - 1)) / 10
    if C_NSX < 26:
        break
    area = ((NSC - 1) * C_NSX + 25) * ((EWC - 1) * C_EWY + 25) / 4046.85642
    C_EWY = rf * C_EWY
    C_NSX = rf * C_NSX
    for hr in range(0, 24):
        sf[hr] = 0
    for hr in range(0, 24):
        if alpha[hr] * dni[hr] > 0:  # preforming area calculations only if sun is above horizon
            sf[hr] = sf[hr] + dni[hr] * conv_c2s(math.radians(alpha[hr]), math.radians(gamma[hr]))
    if sum(sf) > max_energy:
        max_energy = sum(sf)
        max_EW = C_EWY
        max_NS = C_NSX
    print(counter, "of 42", "\tNSX = ", round(C_NSX) / 100, "\tEWY =", round(C_EWY) / 100, "\tThermal Power =",
          round(sum(sf) * 100) / 100, "\tArea =", round(area * 100) / 100)
    counter = counter + 1

# Third 20 Step Process
print('Performing Final Step of Optimisation...')
for i in range(math.floor(max_EW / rf * 10) - 10, math.ceil(max_EW / rf * 10) + 11):
    C_EWY = i / 10
    C_NSX = math.floor(10 * ((max_area / ((EWC - 1) * C_EWY + 25)) - 25) / (NSC - 1)) / 10
    if C_NSX < 26:
        break
    area = ((NSC - 1) * C_NSX + 25) * ((EWC - 1) * C_EWY + 25) / 4046.85642
    C_EWY = rf * C_EWY
    C_NSX = rf * C_NSX
    for hr in range(0, 24):
        sf[hr] = 0
    for hr in range(0, 24):
        if alpha[hr] * dni[hr] > 0:  # preforming area calculations only if sun is above horizon
            sf[hr] = sf[hr] + dni[hr] * conv_c2s(math.radians(alpha[hr]), math.radians(gamma[hr]))
    if sum(sf) > max_energy:
        max_energy = sum(sf)
        max_EW = C_EWY
        max_NS = C_NSX
    print(counter, "of 42", "\tNSX = ", round(C_NSX) / 100, "\tEWY =", round(C_EWY) / 100, "\tThermal Power =",
          round(sum(sf) * 100) / 100, "\tArea =", round(area * 100) / 100)
    counter = counter + 1

with open('CSVData_Output.csv', 'w') as csv_file_out:
    writer = csv.writer(csv_file_out)
    writer.writerows(map(lambda x: [x], sf))
csv_file_out.close()

print("Calculation and Writing completed in", time.process_time() - start_time)
print(max_NS, max_EW, max_energy)
