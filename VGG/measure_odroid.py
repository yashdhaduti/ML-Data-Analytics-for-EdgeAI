import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import csv
from timeit import default_timer as timer
import time

def getTelnetPower(SP2_tel, last_power):
    """
    read power values using telnet.
    """
    # Get the latest data available from the telnet connection without blocking
    tel_dat = str(SP2_tel.read_very_eager()) 
    print('telnet reading:', tel_dat)
    # find latest power measurement in the data
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2:findex].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power

def getTemps():
    """
    obtain the temp values from sysfs_paths.py
    """
    templ = []
    # get temp from temp zones 0-3 (the big cores)
    for i in range(4):
        temp = float(open(sysfs.fn_thermal_sensor.format(i),'r').readline().strip())/1000
        templ.append(temp)
    # Note: on the 5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped. Therefore, swap them back.
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ

# create a text file to log the results
out_fname = 'odroid.csv'
header = ["time", "power", "temp4", "temp5", "temp6", "temp7"]
out_file = open(out_fname, 'w')
writer = csv.writer(out_file, delimiter=',', lineterminator='\n')
writer.writerow(header)

# measurement   
SP2_tel = tel.Telnet("192.168.4.1")
total_power = 0.0 
for i in range(7000):  
    last_time = time.time()#time_stamp
    if(i == 0):
        start = last_time
    total_power = getTelnetPower(SP2_tel, total_power)
    print('telnet power:', total_power)
    # temp for big cores
    temps = getTemps()
    print('temp of big cores:', temps)
    time_stamp = last_time
    # Data writeout:
    writer.writerow([time_stamp - start, total_power, temps[0], temps[1], temps[2], temps[3]])
    elapsed = time.time() - last_time
    DELAY = 0.2
    time.sleep(max(0, DELAY - elapsed))