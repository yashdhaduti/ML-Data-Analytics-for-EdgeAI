import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import csv
import gpiozero
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

# create a text file to log the results
out_fname = 'pi.csv'
header = ["time", "power", "temperature"]
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
    cpu_temp = gpiozero.CPUTemperature().temperature
    print('cpu temp' + str(cpu_temp))
    time_stamp = last_time
    # Data writeout:
    writer.writerow([time_stamp - start, total_power, cpu_temp])
    elapsed = time.time() - last_time
    DELAY = 0.2
    time.sleep(max(0, DELAY - elapsed))