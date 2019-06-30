from matplotlib import pyplot as plt


fileDirectory = "/home/user/Documents/simEnv_2018_07_31"

with open(fileDirectory + "/file_simResults_Passive.txt", 'r') as file:
    content = file.readlines()

content = [x.strip() for x in content]
time_stamp_vec = []
excFreq_vec = []
roadExc_vec = [[], [], [], [], [], [], [], [], [], [], [], []]
# 0:7 disp 8:15 vel
l_sim_fullCar_Y = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

for line in content[1:]:
    splitted_line = line.split(';')
    time_stamp_vec.append(float(splitted_line[0]))
    excFreq_vec.append(float(splitted_line[1]))
    for num in range(0, 12):
        roadExc_vec[num].append(float(splitted_line[2 + num]))
    for num in range(0, 24):
        l_sim_fullCar_Y[num].append(float(splitted_line[14 + num]))

with open(fileDirectory + "/file_simResults_SkyHook.txt", 'r') as file:
    content = file.readlines()

content = [x.strip() for x in content]
time_stamp_vec = []
excFreq_vec = []
roadExc_vec = [[], [], [], [], [], [], [], [], [], [], [], []]
# 0:7 disp 8:15 vel
l_sim_fullCar_Y_SH = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

for line in content[1:]:
    splitted_line = line.split(';')
    time_stamp_vec.append(float(splitted_line[0]))
    excFreq_vec.append(float(splitted_line[1]))
    for num in range(0, 12):
        roadExc_vec[num].append(float(splitted_line[2 + num]))
    for num in range(0, 24):
        l_sim_fullCar_Y_SH[num].append(float(splitted_line[14 + num]))

plot1, = plt.plot(l_sim_fullCar_Y[14], 'b-')
plt.xlabel('Probes')
# Make the y-axis label, ticks and tick labels match the line color.
plt.tick_params('y', colors='b')
plot2, = plt.plot(l_sim_fullCar_Y_SH[14], 'r', label="SH")
plt.ylabel('velocity', color='r')
plt.tick_params('y', colors='r')
plt.legend((plot1, plot2), ("Passive Active Suspension", "SH Semiactive Suspension"))
plt.title("Czy alg wgl dziala??")
plt.show()
