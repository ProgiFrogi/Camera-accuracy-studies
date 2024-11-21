import numpy as np
from random import randint

#for 950*720 ?????
eps_dbscan_for_clusters = 0.035
eps_dbscan_for_small_lines = 0.05/10

min_cluster_len = 80 # 100
#для Хаффа
rho= 2 #8
theta=np.pi / (140) #140
threshold= 80#160
minLineLength=  70 #80
maxLineGap= 10 #10

color = [(randint(10, 255), randint(10, 255), randint(10, 255))]*100
colors =  [(randint(10, 255), randint(10, 255), randint(10, 255))]*100

''' for old picture 
eps_dbscan_for_small_lines = 0.1
eps_dbscan_for_clusters = 0.05
min_cluster_len = 120
#для Хаффа
rho= 8 #10
theta=np.pi / (140) #80
threshold= 60#140
minLineLength=  80 #150
maxLineGap= 15 #50
'''