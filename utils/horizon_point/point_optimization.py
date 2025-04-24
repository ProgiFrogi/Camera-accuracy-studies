import numpy as np

def horizon_point_optimization_loss(point,segments):
    ans = 0
    # print(len(segments))
    # print(type(point),point)
    point = np.array(point)
    for i in segments:
        center = (i[0]+i[1])/2
        direction = i[1]-i[0]
        length = np.linalg.norm(direction)
        direction/=length
        direction2 = point-center
        direction2/=np.linalg.norm(direction2)
        ans += np.abs(np.arccos(np.abs(np.dot(direction,direction2))))*length

def horizon_point_optimization_loss_v2(point,segments):
    ans = 0
    # print(len(segments))
    # print(type(point),point)
    point = np.array(point)
    for i in segments:
        center = (i[0]+i[1])/2
        direction = i[1]-i[0]
        length = np.linalg.norm(direction)
        direction/=length
        direction2 = point-center
        direction2/=np.linalg.norm(direction2)
        ans += np.tan(np.abs(np.arccos(np.abs(np.dot(direction,direction2)))))*length
    return ans