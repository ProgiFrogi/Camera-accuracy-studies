import math
def points_on_ellipse(x_0, y_0, a, b, angle, n: int = 10):
    points = list()
    for i in range(n):
        point = (math.cos(i / math.pi / 2), math.sin(i / math.pi / 2))
        point = (point[0] * a, point[1] * b)
        point = (point[0] * math.cos(angle) + point[1] * math.sin(angle),
                 -point[0] * math.sin(angle) + point[1] * math.cos(angle))
        point = (point[0] + x_0, point[1] + y_0)
        points.append(point)
    return points


def points_on_line(from_x, from_y, to_x, to_y):
    ans = list()
    if math.fabs(from_x - to_x) > math.fabs(from_y - to_y):
        step_x = (to_x - from_x) / (to_y - from_y)
        for i in range(int(from_y), int(to_y) + 1, int((to_y - from_y) / math.fabs(to_y - from_y))):
            ans.append((int(step_x * i + from_x), int(i + from_y)))
    else:
        step_y = (to_y - from_y) / (to_x - from_x)
        for i in range(int(from_x), int(to_x) + 1, int((to_x - from_x) / math.fabs(to_x - from_x))):
            ans.append((int(i + from_x), int(step_y * i + from_y)))
    return ans


def points_on_line_v2(from_x, from_y, to_x, to_y, n_steps=None):
    if n_steps is None:
        return points_on_line(from_x, from_y, to_x, to_y)
    ans = list()
    step_x = (to_x - from_x) / n_steps
    step_y = (to_y - from_y) / n_steps
    for i in range(n_steps):
        ans.append((round(from_x + step_x * i), round(from_y + step_y * i)))
    return ans


def metric_by_points_(x_0, y_0, a, b, angle, data, n: int = 20): 
    points = points_on_ellipse(x_0, y_0, a, b, angle, n)
    ans = None
    cntr = 0
    for i in range(n):
        for j in points_on_line_v2(points[i][0], points[i][1], points[(i + 1) % n][0], points[(i + 1) % n][1],
                                   1 + round(max(abs(points[i][0] - points[(i + 1) % n][0]),
                                                 abs(points[i][0] - points[(i + 1) % n][0])))):
            if 0 <= j[0] < data.shape[0] and 0 <= j[1] < data.shape[1]:
                if ans is None:
                    ans = data[j[0]][j[1]]
                else:
                    ans += data[j[0]][j[1]]
                if data[j[0]][j[1]] == 0:
                    cntr +=1
                    # pass
    cntr*=0.3
    if ans is None:
        return cntr 
    return ans + cntr

def metric_by_mask(x_0, y_0, a, b, angle, data, width: int = 20,debug=False):
    """
    this function constructs mask with form of ellipse,applies it to data and computes all white pixels
    this function cannot be used as loss
    assumed that max(data) gives max value for this data type so everything scaled down by factor of this value
    """
    import cv2
    import numpy as np
    # print(data.shape)
    data = cv2.resize(data,(500,500))#so thickness will not matter for same image
    max_point = data.max()
    #data/=max_point # this line replaced with /max_point in return statement for optimization
    mask = cv2.ellipse(np.zeros(data.shape).astype(np.uint8),[int(x_0),int(y_0)],[int(a),int(b)],angle,0,360,255,width)
    metric = np.sum(data[mask].astype(int))
    ellipse_perimeter = math.sqrt(a*a+b*b)# O(ellipse_perimeter_approximation)
    if a==0 or b == 0:
        return -float("inf")
    metric/=ellipse_perimeter
    if debug:
        print(metric/max_point,a,b)
    return metric/max_point
    

def metric_by_points(x_0, y_0, a, b, angle, data, n: int = 20):
    max_point = max(data)
    data/=max_point
    return metric_by_points_(x_0, y_0, a, b, angle, data, n)
    
def loss_by_points(x_0, y_0, a, b, angle, data, n: int = 20):
    raise NotImplementedError()
    import torch
    max_point = max(data)#todo:add here nograd or something 
    data/=max_point
    return metric_by_points_(x_0, y_0, a, b, angle, data, n)#todo check case where only cntr returned, try to wrap it into torch.tensor with 0 grad
