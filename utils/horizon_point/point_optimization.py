import numpy as np

def horizon_point_optimization_loss(point,segments):
    ans = 0
    # print(len(segments))
    # print(type(point),point)
    point = np.array(point).astype(np.float64)
    segments = np.array(segments).astype(np.float64)
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
    point = np.array(point).astype(np.float64)
    segments = np.array(segments).astype(np.float64)
    for i in segments:
        center = (i[0]+i[1])/2
        direction = i[1]-i[0]
        length = np.linalg.norm(direction)
        direction/=length
        direction2 = point-center
        direction2/=np.linalg.norm(direction2)
        ans += np.tan(np.abs(np.arccos(np.abs(np.dot(direction,direction2)))))*length
    return ans

def horizon_point_optimization_loss_v3(point,segments):
    ans = 0
    point = np.array(point).astype(np.float64)
    segments = np.array(segments).astype(np.float64)
    for i in segments:
        center = (i[0]+i[1])/2
        direction = i[1]-i[0]
        direction2 = point-center
        direction2/=np.linalg.norm(direction2)
        from .get_horizon_point import cross2d
        ans += np.abs(cross2d(direction2,direction))/2
    return ans

def horizon_point_optimization_loss_v3_1(point,segments):
    point = np.array(point, dtype=np.float64)
    segments = np.array(segments, dtype=np.float64)

    centers = np.mean(segments, axis=1)
    
    directions = segments[:, 1] - segments[:, 0]
    
    direction2 = point - centers
    norm = np.linalg.norm(direction2, axis=1, keepdims=True)
    direction2 = np.divide(direction2, norm, where=norm!=0)  # avoid division by zero
    
    cross_products = np.abs(direction2[:, 0] * directions[:, 1] - direction2[:, 1] * directions[:, 0])
    
    ans = np.sum(cross_products) / 2
    return ans


def optimize_point(point,segments,tol=1e-2):
    from scipy.optimize import minimize
    return minimize(horizon_point_optimization_loss_v3_1,point,segments, method = 'Nelder-Mead',tol=tol).x

def segment_clusterization_loss(eps,segments,tol=1e-2,debug=False):
    segments = np.array(segments).astype(np.float64)
    from .get_horizon_point import cluster_segments
    clusters = cluster_segments(segments,eps)
    losses = list()
    for i in range(len(clusters)):
        new_point = optimize_point(clusters[i][1],[segments[j] for j in clusters[i][0]],tol=tol)#todo: convert from ids to segments
        losses.append((horizon_point_optimization_loss_v3_1(new_point,[segments[j] for j in clusters[i][0]])/((len(clusters[i][0])**2))**2,i))
    losses = sorted(losses,key=lambda x: x[0])
    sum_loss = 0
    cnt = 0
    sum_all_loss = 0
    if debug:
        lenghts = list()
        for i in clusters:
            lenghts.append(len(i[0]))
        print(f"sizes of clusters:{lenghts}")
    for i in losses:
        if len(clusters[i[1]][0])>2 and cnt<2:
            sum_loss+=i[0]#todo:if inf
            cnt+=1
        else:
            sum_all_loss+=1
        sum_all_loss+=i[0]
    if cnt == 0:
        return 1e9
    if cnt == 1:
        return 1e7*sum_all_loss +eps #very big number
    return sum_loss*10+eps+sum_all_loss

def loss_wrapper(eps, params):
    if eps<0:
        return 1e9
    if "segments" in params:
        result = 0
        if params['tol'] is not None:
            result = segment_clusterization_loss(eps,params["segments"],tol=params['tol'],debug=params['debug'] if params['debug'] else False)
        else:
            result = segment_clusterization_loss(eps,params["segments"],tol=1e-2,debug=params['debug'] if params['debug'] else False)
        if params['debug']:
            print(f"loss eps:{eps},loss value:{result}")
        return result
            
    raise RuntimeError("no segments during loss calculation")

def optimize_eps(segments,tol=1e-2,debug=False):
    from scipy.optimize import minimize_scalar
    return minimize_scalar(loss_wrapper,bracket=(1e-7,10),args={"segments":segments,"tol":tol,"debug":debug},method = 'brent',tol=tol).x
    