import math
import numpy as np
from numba import jit

#@jit
def equidistant_ellipse_points(x_0, y_0, a, b,angle, n = 10):
    """
    Returns approximately n equidistant points on an ellipse.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        n (int): The desired number of equidistant points.

    Returns:
        numpy.ndarray: An array of shape (n, 2) containing the (x, y) coordinates
                       of the approximately equidistant points.
    """
    if a <= 0 or b <= 0 or n <= 0:
        raise ValueError("Semi-axes (a, b) and number of points (n) must be positive.")

    points = []
    perimeter_approx = ellipse_perimeter_approximation(a,b)
    arc_length_increment = perimeter_approx / n

    theta = np.linspace(0, 2 * np.pi, 200*n)  # More points for better arc length approximation
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    cumulative_arc_lengths = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    cumulative_arc_lengths[0] = 0

    target_arc_length = 0
    current_point_index = 0
    for _ in range(n):
        while current_point_index < len(cumulative_arc_lengths) - 1 and \
              cumulative_arc_lengths[current_point_index + 1] <= target_arc_length:
            current_point_index += 1
        point = None
        if current_point_index < len(x):
            point = (a * np.cos(theta[current_point_index]), b * np.sin(theta[current_point_index]))
        else:
            # Handle the case where we reach the end due to approximation
            point = (a * np.cos(0), b * np.sin(0)) # Or the last point

        target_arc_length += arc_length_increment

        point = (point[0] * math.cos(angle) - point[1] * math.sin(angle),
        point[0] * math.sin(angle) + point[1] * math.cos(angle))
        point = (point[0] + x_0, point[1] + y_0)
        points.append(point)

    return points



#@jit
def points_on_ellipse(x_0, y_0, a, b, angle, n: int = 10):
    points = list()
    for i in np.linspace(0, 2 * np.pi, n, endpoint=False):
        point = (math.cos(i), math.sin(i))
        point = (point[0] * a, point[1] * b)
        point = (point[0] * math.cos(angle) - point[1] * math.sin(angle),
                 point[0] * math.sin(angle) + point[1] * math.cos(angle))
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
    ellipse_perimeter = ellipse_perimeter_approximation(a,b)
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

from collections import deque

#@jit
def find_path(array, start, end, timeout=2000):
    """
    Find a path from start to end in a numpy array, moving only through non-zero elements.
    
    Args:
        array: 2D numpy array where non-zero elements represent passable cells
        start: Tuple (row, col) representing the starting position
        end: Tuple (row, col) representing the target position
    
    Returns:
        List of tuples representing the path from start to end, or None if no path exists
    """
    rows, cols = array.shape
    if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
        end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols):
        return None
    
    if array[start] == 0 or array[end] == 0:
        return None
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = deque()
    queue.append(start)
    
    # Dictionary to keep track of visited cells and their parents
    visited = {start: None}
    
    while queue and timeout>0:
        timeout-=1
        current = queue.popleft()
        
        # If we've reached the end, reconstruct the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            return path[::-1]  # Reverse to get start to end
        
        # Explore neighbors
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            
            # Check if neighbor is within bounds and passable
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                array[neighbor] != 0 and neighbor not in visited):
                
                visited[neighbor] = current
                queue.append(neighbor)
    
    # No path found
    return None

#@jit
def find_path_length(array, start, end,timeout=2000):
    """
    Find the length of the shortest path from start to end in a numpy array, 
    moving only through non-zero elements.
    
    Args:
        array: 2D numpy array where non-zero elements represent passable cells
        start: Tuple (row, col) representing the starting position
        end: Tuple (row, col) representing the target position
    
    Returns:
        Integer representing the path length (number of steps), or -1 if no path exists
    """
    # Check if start or end is out of bounds or zero
    rows, cols = array.shape
    if (start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols or
        end[0] < 0 or end[0] >= rows or end[1] < 0 or end[1] >= cols):
        # print("wrong start/end point: out of bounds")
        return None
    
    if array[start] == 0 or array[end] == 0:
        # print("wrong start/end point")
        return None
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    queue = deque()
    queue.append((start, 0))
    
    visited = set()
    visited.add(start)
    
    while queue and timeout>0:
        timeout-=1
        current, distance = queue.popleft()
        
        if current == end:
            return distance
        
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                array[neighbor] != 0 and neighbor not in visited):
                
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    # No path found
    # if timeout<=0:
    #     print("timeout")
    return None
#@jit
def ramanujan_(a, b):
    h = ((a - b)/(a + b))**2
    return math.pi * (a + b) * (1 + (3*h)/(10 + math.sqrt(4 - 3*h)))

#@jit
def cantrell_(a, b):
    h = ((a - b)/(a + b))**2
    coefs = [1,1/4,1/64,1/256,25/16384]
    sum_pre_h= 0
    mul = 1
    for i in coefs:
        sum_pre_h+=mul*i
        mul*=h
    return math.pi * (a + b) * sum_pre_h
#@jit
def simple_(a, b):
    return 2 * math.pi * math.sqrt((a**2 + b**2)/2)
#@jit
def muir_(a, b):
    return 2 * math.pi * ((a**1.5 + b**1.5)/2)**(1/1.5)
#@jit
def ellipse_perimeter_approximation(a, b):
    """
    Calculate approximation of ellipse perimeter.

    Average Errors:
    Ramanujan: Avg Error = 0.001413%, Max Error = 0.039977%
    Cantrell: Avg Error = 0.008340%, Max Error = 0.170823%
    Simple  : Avg Error = 3.842144%, Max Error = 11.072068%
    Muir    : Avg Error = 0.158268%, Max Error = 1.045971%
    
    Args:
        a: Semi-major axis length
        b: Semi-minor axis length
    
    Returns:
        Approximate perimeter of the ellipse
    """

    return cantrell_(a,b)
import numpy as np
from collections import deque

def find_nearest_one_manhattan_optimized(grid):
    """
    Finds the nearest '1' for each point in a 2D array using an optimized
    approach based on Breadth-First Search (BFS) with Manhattan distance.

    Args:
        grid (np.ndarray): A 2D bool NumPy array .

    Returns:
        np.ndarray: A 2D NumPy array of the same shape as the input grid,
                      where each element represents the Manhattan distance to the
                      nearest '1'.
    """
    rows, cols = grid.shape
    distance_grid = np.full_like(grid, -1, dtype=int)  # Initialize with -1
    nearest_grid = np.ones((grid.shape[0],grid.shape[1],2),dtype=int)*(-1)
    queue = deque()

    # Initialize the queue with all the '1's and set their distance to 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c]:
                distance_grid[r, c] = 0
                queue.append((r, c, 0))  # (row, col, distance)
                nearest_grid[r,c] = np.array([r,c])

    # Perform BFS
    while queue:
        r, c, dist = queue.popleft()

        # Explore adjacent cells (up, down, left, right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check if the new cell is within the grid boundaries and hasn't been visited
            if 0 <= nr < rows and 0 <= nc < cols and distance_grid[nr, nc] == -1:
                distance_grid[nr, nc] = dist + 1
                queue.append((nr, nc, dist + 1))
                nearest_grid[nr,nc] = nearest_grid[r,c]

    return distance_grid,nearest_grid
#@jit
def search_point_(point,data,max_rad=5):#max search pixel distance
    point = point.astype(int)
    points = [(0,0)]
    for distance in range(1, max_rad + 1):
        # Generate all possible points at the current Manhattan distance
        for dx in range(-distance, distance + 1):
            dy = distance - abs(dx)
            if dy >0:
                for sign in [-1, 1]:
                    x = dx
                    y = sign * dy
                    points.append((x, y))
            else:
                points.append((dx,0))
    points = np.array(points)+point

    for x,y in points:
            if 0 < y < data.shape[1] and 0 < x < data.shape[0]:
                if data[x][y] > 0:
                    return np.array([x,y])
    return None

def sigmoid_(z):
    return 1/(1 + np.exp(-z))

def calculate_manhattan_distances(nearest_points_array):
    """
    Calculate the Manhattan distance for each cell to its nearest point using vectorized operations.

    Parameters:
    nearest_points_array (np.array): 2D array where each element is the coordinates of the nearest point.

    Returns:
    np.array: 2D array of Manhattan distances.
    """
    # Create a grid of indices
    i_indices, j_indices = np.indices(nearest_points_array.shape[:2])

    # Extract the nearest point coordinates
    nearest_i = nearest_points_array[..., 0]
    nearest_j = nearest_points_array[..., 1]

    # Calculate the absolute differences
    abs_diff_i = np.abs(i_indices - nearest_i)
    abs_diff_j = np.abs(j_indices - nearest_j)

    # Sum the absolute differences to get the Manhattan distance
    distances = abs_diff_i + abs_diff_j

    return distances

#@jit
def metric_by_near_segments_base(x_0, y_0, a, b, angle, data,help_nearest=None,debug=False,max_rad=50):
    """
    for each pixel of image that intersects our ellipse we search for nearest white pixel(top k?). then we search white distance between this pixel and previous.
    if there is no white length,then distance is x.
    if there is no white point in radius y, then we add z to metric
    after calculation, we divide sum by ellipse perimeter and apply sigmoid to result: we want to allow some error, i.e. if there is no half of ellipse it is ok if other half is an ellipse, but if it found ellipses on lines we want to strike metric hard for this error
    """
    n_point_sample = max(1000,int(a+b)*4)

    if data is None or len(data.shape)!=2:
        raise RuntimeError(f"wrong image shape: should be (.,.) but it is {data.shape}")
    if a==0 or b ==0:
        return 0

    el_points = equidistant_ellipse_points(x_0,y_0,a,b,angle,n_point_sample)
    penalty = 0
    prev_point = None
    prev_p_ell = None
    no_nearest = 0
    no_distance = 0
    non_distinct_nearest_cnt = 0
    non_distinct_nearest_cnt_tmp = 0
    cluster_cntr = 0
    sum_distances = 0
    seen_points = []
    seen_p_values = []
    nearest_points = []
    if debug and False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(np.array(el_points)[:, 0], np.array(el_points)[:, 1], color='red')
        plt.xlim(0, 700)
        plt.ylim(0,700)
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
    for point_t in el_points:
        point = np.array(point_t,dtype=np.int32)
        # point = np.flip(point)
        if prev_p_ell is not None and (point==prev_p_ell).all():
            n_point_sample-=1
            continue
        if debug:
            seen_points.append(point)
            if prev_p_ell is not None:
                if np.sum(np.abs(point-prev_p_ell))>2:
                    print("-"*100,"probably bad")
        current_point = None
        if help_nearest is None:
            current_point = search_point_(point,data,max_rad=max_rad)
        else:
            current_point = np.flip(help_nearest[tuple(np.flip(point))])
            # current_point = np.flip(current_point)
            # current_point = np.array([help_nearest[point[0],point[1]][1],help_nearest[point[0],point[1]][0]])
            seen_p_values.append(np.sum(np.abs(current_point-point)))
            nearest_points.append(current_point)
            if np.sum(np.abs(current_point-point))>max_rad:
                if debug and False:
                    print("bad: lenght too big: ",np.sum(np.abs(current_point-point)))
                current_point = None
            else:
                sum_distances+=np.sum(np.abs(current_point-point))
        if current_point is not None and prev_point is not None and (current_point == prev_point).all():
            non_distinct_nearest_cnt_tmp+=1
        elif non_distinct_nearest_cnt_tmp>0:
            non_distinct_nearest_cnt+=non_distinct_nearest_cnt_tmp
            cluster_cntr+=1
            non_distinct_nearest_cnt_tmp=0
        if current_point is None:
            no_nearest+=1
        if prev_point is not None and current_point is not None:
            distance = find_path_length(data,tuple(current_point),tuple(prev_point))
            # print(distance)
            if distance is None:
                # penalty+=2*np.sum(np.abs(prev_p_ell-point))
                no_distance+=1
            else:
                # print(point,current_point)
                # print(np.abs(np.sum(np.abs(prev_p_ell-point))-np.sum(np.abs(prev_point-current_point))))
                # print(np.abs(np.sum(np.abs(prev_p_ell-point))-float(distance)))
                penalty+=max(np.abs(np.sum(np.abs(prev_p_ell-point))-np.sum(np.abs(prev_point-current_point))),
                np.abs(np.sum(np.abs(prev_p_ell-point))-float(distance)))
        # print(penalty/((a+b)*4))
        if current_point is not None:
            penalty+=np.linalg.norm(point-current_point)
        prev_p_ell = point.copy()
        if current_point is not None:
            prev_point = current_point
    # print(penalty/((a+b)*4))
    # return 1-sigmoid(penalty/((a+b)*4)-1)
    if debug:
        print(f"x:{x_0},y:{y_0},a{a},b{b},no_nearest%: {no_nearest/n_point_sample}, no_distance%: {no_distance/n_point_sample}, cluster_cntr%: {cluster_cntr/n_point_sample},avg distance: {float("inf") if (n_point_sample-no_distance-no_nearest == 0)  else sum_distances/(n_point_sample-no_distance-no_nearest)},penalty/((a+b)*4): {penalty/((a+b)*4)} max_rad: {max_rad}")
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(calculate_manhattan_distances(help_nearest))
        plt.scatter(x=np.array(seen_points)[:, 0], y=np.array(seen_points)[:, 1],c=np.ma.masked_array(np.array(seen_p_values),mask=(np.array(seen_p_values)<45)))
        plt.scatter(x=np.array(nearest_points)[:, 0], y=np.array(nearest_points)[:, 1],c='red')
        plt.xlim(0, 700)
        plt.ylim(0,700)
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
    if (n_point_sample-no_distance-no_nearest == 0):
        return -100

    if no_nearest/n_point_sample>0.6 or cluster_cntr/n_point_sample<0.01:
        if debug:
            print("not enough points")
        return -float("inf")
    return 1/(1+((penalty/((a+b)*4))+(10*no_nearest/n_point_sample)**3+(5*no_distance/n_point_sample)**3)+(10*(1-cluster_cntr/n_point_sample))**3)



def metric_by_near_segments_base_by_direction(x_0, y_0, a, b, angle, data,debug=False):
    """
    for each pixel of image that intersects our ellipse we search for nearest white pixel(top k?). then we search white distance between this pixel and previous.
    if there is no white length,then distance is x.
    if there is no white point in radius y, then we add z to metric
    after calculation, we divide sum by ellipse perimeter and apply sigmoid to result: we want to allow some error, i.e. if there is no half of ellipse it is ok if other half is an ellipse, but if it found ellipses on lines we want to strike metric hard for this error
    """
    n_point_sample = min(1000,int(a+b)*4)

    if data is None or len(data.shape)!=2:
        raise RuntimeError(f"wrong image shape: should be (.,.) but it is {data.shape}")
    if a==0 or b ==0:
        return 0

    el_points = equidistant_ellipse_points(x_0,y_0,a,b,angle,n_point_sample)
    penalty = np.float64(0)
    prev_point = None
    prev_p_ell = None
    no_nearest = 0
    no_distance = 0
    distinct_nearest_cnt = 0
    sum_distances = 0
    for point in el_points:
        point = np.array(point).astype(int)
        if (point==prev_p_ell).all():
            n_point_sample-=1
            continue
        current_point = search_point_(point,data,max_rad=20)
        if prev_point is None:
            if current_point is None:
                no_nearest+=1
            pass
        elif current_point is None:
            # penalty+=4*np.sum(np.abs(prev_p_ell-point))
            no_nearest+=1
            # print("bad, not found")
        else:
            distance = find_path_length(data,tuple(current_point),tuple(prev_point))
            sum_distances+=distance
            # print(distance)
            if distance is None:
                # penalty+=2*np.sum(np.abs(prev_p_ell-point))
                no_distance+=1
            else:
                # print(point,current_point)
                # print(np.abs(np.sum(np.abs(prev_p_ell-point))-np.sum(np.abs(prev_point-current_point))))
                # print(np.abs(np.sum(np.abs(prev_p_ell-point))-float(distance)))
                first = prev_p_ell-point
                second = prev_point-current_point
                penalty+=np.sqrt(np.abs(first[0]*second[1]-first[1]*second[0]))
        # print(penalty/((a+b)*4))
        if current_point is not None:
            penalty+=np.linalg.norm(point-current_point)
        prev_p_ell = point
        if current_point is not None:
            prev_point = current_point
    # print(penalty/((a+b)*4))
    # return 1-sigmoid(penalty/((a+b)*4)-1)
    if debug:
        print(f"no_nearest/n_point_sample: {no_nearest/n_point_sample}, no_distance/n_point_sample: {no_distance/n_point_sample}, avg distance: {sum_distances/(n_point_sample-no_distance-no_nearest)},penalty/((a+b)*4): {penalty/((a+b)*4)} ")
    if no_nearest/n_point_sample>0.6:
        return -float("inf")
    return -(penalty/((a+b)*4))-675*(no_nearest/n_point_sample)**3-125*(no_distance/n_point_sample)**3



if __name__ == "__main__":
    #test 
    print('-'*100)
    print(metric_by_near_segments_base(30,30,10,20,0,np.ones((100,100)).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(20,20,10,20,0,np.zeros((100,100))))
    print('-'*100)
    print(metric_by_near_segments_base(20,20,10,20,0,(np.random.rand(100,100)>(np.ones((100,100))*(1-4e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(20,20,10,20,0,(np.random.rand(100,100)>(np.ones((100,100))*(1-4e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(50,50,20,40,0,(np.random.rand(100,100)>(np.ones((100,100))*(1-4e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(50,50,20,40,0,(np.random.rand(100,100)>(np.ones((100,100))*(1-4e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(500,500,200,400,0,(np.random.rand(1000,1000)>(np.ones((1000,1000))*(1-4e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(500,500,400,400,0,(np.random.rand(1000,1000)>(np.ones((1000,1000))*(1-4e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(50,50,40,40,0,(np.random.rand(100,100)>(np.ones((100,100))*(1-9e-1))).astype(int)))
    print('-'*100)
    print(metric_by_near_segments_base(50,50,10,10,0,(np.random.rand(100,100)>(np.ones((100,100))*(1e-1))).astype(int)))
    # print(find_path_length(np.array([[1,1],[1,1]]),(0,0),(1,1)))

    # print(np.sum(np.random.rand(1000,1000)>(np.ones((1000,1000))*(2e0-1e0))).astype(int))