from bisect import bisect_left
import numpy as np

class line:
    def __init__(self, center=np.array([0, 0]), direction=np.array([1, 0]), calc_rot: int = 1, internal_id=-1):
        """Initialize a line with a center point, direction vector, rotation flag, and optional ID.

        Args:
            center (np.ndarray): 2D point representing the line's center. Defaults to [0, 0].
            direction (np.ndarray): 2D vector defining the line's direction. Normalized internally.
            calc_rot (int): Rotation flag (1 or -1). If 1, clockwise tangent points to center; if -1, opposite.
            internal_id (int): Optional identifier for the line. Defaults to -1.

        Attributes:
            center (np.ndarray): Line's center point.
            direction (np.ndarray): Normalized direction vector.
            calc_rot (int): Rotation flag.
            points (list): List of projected points along the line with associated line IDs.
            other_lines_ids (list): IDs of lines intersecting at each point.
            internal_id (int): Line identifier.
        """
        
        self.center = center
        self.direction = direction / np.linalg.norm(direction)
        # self.space = np.ndarray([0,1])# space is orthogonal to direction. if sign(proj(space,vec))*sign(proj(direction,vec)) is positive, than vec in target cone, if negative - othervise
        self.calc_rot = calc_rot  # 1 or -1, if 1, then if rotating clockwise tangent to rotation will be going to center, -1 otherwise
        self.points = list()
        self.other_lines_ids = list()
        self.internal_id = internal_id

    def add_point(self, point, line_id):
        """Add a point to the line with its projection and associated line ID.

        Args:
            point (np.ndarray): 2D point to project onto the line.
            line_id (int): ID of the intersecting line associated with this point.
        """
        self.points.append([np.dot(self.direction, point - self.center), line_id])

    def sort_points(self):
        """Sort points by their projection along the line and extract line IDs.

        Updates `points` to contain only projections and populates `other_lines_ids` with corresponding line IDs.
        """
        self.points = sorted(self.points)
        for i in range(len(self.points)):
            self.other_lines_ids.append(self.points[i][1])
            self.points[i] = self.points[i][0]

    def get_other_line(self, id):
        """Get the ID of the line associated with a point at the given index.

        Args:
            id (int): Index of the point.

        Returns:
            int: ID of the associated line.
        """
        return self.other_lines_ids[id]

    def get_point_id(self, point):
        """Find the index of the point closest to the projection of the input point.

        Args:
            point (np.ndarray): 2D point to project onto the line.

        Returns:
            int: Index of the closest projected point.
        """
        proj = np.dot(self.direction, point - self.center)
        near_id = bisect_left(self.points, proj)
        if near_id >= len(self.points):
            # print("wrong guess", proj-self.points[-1])
            near_id = len(self.points) - 1
        now = np.linalg.norm(proj - self.points[near_id])
        other1 = other2 = now * 2
        if near_id < len(self.points) - 1:
            other1 = np.linalg.norm(proj - self.points[near_id + 1])
        if near_id > 0:
            other2 = np.linalg.norm(proj - self.points[near_id - 1])
        if other1 < now:
            near_id += 1
        elif other2 < now:
            near_id -= 1
        # print(now,other1,other2)
        # print("distance ", np.linalg.norm(point-self.get_point(near_id)))
        # for i in range(-1,len(self.points)+1):
        #     print("to other",np.linalg.norm(point-self.get_point(i)))
        return near_id

    def next_point(self, point):
        """Get the next point along the line relative to the input point's projection.

        Args:
            point (np.ndarray): 2D point to project onto the line.

        Returns:
            np.ndarray or None: Coordinates of the next point, or None if no next point exists.
        """
        pos = self.get_point_id(point) + 1
        if pos == len(self.points):
            return None
        return self.get_point(pos)

    def prev_point(self, point):
        """Get the previous point along the line relative to the input point's projection.

        Args:
            point (np.ndarray): 2D point to project onto the line.

        Returns:
            np.ndarray or None: Coordinates of the previous point, or None if no previous point exists.
        """
        pos = self.get_point_id(point) - 1
        if pos == -1:
            return None
        return self.get_point(pos)

    def next_point_rot(self, point_id, prev):
        """Determine the next point index based on rotation direction from a previous point.

        Args:
            point_id (int): Index of the current point.
            prev (np.ndarray): 2D coordinates of the previous point.

        Returns:
            int: Index of the next point.

        Raises:
            Exception: If no appropriate rotation direction is found.
        """
        current = self.get_point(point_id)
        if cross2d(current - prev, self.get_point(point_id - 1) - current) > 0:
            return point_id - 1
        elif cross2d(current - prev, self.get_point(point_id + 1) - current) > 0:
            return point_id + 1
        elif np.dot(self.direction,current-prev)>0:
            return point_id + 1
        elif np.dot(self.direction,current-prev)<0:
            return point_id - 1
        raise Exception("no appropriate direction found")

    def should_add(self, from_point_id, to_point_id):
        """Check if a segment between two points should be included based on rotation and geometry.

        Args:
            from_point_id (int): Index of the starting point.
            to_point_id (int): Index of the ending point.

        Returns:
            bool: True if the segment should be added, False otherwise.
        """
        val = np.sign(np.dot(self.get_point(to_point_id) - self.get_point(from_point_id), self.center - (
                    self.get_point(to_point_id) + self.get_point(from_point_id)) / 2)) * self.calc_rot
        # print(val)
        return val < 0
        # return cross2d(self.get_point(to_point_id)-self.get_point(from_point_id),self.direction)>0

    def should_add_point(self, point):
        """Check if a point should be added as if it were the previous point in a convex hull.

        Args:
            point (np.ndarray): 2D point to evaluate.

        Returns:
            bool: True if the point should be added, False otherwise.
        """
        proj_point = np.dot(point - self.center, self.direction) * self.direction + self.center
        other = np.dot(point - self.center, self.direction) / 2 * self.direction + self.center
        if cross2d(proj_point - point, other - proj_point) < 0:
            other = proj_point - (other - proj_point)
        val = np.sign(np.dot(other - proj_point, self.center - (other + proj_point) / 2)) * self.calc_rot
        return val < 0

    # def is_inside(self,point):
    #     point-=self.center
    #     value = np.sign(np.dot(self.space,point)*np.dot(self.direction,point))
    #     if value>0:
    #         return True
    #     else:
    #         return False

    def get_point(self, id):
        """Get the 2D coordinates of a point at the given index, including boundary cases.

        Args:
            id (int): Index of the point. Use -1 for a far-left point, len(points) for a far-right point.

        Returns:
            np.ndarray: 2D coordinates of the point.

        Raises:
            IndexError: If the index is invalid (e.g., less than -1 or greater than len(points)).
        """
        if id == -1:  # or id == len(self.points):
            return (self.points[0] - 10000) * self.direction + self.center
        elif id == len(self.points):
            return (self.points[-1] + 10000) * self.direction + self.center
        elif 0 <= id < len(self.points):
            return self.points[id] * self.direction + self.center
        else:
            raise IndexError("wrong index of point in line: " + str(id))

    def get_point_len(self):
        """Get the number of points stored in the line.

        Returns:
            int: Number of points.
        """
        return len(self.points)


def intersect(a: line, b: line) -> np.ndarray:
    """Calculate the intersection point of two lines.

    Args:
        a (line): First line object.
        b (line): Second line object.

    Returns:
        np.ndarray: 2D coordinates of the intersection point.

    Note:
        Assumes lines are not parallel; parallel lines may produce numerically unstable results.
    """
    find_det = lambda a, b, c, d: a * d - b * c
    find_point_1 = lambda a, b, c, d, coord, acess: find_det(find_det(a[0], acess(a[1]), b[0], acess(b[1])),
                                                             find_det(a[coord], 1, b[coord], 1),
                                                             find_det(c[0], acess(c[1]), d[0], acess(d[1])),
                                                             find_det(c[coord], 1, d[coord], 1))
    find_point = lambda a, b, c, d, coord: find_point_1(a, b, c, d, coord, lambda x: x) / find_point_1(a, b, c, d, 1,
                                                                                                       lambda x: 1)
    res = np.array([find_point(a.center, a.center + a.direction, b.center, b.center + b.direction, 0),
                    find_point(a.center, a.center + a.direction, b.center, b.center + b.direction, 1)])
    # print(res)
    return res


def get_horizon_point(segments, eps):
    """Find the best horizon point where the largest subset of segments can intersect within a given tolerance.

    This function identifies a point (horizon point) where the maximum number of line segments, when extended
    within a tolerance `eps`, can converge. It runs in O(n) time, where n is the number of segments.

    Args:
        segments (list): List of segments, where each segment is a numpy array of two 2D points [[x1, y1], [x2, y2]].
        eps (float): Maximum allowable deviation for segment endpoints to consider intersection.

    Returns:
        tuple: (value, point, segments_set)
            - value (int): Number of segments in the largest intersecting subset.
            - point (np.ndarray): 2D coordinates of the horizon point.
            - segments_set (set): Set of segment IDs that intersect at the horizon point.

    Note:
        If `eps` exceeds half the length of a segment, it is adjusted to 99% of that length to ensure numerical stability.
    """
    lines: list[line] = list()
    segments = np.array(segments).astype(np.float64)
    iterator = -1
    for i in segments:
        iterator += 1
        center = (i[0] + i[1]) / 2
        from math import asin, sin, cos
        direction = i[0] - i[1]
        if eps > (np.linalg.norm(direction) / 2):
            eps = np.linalg.norm(direction) / 2 * 0.99
        angle = asin(eps / (np.linalg.norm(direction) / 2))
        direction_y = np.linalg.norm(direction) / 2 * cos(angle) * sin(angle)
        direction_x = np.linalg.norm(direction) / 2 * (1 - sin(angle) * sin(angle))
        direction /= np.linalg.norm(direction)
        direction_ort = np.array([direction[1], -direction[0]])
        for j in [-1, 1]:
            line_dir = direction * direction_x + direction_ort * direction_y * j
            lines.append(
                line(center, line_dir, np.sign(cross2d(line_dir, direction) * np.dot(line_dir, direction)), iterator))
    # segs_from_points = dict()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if np.abs(cross2d(lines[i].direction, lines[j].direction)) < 1e-6:
                continue
            point = intersect(lines[i], lines[j])
            lines[i].add_point(point, j)
            lines[j].add_point(point, i)
            # print(np.linalg.norm(lines[i].direction))
            # print()
            # print(np.dot(lines[i].direction,point-lines[i].center)*lines[i].direction+lines[i].center - point)
            # print(cross2d(lines[i].direction,point-lines[i].center))
            # print(lines[i].get_point(lines[i].get_point_len()-1)-point)
            # print(lines[j].get_point(lines[j].get_point_len()-1)-point)
            # segs_from_points[point] = [lines[i],lines[j]]
    for j in range(len(lines)):
        lines[j].sort_points()
        # print(lines[j].other_lines_ids)
    segments_to_visit = set()
    start_segments = list()
    for i in range(len(lines)):
        for j in range(-1, lines[i].get_point_len()):
            segments_to_visit.add((j, j + 1, i))
            segments_to_visit.add((j + 1, j, i))
            if j == -1:
                start_segments.append([j, j + 1, i])
            if j == lines[i].get_point_len() - 1:
                start_segments.append([j + 1, j, i])

    # print(lines,segments_to_visit,start_segments)

    def next_point(previous_point_id, current_point_id, current_line_id):
        """Find the next point in the convex hull traversal.

        Args:
            previous_point_id (int): Index of the previous point.
            current_point_id (int): Index of the current point.
            current_line_id (int): ID of the current line.

        Returns:
            tuple or None: (prev_id, next_id, next_line_id) for the next point, or None if no next point exists.
        """
        if current_point_id >= lines[current_line_id].get_point_len() or current_point_id < 0:
            return None
        next_line_id = lines[current_line_id].get_other_line(current_point_id)
        other_id = lines[next_line_id].get_point_id(lines[current_line_id].get_point(current_point_id))
        # assert(np.linalg.norm(lines[next_line_id].get_point(other_id)-lines[current_line_id].get_point(current_point_id))<1e5)
        next_id = lines[next_line_id].next_point_rot(other_id, lines[current_line_id].get_point(previous_point_id))
        # assert(cross2d(lines[next_line_id].get_point(other_id)-lines[current_line_id].get_point(previous_point_id),
        #                 lines[next_line_id].get_point(next_id)-lines[next_line_id].get_point(other_id))
        #         >0)
        return (other_id, next_id, next_line_id)

    def get_segment(point1, point2, line_id, write_lines=False):
        """Compute properties of a convex hull segment, including area and intersecting segments.

        Args:
            point1 (int): Index of the starting point.
            point2 (int): Index of the ending point.
            line_id (int): ID of the line.
            write_lines (bool): If True, return the set of segment IDs; otherwise, omit.

        Returns:
            tuple: (value, area, avg_point, [line_ids])
                - value (int): Number of segments intersecting in the hull.
                - area (float): Area of the convex hull (or inf if open).
                - avg_point (np.ndarray): Average point of the hull vertices.
                - line_ids (set, optional): IDs of segments in the hull (if write_lines=True).
        """
        avg_point = lines[line_id].get_point(point1)
        avg_cnt = 1
        line_ids = set()
        area = 0
        # area = cross2d(lines[line_id].get_point(point1),lines[line_id].get_point(point2)-lines[line_id].get_point(point1))
        # print("start")
        while True:
            # print(point1,point2,line_id)
            avg_point += lines[line_id].get_point(point2)
            area += cross2d(lines[line_id].get_point(point1),
                            lines[line_id].get_point(point2) - lines[line_id].get_point(point1))
            avg_cnt += 1
            if (point1, point2, line_id) not in segments_to_visit:
                # print("closed")
                break
            segments_to_visit.remove((point1, point2, line_id))
            nxt = next_point(point1, point2, line_id)
            if lines[line_id].should_add(point1, point2):
                # print("added")
                line_ids.add(lines[line_id].internal_id)
            if nxt is None:
                # print("open")
                area = float('inf')
                break
            point1, point2, line_id = nxt
        # print(len(line_ids))
        value = 0 # number of segments that intersected in this convex hull
        avg_point /= avg_cnt
        tmp_set = set()
        line_ids = set()
        for i in range(len(lines)):
            if lines[i].should_add_point(avg_point):
                if lines[i].internal_id in tmp_set:
                    value += 1
                    line_ids.add(lines[i].internal_id)
                else:
                    tmp_set.add(lines[i].internal_id)
        if not write_lines:
            return (value, area, avg_point)
        else:
            return (value, area, avg_point, line_ids)

    def get_best_point():
        """Find the best horizon point by evaluating all possible convex hulls.

        Returns:
            tuple: (best_val, best_point, best_segs)
                - best_val (int): Maximum number of intersecting segments.
                - best_point (np.ndarray): 2D coordinates of the best horizon point.
                - best_segs (set): IDs of segments intersecting at the best point.
        """
        best_val = 0
        best_id = [0, 0, 0]
        best_point = np.array([0, 0])
        best_segs = set()
        best_area = 0
        for i in start_segments:
            val, area, point, segs = get_segment(i[0], i[1], i[2], True)
            if val > best_val or (val == best_val and area > best_area):
                best_val = val
                best_point = point
                best_area = area
                best_segs = segs
        while len(segments_to_visit) > 0:
            i = None
            for j in segments_to_visit:
                i = j
                break
            val, area, point, segs = get_segment(i[0], i[1], i[2], True)
            if val > best_val or (val == best_val and area > best_area):
                best_val = val
                best_point = point
                best_area = area
                best_segs = segs

        return (best_val, best_point, best_segs)

    return get_best_point()

def cross2d(x, y):
    return np.array([x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]])



