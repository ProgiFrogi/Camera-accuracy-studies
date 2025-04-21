from bisect import bisect_left
import numpy as np

class line:
    def __init__(self, center=np.array([0, 0]), direction=np.array([1, 0]), calc_rot: int = 1, internal_id=-1):
        self.center = center
        self.direction = direction / np.linalg.norm(direction)
        # self.space = np.ndarray([0,1])# space is orthogonal to direction. if sign(proj(space,vec))*sign(proj(direction,vec)) is positive, than vec in target cone, if negative - othervise
        self.calc_rot = calc_rot  # 1 or -1, if 1, then if rotating clockwise tangent to rotation will be going to center, -1 otherwise
        self.points = list()
        self.other_lines_ids = list()
        self.internal_id = internal_id

    def add_point(self, point, line_id):
        self.points.append([np.dot(self.direction, point - self.center), line_id])

    def sort_points(self):
        self.points = sorted(self.points)
        for i in range(len(self.points)):
            self.other_lines_ids.append(self.points[i][1])
            self.points[i] = self.points[i][0]

    def get_other_line(self, id):
        return self.other_lines_ids[id]

    def get_point_id(self, point):
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
        pos = self.get_point_id(point) + 1
        if pos == len(self.points):
            return None
        return self.get_point(pos)

    def prev_point(self, point):
        pos = self.get_point_id(point) - 1
        if pos == -1:
            return None
        return self.get_point(pos)

    def next_point_rot(self, point_id, prev):
        ans = self.get_point(point_id - 1)
        current = self.get_point(point_id)
        if cross2d(current - prev, ans - current) > 0:
            return point_id - 1
        elif cross2d(current - prev, self.get_point(point_id + 1) - current) > 0:
            return point_id + 1
        raise Exception("no appropriate direction found")

    def should_add(self, from_point_id, to_point_id):

        val = np.sign(np.dot(self.get_point(to_point_id) - self.get_point(from_point_id), self.center - (
                    self.get_point(to_point_id) + self.get_point(from_point_id)) / 2)) * self.calc_rot
        # print(val)
        return val < 0
        # return cross2d(self.get_point(to_point_id)-self.get_point(from_point_id),self.direction)>0

    def should_add_point(self, point):  # as if this point was previous point in hull
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
        if id == -1:  # or id == len(self.points):
            return (self.points[0] - 10000) * self.direction + self.center
        elif id == len(self.points):
            return (self.points[-1] + 10000) * self.direction + self.center
        elif 0 <= id < len(self.points):
            return self.points[id] * self.direction + self.center
        else:
            raise IndexError("wrong index of point in line: " + str(id))

    def get_point_len(self):
        return len(self.points)


def intersect(a: line, b: line) -> np.ndarray:
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
    """
    this function for given set of lines and eps calculates best horizon point.
    horizon point is point on horizon to which all parallel lines are converging.
    this function finds biggest subset of segments that if their endpoints will move no more than eps, they could intersect in one point.
    this code working for O(n) where n is size of segments.
    :param segments: list of segments
    :param eps:
    :return: point
    """
    lines: list[line] = list()
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
        value = 0
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
