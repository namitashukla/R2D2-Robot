# CIS 521: R2D2 - Homework 2
from typing import List, Tuple, Set, Optional
import queue
import math
import numpy as np
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from itertools import permutations
from collections import defaultdict, deque
import heapq

student_name = "Type your full name here."

Vertex = Tuple[int, int]
Edge = Tuple[Vertex, Vertex]

DROID_SPEED = 100
DROID_SPEED_SQUARE_PER_SEC = 1


# Part 1: Compare Different Searching Algorithms
class Graph:
    """A directed Graph representation"""

    def __init__(self, vertices: Set[Vertex], edges: Set[Edge]):
        self.vertices = vertices
        self.edges = edges
        self.adjacency_list = defaultdict(set)
        for u, v in self.edges:
            self.adjacency_list[u].add(v)
            self.adjacency_list[v].add(u)

    def neighbors(self, u: Vertex) -> Set[Vertex]:
        """Return the neighbors of the given vertex u as a set"""
        return self.adjacency_list[u]

    def _backtrace(self, parent, goal):
        path = [goal]
        curr = parent[goal]
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path

    def bfs(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """
        Use BFS algorithm to find the path from start to goal in the given graph.

        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search.
        """

        queue = deque([start])
        parent = {start: None}

        while queue:
            node = queue.popleft()
            if node == goal:
                return self._backtrace(parent, goal), set(parent.keys())
            for adj_vertex in self.neighbors(node):
                if adj_vertex not in parent:
                    parent[adj_vertex] = node
                    queue.append(adj_vertex)

    def dfs(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """
        Use DFS algorithm to find the path from start to goal in the given graph.

        :return: a tuple (valid_path, node_visited),
                 where valid_path is a list of vertices that represents the path from start to goal (no need to be shortest), and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search.
        """

        visited = set()

        def dfs_visit(node):
            if node in visited:
                return None
            visited.add(node)
            if node == goal:
                return [goal]
            for neighbor in self.neighbors(node):
                res = dfs_visit(neighbor)
                if res:
                    res.append(node)
                    return res

        valid_path = dfs_visit(start)
        valid_path.reverse()

        return valid_path, visited

    def a_star(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """
        Use A* algorithm to find the path from start to goal in the given graph.

        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search.
        """

        parent = {start: None}
        best_so_far = {start: 0}
        h = lambda n: abs(n[0] - goal[0]) + abs(n[1] - goal[1])
        heap = [(h(start), 0, start)]

        while len(heap) > 0:
            _, cum_distance, node = heapq.heappop(heap)

            if best_so_far[node] < cum_distance:
                continue

            if node == goal:
                return self._backtrace(parent, goal), set(parent.keys())

            neighbor_cum_distance = cum_distance + 1
            for neighbor_node in self.neighbors(node):
                if neighbor_cum_distance < best_so_far.get(neighbor_node, float("inf")):
                    heapq.heappush(
                        heap,
                        (
                            neighbor_cum_distance + h(neighbor_node),
                            neighbor_cum_distance,
                            neighbor_node,
                        ),
                    )
                    best_so_far[neighbor_node] = neighbor_cum_distance
                    parent[neighbor_node] = node

    def tsp(
        self, start: Vertex, goals: Set[Vertex]
    ) -> Tuple[Optional[List[Vertex]], Optional[List[Vertex]]]:
        """
        Use A* algorithm to find the path that begins at start and passes through all the goals in the given graph,
        in an order such that the path is the shortest.

        :return: a tuple (optimal_order, shortest_path),
                 where shortest_path is a list of vertices that represents the path from start that goes through all the
                 goals such that the path is the shortest; optimal_order is an ordering of goals that you visited in
                 order that results in the above shortest_path. Return (None, None) if no such path exists.
        """
        shortest_length = float("inf")
        optimal_order = None
        shortest_path = None
        for ordered_goals in permutations(goals):
            paths = [start]
            prev = start
            for goal in ordered_goals:
                path, _ = self.a_star(prev, goal)
                paths += path[1:]
                prev = goal
            if len(paths) < shortest_length:
                shortest_length = len(paths)
                shortest_path = paths
                optimal_order = ordered_goals
        return optimal_order, shortest_path


# Part 2: Let your R2-D2 rolling in Augment Reality (AR) Environment
def get_transformation(k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate the transformation matrix using the given equation P = K x (R | T)"""
    return np.matmul(k, np.concatenate((r, t), axis=1))


def convert_3d_to_2d(
    p: np.ndarray, points_3d: List[Tuple[float, float, float]]
) -> List[Tuple[int, int]]:
    """Convert a list of 3D real world points to 2D image points in pixels given the transformation matrix,
    preserving the order of the points."""
    print("p: ", p)
    print("points_3d: ", points_3d)
    points_2d = []
    print(np.vstack([np.array(l) for l in points_3d]))
    for x, y, z in np.matmul(p, np.vstack([np.array(l) for l in points_3d])).tolist():
        x /= z
        y /= z
        points_2d.append(x, y)
    return points_2d


def in_box(point, bounding_points):
    point_x, point_y = point
    min_x = min(x for x, y in bounding_points)
    max_x = max(x for x, y in bounding_points)
    min_y = min(y for x, y in bounding_points)
    max_y = max(y for x, y in bounding_points)
    return min_x <= point_x <= max_x and min_y <= point_y <= max_y


def convert_2d_to_relative(
    point_2d: Tuple[int, int], maze_in_2d: List[List[Tuple[int, int]]]
) -> Optional[Vertex]:
    """Convert a 2D image point to maze coordinates using the given maze coordinates in 2D image.
    Return None if the 2D point isn't in the maze. Assume the coordinates are axis-aligned."""
    for i in range(len(maze_in_2d) - 1):
        for j in range(len(maze_in_2d[i]) - 1):
            if in_box(
                point_2d,
                (
                    maze_in_2d[i][j],
                    maze_in_2d[i + 1][j],
                    maze_in_2d[i + 1][j + 1],
                    maze_in_2d[i][j + 1],
                ),
            ):
                return (i, j)


def path_to_moves(path: List[Vertex]) -> List[Tuple[int, int]]:
    """Taking a list of vertices and returns a list of droid actions (heading, steps)"""

    heading_mapping = {(0, 1): 90, (0, -1): 270, (1, 0): 180, (-1, 0): 0}

    cum_distance = 0
    curr_heading = None
    actions = []

    prev = path[0]
    for curr in path[1:]:
        diff = (curr[0] - prev[0], curr[1] - prev[1])
        heading = heading_mapping[diff]
        if curr_heading is None:
            curr_heading = heading
        if heading != curr_heading:
            actions.append((curr_heading, cum_distance))
            cum_distance = 0
        cum_distance += 1
        prev = curr
        curr_heading = heading
    if cum_distance:
        actions.append((curr_heading, cum_distance))

    return actions


def droid_roll(path: List[Vertex]):
    """Make your droid roll with the given path. You should decide speed and time of rolling each move."""
    moves = path_to_moves(path)
    with SpheroEduAPI(scanner.find_toy()) as droid:
        for heading, distance in moves:
            droid.roll(DROID_SPEED, heading, distance / DROID_SPEED_SQUARE_PER_SEC)
