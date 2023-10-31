#! /usr/bin/env python
#! -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import heapq
import matplotlib.pyplot as plt
import time

# 向量
class Vector:
    def __init__(self, x: int, y: int) -> None:
        self.x_ = x
        self.y_ = y

    def value(self):
        return np.sqrt(self.x_ ** 2 + self.y_ ** 2)
    
    def norm(self):
        if self.x_ != 0:
            self.x_ = self.x_ // np.abs(self.x_)
        if self.y_ != 0:
            self.y_ = self.y_ // np.abs(self.y_)
    
    def __str__(self) -> str:
        return str(self.x_) + " " + str(self.y_)

# 点
class Point:
    # 构造函数
    def __init__(self, x: int, y: int) -> None:
        self.x_ = x
        self.y_ = y

    def __add__(self, other: Vector):
        return Point(self.x_ + other.x_, self.y_ + other.y_)

    def __sub__(self, other) -> Vector:
        return Vector(self.x_ - other.x_, self.y_ - other.y_)
    
    def __eq__(self, other) -> bool:
        return self.x_ == other.x_ and self.y_ == other.y_
    
    def __str__(self) -> str:
        return str(self.x_) + " " + str(self.y_)
    
    def hash(self):
        return (self.x_, self.y_)

# 节点
class Node:
    def __init__(self, point: Point, cost=0.0, heuristic=0.0, parent=None) -> None:
        self.point_ = point
        self.cost_ = cost
        self.heuristic_ = heuristic
        self.f_value_ = cost + heuristic
        self.parent_ = parent

    def __lt__(self, other):
        return self.f_value_ < other.f_value_
    
    def hash(self):
        return self.point_.hash()

# A星搜索
class Astar:
    def __init__(self) -> None:
        self.height_ = None
        self.width_ = None
        self.occupancy_grid_ = None
        self.start_point_ = None
        self.goal_point_ = None
        self.candidate_dires_ = [Vector(1, 0), Vector(0, 1), Vector(0, -1), Vector(-1, 0), Vector(1, 1), Vector(1, -1), Vector(-1, 1), Vector(-1, -1)]
    
    # 进行路径搜索
    def search(self, start_point: Point, goal_point: Point, occupancy_grid: np.array):
        # 初始化变量
        self.height_, self.width_ = occupancy_grid.shape
        self.occupancy_grid_ = occupancy_grid
        self.start_point_ = start_point
        self.goal_point_ = goal_point
        # 构建初始节点和目标节点
        start_node = Node(start_point, heuristic=(goal_point - start_point).value())
        goal_node = Node(goal_point)
        # 构建开集合和闭集合
        open_set = [start_node]
        heapq.heapify(open_set)
        close_set = set()
        # 开始进行路径搜索
        while open_set:
            # 得到当前节点
            cur_node = heapq.heappop(open_set)
            # 判断当前节点是否在close集合中
            if cur_node.hash() in close_set:
                continue
            # 当前节点加入close集合
            close_set.add(cur_node.hash())
            # 判断当前节点是否为终点
            if cur_node.point_ == self.goal_point_:
                goal_node = cur_node
                break
            # 根据方向进行搜索
            for search_dire in self.candidate_dires_:
                neighbor_point = cur_node.point_ + search_dire
                if self.isAvailable(neighbor_point):
                    if neighbor_point.hash() in close_set:
                        continue
                    # 构建新节点
                    new_node = Node(point=neighbor_point, cost=cur_node.cost_ + (neighbor_point - cur_node.point_).value(), heuristic=(self.goal_point_ - neighbor_point).value(), parent=cur_node)
                    # 加入open集合
                    heapq.heappush(open_set, new_node)
        # 判断是否搜索成功
        if goal_node.parent_ is None:
            print("path search failed")
            return None
        # 进行路径回溯
        print("path search success")
        path = list()
        cur_node = goal_node
        while cur_node is not None:
            path.append(cur_node.point_)
            cur_node = cur_node.parent_
        path = reversed(path)
        return path
    
    # 判断点是否可行
    def isAvailable(self, point) -> bool:
        if 0 <= point.x_ < self.height_ and 0 <= point.y_ < self.width_ and self.occupancy_grid_[point.x_][point.y_] == 0:
            return True
        else:
            return False

if __name__== "__main__":
    # 读取图片作为占据栅格图
    map_path = os.path.dirname(__file__) + "/../data/test.png"
    if not os.path.exists(map_path):
        print("occupancy grid does not exist in ", map_path)
        exit(0)
    occupancy_grid = 255 - cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

    # 随机设置起点和终点
    np.random.seed(0)
    height, width = occupancy_grid.shape
    while True:
        start_point = Point(np.random.randint(0, height), np.random.randint(0, width))
        if occupancy_grid[start_point.x_][start_point.y_] == 0:
            break
    while True:
        goal_point = Point(np.random.randint(0, height), np.random.randint(0, width))
        if occupancy_grid[goal_point.x_][goal_point.y_] == 0 and (goal_point - start_point).value() > height:
            break
    
    # 确认规划设置环境
    plt.figure()
    plt.imshow(occupancy_grid)
    plt.plot(start_point.y_, start_point.x_, marker="o", color="red")
    plt.plot(goal_point.y_, goal_point.x_, marker="o", color="green")
    plt.show()

    # 进行路径生成
    astar = Astar()
    time_start = time.time()
    searched_path = astar.search(start_point, goal_point, occupancy_grid)
    time_end = time.time()
    print("time consuming: ", time_end - time_start, " s")
    # 进行可视化
    if searched_path is not None:
        xs, ys = list(), list()
        for p in searched_path:
            xs.append(p.x_)
            ys.append(p.y_)
        plt.figure()
        plt.imshow(occupancy_grid)
        plt.plot(start_point.y_, start_point.x_, marker="o", color="red")
        plt.plot(goal_point.y_, goal_point.x_, marker="o", color="green")
        plt.plot(ys, xs)
        plt.show()