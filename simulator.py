import random
import math
import pickle
import igraph
import pandas as pd
import numpy as np
import copy
from ortools.graph.python import min_cost_flow

# 城市参数
class CityParam:

    def __init__(self):
        self.file_path1 = "" # 日期相关数据文件的路径
        self.file_path2 = "" # 精度相关数据文件的路径
        self.taxi_number = 5000 # 城市出租车数量

        self.simulated_start_time = 0 * 60 *60 # 环境模拟开始时间,单位秒
        self.simulated_end_time = 24 * 60 *60 # 环境模拟终止时间,单位秒
        self.step_interval = 600 # 每个时间步的间隔,单位秒
        self.timeline_interval = 1 # 时间线的间隔时间,单位秒
        self.trip_response_interval = 60 # 出行未被响应,重新发起的时间间隔,单位秒
        self.trip_tolerance_time = 600 # 乘客能容忍的最大等待时长
        self.taxi_to_idle_weight = 0.5 # 在服务的出租车空闲转换权重


class Event:

    def __init__(self, type, taxi_id, trip_id):
        self.type = type
        self.taxi_id = taxi_id
        self.trip_id = trip_id
        

class Trip:
    
    def __init__(self, trip_id, generation_time, departure_node, arrival_node, cost_time, amount):
        self.id = trip_id
        self.taxi_id = None
        self.status = 0
        self.generation_time = generation_time
        self.departure_node = departure_node
        self.arrival_node = arrival_node
        self.response_time = None
        self.reach_time = None
        self.cost_time = cost_time
        self.amount = amount
    
    # 重置
    def reset(self):
        self.taxi_id = None
        self.status = 0
        self.response_time = None
        self.reach_time = None
    
    # 出行被出租车响应
    def responsed(self, taxi_id, response_time, reach_time):
        self.status = 1
        self.taxi_id = taxi_id
        self.response_time = response_time - self.generation_time
        self.reach_time = reach_time

    # 出行完成
    def finish(self):
        self.status = 2

    # 出行取消
    def cancel(self, cancel_time):
        self.status = 3
        self.response_time = cancel_time - self.generation_time
        self.reach_time = 0


class Taxi:

    def __init__(self, taxi_id, node):
        self.id = taxi_id
        self.status = 3
        self.node = node
        self.initial_node = node
        self.working_time = 0

        self.trip_id = None
        self.destination_node = -1
        self.in_grids_clock = None

    # 重置
    def reset(self):
        self.status = 3
        self.node = self.initial_node
        self.working_time = 0

        self.trip_id = None
        self.destination_node = -1
        self.in_grids_clock = None


    # 出租车响应出行
    def receive_trip(self, trip_id):
        self.status = 1 # 服务中
        self.trip_id = trip_id
    
    # 出租车完成出行
    def finish_trip(self, node):
        self.status = 0 # 空闲
        self.node = node
    
    # 目的地重定位
    def reposition(self, destination_node):
        self.status = 2 # 重定位
        self.destination_node = destination_node

    # 车辆重定位中途位置转移
    def transfer(self):
        self.node = self.destination_node

    # 完成重定位
    def finish_reposition(self):
        self.status = 0 # 空闲
        self.destination_node = -1

    def in_grids(self, clock):
        self.status = 0
        self.in_grids_clock = clock

    def out_grids(self, clock):
        self.status = 3
        self.working_time += (clock - self.in_grids_clock)
        self.destination_node = -1


class City:
    
    def __init__(self, param):
        self.param = param
        self.done = None # 模拟是否结束
        self.leaving_taxi_number = None # 离开的出租车数量

        # 构建网格相关
        self.grid_number = None
        self.grid_neighbors = None # 网格相邻关系
        self.load_grid_set()
        print("Finish load grids!", self.grid_number)

        # 构建路网相关
        self.node_number = None
        self.hot_nodes = None # 热点路网节点及其热度
        self.hot_nodes_of_grid = None # 网格中的热点
        self.grid_of_nodes = None # 节点对应网格
        self.road_time_cost = None # 路口之间的时间花费
        self.load_node_set()
        print("Finish load nodes!", self.node_number, len(self.hot_nodes.keys()))

        # 初始化时间线相关
        self.clock = None # 时钟
        self.time_line = None # 时间线

        # 初始化出租车相关
        self.taxi_number = self.param.taxi_number
        self.initial_taxi_set = None # 车队初始情况
        self.initial_idle_taxi_of_grids = None # 网格空车分布初始情况
        self.construct_taxi_set()
        self.taxi_set = None # 出租车队
        self.idle_taxi_of_grids = None # 当前时刻的网格空车分布情况
        self.out_taxis = None # 在野出租车
        self.service_number_of_grids = None # 当前网格的服务出租车数量
        # 每个step网格预计到达的新出租车情况
        self.coming_taxi_of_grids = None
        print("Finish construct taxis!", self.taxi_number)

        # 初始化出行相关
        self.trip_number = None
        self.trip_set = None
        self.out_trips = None # 在野出租车流入的出行
        self.load_trip_set()
        self.grid_existing_trip = None # 每个step的存在出行数量
        print("Finish load trips!", self.trip_number)

    # 重置环境，模拟前必须先重置环境
    def reset(self):
        self.done = False # 模拟是否结束
        self.leaving_taxi_number = 0 # 离开的出租车数量

        # 初始化时间线相关
        self.clock = self.param.simulated_start_time # 时钟
        self.time_line = [[] for _ in range(math.ceil(self.param.simulated_end_time / self.param.timeline_interval))] # 时间线

        # 初始化出租车相关
        self.taxi_number = self.param.taxi_number
        # self.construct_taxi_set() # 重新初始化车队位置
        self.taxi_set = copy.deepcopy(self.initial_taxi_set) # 出租车队
        self.idle_taxi_of_grids = copy.deepcopy(self.initial_idle_taxi_of_grids) # 当前时刻的网格空车分布情况
        self.out_taxis = [] # 在野出租车
        self.service_number_of_grids = [0 for _ in range(self.grid_number)] # 当前网格的服务出租车数量
        # 每个step网格预计到达的新出租车情况
        self.coming_taxi_of_grids = [[[] for _ in range(math.ceil(self.param.simulated_end_time / self.param.step_interval))] for _ in range(self.grid_number)]

        # 初始化出行相关
        self.grid_existing_trip = [0 for _ in range(self.grid_number)] # 每个step的存在出行数量
        for trip in self.trip_set:
            trip.reset()
            if self.grid_of_nodes[trip.departure_node] != -1: # 在规定网格中，出行出现
                self.time_line[trip.generation_time // self.param.timeline_interval].append(Event(0, -1, trip.id))

    # 清除环境
    def clear(self):
        # 清除网格相关
        self.grid_neighbors.clear()
        # 清除路网相关
        self.hot_nodes.clear()
        self.hot_nodes_of_grid.clear()
        self.grid_of_nodes.clear()
        self.road_time_cost.clear()
        # 清除化时间线相关
        self.time_line.clear()
        # 初始化出租车相关
        self.initial_taxi_set.clear()
        self.taxi_set.clear()
        self.initial_idle_taxi_of_grids.clear()
        self.idle_taxi_of_grids.clear()
        self.out_taxis.clear()
        self.service_number_of_grids.clear()
        self.coming_taxi_of_grids.clear()
        # 清除出行相关
        self.trip_set.clear()
        self.grid_existing_trip.clear()

    # 加载城市栅格
    def load_grid_set(self):
        df = pd.read_csv(self.param.file_path2+'grid_neighbors.csv', header=0)
        self.grid_number = len(df)
        # 加载网格相邻关系
        self.grid_neighbors = []
        for index, row in df.iterrows():
            self.grid_neighbors.append([index, row['neighbor1'], row['neighbor2'], row['neighbor3'], row['neighbor4'], row['neighbor5'], row['neighbor6'],])

    # 加载路网
    def load_node_set(self):
        df1 = pd.read_csv(self.param.file_path2+'all_nodes.csv', header=0)
        self.node_number = 0
        self.grid_of_nodes = [] # 节点对应网格(转换后的)
        for index, row in df1.iterrows():
            self.grid_of_nodes.append(row['grid_id'])
            if row['grid_id'] != -1: # 在网格中且有出行的路口
                self.node_number += 1

        df2 = pd.read_csv(self.param.file_path2+'hot_nodes.csv', header=0)
        self.hot_nodes = {} # 热点路网节点及其热度
        self.hot_nodes_of_grid = [[] for _ in range(self.grid_number)] # 每个网格包含的热点节点
        for index, row in df2.iterrows():
            self.hot_nodes[row['node_id']] = row['departure_number']
            self.hot_nodes_of_grid[row['grid_id']].append(row['node_id'])
        
        # 读取路网时间花费
        with open(self.param.file_path2+'cost_time.pkl', 'rb') as f:
            self.road_time_cost = pickle.load(f)
    
    # 构建出租车队
    def construct_taxi_set(self):
        self.initial_taxi_set = []
        self.initial_idle_taxi_of_grids = [[] for _ in range(self.grid_number)]
        hot_nodes_keys = [key for key in self.hot_nodes.keys()]
        for i in range(self.taxi_number):
            # 随机选择一个热点路口
            random_node = hot_nodes_keys[random.randint(0, len(hot_nodes_keys)-1)]
            random_grid = self.grid_of_nodes[random_node]
            taxi = Taxi(i, random_node)
            taxi.in_grids(self.param.simulated_start_time)
            self.initial_taxi_set.append(taxi)
            self.initial_idle_taxi_of_grids[random_grid].append(i)
    
    # 加载出行集合
    def load_trip_set(self):
        df = pd.read_csv(self.param.file_path1+'trips.csv', header=0)
        self.trip_number = 0
        self.trip_set = []
        self.out_trips = [[] for _ in range(math.ceil(self.param.simulated_end_time / self.param.step_interval))]
        for index, row in df.iterrows():
            departure_node = int(row['departure_node'])
            arrival_node = int(row['arrival_node'])
            generation_time = int(row['departure_time'])
            arrival_time = generation_time + int(row['cost_time'])
            # 除去部分不在规定时间的出行,只看出发时间
            if generation_time >= self.param.simulated_end_time: break
            # 出发到达点均不在规定网格中
            if self.grid_of_nodes[departure_node] == -1 and self.grid_of_nodes[arrival_node] == -1: continue
            # 出发点在规定网格中
            if self.grid_of_nodes[departure_node] != -1 and generation_time < self.param.simulated_start_time: continue
            # 出发点不在规定网格中，但到达点在，车辆流入
            if self.grid_of_nodes[departure_node] == -1:
                if arrival_time >= self.param.simulated_start_time and arrival_time < self.param.simulated_end_time:
                    self.out_trips[arrival_time // self.param.step_interval].append(self.trip_number)
                else: continue
            self.trip_set.append(Trip(self.trip_number, generation_time, departure_node, arrival_node, int(row['cost_time']), row['amount']))
            self.trip_number += 1

    # 在野出租车流入
    def taxi_inflow(self):
        for i in self.out_trips[self.clock // self.param.step_interval]:
            trip = self.trip_set[i]
            departure_node = trip.departure_node
            arrival_node = trip.arrival_node
            arrival_grid = self.grid_of_nodes[arrival_node]
            arrival_time = trip.generation_time + trip.cost_time
            if len(self.out_taxis) == 0: # 在野车辆不足
                self.taxi_set.append(Taxi(self.taxi_number, departure_node))
                self.out_taxis.append(self.taxi_number)
                self.taxi_number += 1
            taxi_id = self.out_taxis.pop(0)
            self.taxi_set[taxi_id].node = arrival_node
            self.time_line[arrival_time // self.param.timeline_interval].append(Event(6, taxi_id, trip.id))
            self.coming_taxi_of_grids[arrival_grid][arrival_time // self.param.step_interval].append(taxi_id)

    def step(self):
        self.grid_existing_trip = [0 for _ in range(self.grid_number)] # 每个step重置
        times_in_step = self.param.step_interval // self.param.timeline_interval # 一个时间步的处理次数
        for _ in range(times_in_step): # 处理时刻事件
            event_set = self.time_line[self.clock // self.param.timeline_interval]
            for event in event_set:
                if event.type == 0 or event.type == 1: # 出行出现
                    trip = self.trip_set[event.trip_id]
                    departure_node = trip.departure_node
                    departure_grid = self.grid_of_nodes[departure_node]
                    arrival_node = trip.arrival_node
                    arrival_grid = self.grid_of_nodes[arrival_node]
                    if event.type == 0: self.grid_existing_trip[departure_grid] += 1 # 首次出现记录
                    # 搜索网格的空闲出租车并等待响应
                    if len(self.idle_taxi_of_grids[departure_grid]) == 0: # 无车, 推迟响应
                        next_clock = self.clock + self.param.trip_response_interval
                        if next_clock <= trip.generation_time + self.param.trip_tolerance_time: # 容忍限度内
                            self.trip_set[event.trip_id].response_time = next_clock - trip.generation_time # 更新等待时间
                            if next_clock < self.param.simulated_end_time:
                                # 如推迟到下一个step, 便算首次出现
                                if next_clock >= self.clock + self.param.step_interval:
                                    self.time_line[next_clock // self.param.timeline_interval].append(Event(0, -1, trip.id))
                                else: self.time_line[next_clock // self.param.timeline_interval].append(Event(1, -1, trip.id))
                        else: # 出行被取消
                            self.trip_set[event.trip_id].cancel(self.clock)
                    else: # 立即响应
                        taxi_id = self.idle_taxi_of_grids[departure_grid].pop(0) # 队首空闲车辆
                        if arrival_grid == -1: # 到达地在野
                            reach_time = self.param.step_interval // 2
                            self.taxi_set[taxi_id].out_grids(self.clock + reach_time + trip.cost_time)
                            self.taxi_set[taxi_id].node = arrival_node
                            self.out_taxis.append(taxi_id)
                            self.trip_set[event.trip_id].responsed(taxi_id, self.clock, reach_time) # 出行被响应
                            self.trip_set[event.trip_id].finish()
                            # self.leaving_taxi_number += 1
                            continue
                        self.service_number_of_grids[departure_grid] += 1
                        reach_time = self.road_time_cost[(self.taxi_set[taxi_id].node, departure_node)] # 接人需要的时间
                        if reach_time == -1: reach_time = self.param.step_interval
                        self.taxi_set[taxi_id].receive_trip(trip.id) # 出租车响应
                        self.trip_set[event.trip_id].responsed(taxi_id, self.clock, reach_time) # 出行被响应
                        # 出行完成
                        arrival_time = self.clock + reach_time + trip.cost_time
                        if arrival_time < self.param.simulated_end_time:
                            self.time_line[arrival_time // self.param.timeline_interval].append(Event(2, taxi_id, trip.id))
                            self.coming_taxi_of_grids[arrival_grid][arrival_time // self.param.step_interval].append(taxi_id)
                elif event.type == 2 or event.type == 6: # 出行完成或在野出租车到达
                    trip = self.trip_set[event.trip_id]
                    departure_node = trip.departure_node
                    departure_grid = self.grid_of_nodes[departure_node]
                    arrival_node = trip.arrival_node
                    arrival_grid = self.grid_of_nodes[arrival_node]
                    self.taxi_set[event.taxi_id].finish_trip(arrival_node)
                    self.idle_taxi_of_grids[arrival_grid].append(event.taxi_id)
                    if event.type == 2:
                        self.service_number_of_grids[departure_grid] -= 1
                        self.trip_set[event.trip_id].finish()
                    elif event.type == 6:
                        self.taxi_set[event.taxi_id].in_grids(self.clock)
                    # 完成后进行重定位
                    target_node = self.taxi_set[event.taxi_id].destination_node
                    if target_node != -1:
                        cost_time = self.road_time_cost[(self.taxi_set[event.taxi_id].node, target_node)]
                        if cost_time == -1:
                            cost_time = self.param.step_interval
                        self.taxi_set[event.taxi_id].reposition(target_node)
                        # 赶到下个网格
                        transfer_clock = self.clock + cost_time // 2
                        if transfer_clock < self.param.simulated_end_time:
                            self.time_line[transfer_clock // self.param.timeline_interval].append(Event(4, event.taxi_id, -1))
                        # 重定位完成
                        finish_clock = self.clock + cost_time
                        if finish_clock < self.param.simulated_end_time:
                            self.time_line[finish_clock // self.param.timeline_interval].append(Event(5, event.taxi_id, -1))
                elif event.type == 4: # 车辆重定位中途位置转移
                    taxi = self.taxi_set[event.taxi_id]
                    if taxi.status != 2: continue # 不是重定位状态
                    now_gird = self.grid_of_nodes[taxi.node]
                    destination_grid = self.grid_of_nodes[taxi.destination_node]
                    self.idle_taxi_of_grids[now_gird].remove(taxi.id)
                    self.idle_taxi_of_grids[destination_grid].append(taxi.id)
                    self.taxi_set[taxi.id].transfer()
                elif event.type == 5: # 重定位完成
                    taxi = self.taxi_set[event.taxi_id]
                    if taxi.status != 2: continue # 不是重定位状态
                    self.taxi_set[taxi.id].finish_reposition()

            self.clock += self.param.timeline_interval # 时钟改变
        
        if self.clock >= self.param.simulated_end_time:
            self.done = True

    # 总响应视图
    def rate_view(self):
        total_generation_trip_number = self.trip_number - len(self.out_trips) # 总生成出行数
        total_response_trip_number = 0 # 总响应出行数
        total_response_time = 0 # 总响应时间
        total_cost_time = 0 # 总出行花费时间
        total_amount = 0 # 总出行金额
        for trip in self.trip_set:
            # 不统计出发点在野的出行
            if self.grid_of_nodes[trip.departure_node] == -1: continue
            if trip.status == 1 or trip.status == 2: # 响应或已完成
                total_response_trip_number += 1
            if trip.status == 2: # 出行已完成
                total_cost_time += trip.cost_time
                total_amount += trip.amount
            elif trip.status == 1: # 出行还在进行
                t = self.clock - trip.generation_time - trip.response_time - trip.reach_time
                if t > 0: total_cost_time += t # 已接到人
            total_response_time += trip.response_time

        # 总出租车工作时间
        total_taxi_time = self.param.taxi_number * (self.param.simulated_end_time - self.param.simulated_start_time)
        # total_taxi_time = 0
        # for taxi in self.taxi_set:
        #     if taxi.status != 3:
        #         taxi.working_time += (self.param.simulated_end_time - taxi.in_grids_clock)
        #     total_taxi_time += taxi.working_time

        # 平均响应率 平均响应时间 出租车占用率 GMV
        return total_response_trip_number/total_generation_trip_number, total_response_time/total_generation_trip_number, total_cost_time/total_taxi_time, total_amount/total_taxi_time * 60 * 60
    
    # 每个网格响应视图
    def every_rate_view(self):
        grid_generation_trip_number = [0 for _ in range(self.grid_number)] # 每个网格的出现出行数
        grid_response_trip_number = [0 for _ in range(self.grid_number)] # 每个网格的响应出行数
        grid_response_time = [0 for _ in range(self.grid_number)] # 每个网格的响应时间
        grid_reach_time = [0 for _ in range(self.grid_number)] # 每个网格的接取乘客时间
        grid_cost_time = [0 for _ in range(self.grid_number)] # 每个网格的出行花费时间
        grid_amount = [0 for _ in range(self.grid_number)] # 每个网格的出行金额
        for trip in self.trip_set:
            if trip.generation_time < self.clock:
                grid_index = self.grid_of_nodes[trip.departure_node]
                grid_generation_trip_number[grid_index] += 1
                if trip.status == 1 or trip.status == 2: # 响应或已完成
                    grid_response_trip_number[grid_index] += 1
                    grid_reach_time[grid_index] += trip.reach_time
                if trip.status == 2: # 出行已完成
                    grid_cost_time[grid_index] += trip.cost_time
                    grid_amount[grid_index] += trip.amount
                elif trip.status == 1: # 出行还在进行
                    t = self.clock - trip.generation_time - trip.response_time - trip.reach_time
                    if t > 0: grid_cost_time[grid_index] += t # 已接到人
                grid_response_time[grid_index] += trip.response_time
        total_generation_trip_number = 0
        total_response_trip_number = 0
        total_response_time = 0
        total_reach_time = 0
        total_cost_time = 0
        view_list = []
        for i in range(self.grid_number): # 汇总
            total_generation_trip_number += grid_generation_trip_number[i]
            total_response_trip_number += grid_response_trip_number[i]
            total_response_time += grid_response_time[i]
            total_reach_time += grid_reach_time[i]
            total_cost_time += grid_cost_time[i]
            rate_i = 0
            ave_response_time_i = 0
            if grid_generation_trip_number[i] != 0:
                rate_i = grid_response_trip_number[i] / grid_generation_trip_number[i]
                ave_response_time_i = grid_response_time[i] / grid_generation_trip_number[i]
            view_list.append([rate_i, ave_response_time_i])

        return view_list
    
# 多对多调度方式
class ManyToManyManner(City):

    def __init__(self, param):
        City.__init__(self, param)
        self.action_dim = 7 # 7个动作
        self.state_dim = self.grid_number * 3 # 所有网格的出行、空车、服务车数量
        self.observation_dim = 7 * 3 # 自身和临近网格的出行、空车、服务车数量
        self.state = None # 环境状态
        self.observations = None # 每个网格的观测状态
        self.raw_reward = None # 平衡度奖励
        self.rewards = None # 联合奖励

    def reset(self):
        super().reset()
        self.state = [0 for _ in range(self.state_dim)] # 环境状态
        self.observations = [[0 for _ in range(self.observation_dim)] for __ in range(self.grid_number)] # 每个网格的观测状态
        self.raw_reward = [0 for _ in range(self.grid_number)] # 平衡度奖励
        self.rewards = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.grid_number)] # 联合奖励
        super().taxi_inflow()
        super().step()
        self.update_taxi_state() # 更新环境出租车状态
        self.update_trip_state() # 更新环境出行状态
        self.update_observations() # 更新观测状态

        return self.observations, self.state
    
    # 执行动作, 对网格车辆进行调度, 随机调度
    def take_actions_randomly(self, actions):
        for i in range(self.grid_number):
            action = actions[i]
            idle_taxi_number = len(self.idle_taxi_of_grids[i])
            service_taxi_number = len(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval])
            for j in range(1,7): # 6个相邻网格
                neighbor_grid_index = self.grid_neighbors[i][j]
                # 空车调度
                reposition_number1 = int(idle_taxi_number * action[j])
                for _ in range(reposition_number1):
                    taxi_id = self.idle_taxi_of_grids[i].pop(0) # 取一辆车
                    if neighbor_grid_index == -1: # 调度到在野网格
                        self.taxi_set[taxi_id].out_grids(self.clock)
                        self.taxi_set[taxi_id].node = -1
                        self.out_taxis.append(taxi_id)
                        self.leaving_taxi_number += 1
                    else:
                        # 调整列表中的顺位，在到达下一个网格前，还能响应当前网格的出行
                        self.idle_taxi_of_grids[i].append(taxi_id)
                        # 重定位目的地路口,即目的网格的随机热点
                        target_node = self.hot_nodes_of_grid[neighbor_grid_index][random.randint(0, len(self.hot_nodes_of_grid[neighbor_grid_index])-1)]
                        cost_time = self.road_time_cost[(self.taxi_set[taxi_id].node, target_node)]
                        if cost_time == -1:
                            cost_time = self.param.step_interval
                        self.taxi_set[taxi_id].reposition(target_node)
                        # 赶到下个网格
                        transfer_clock = self.clock + cost_time // 2
                        if transfer_clock < self.param.simulated_end_time:
                            self.time_line[transfer_clock // self.param.timeline_interval].append(Event(4, taxi_id, -1))
                        # 重定位完成
                        finish_clock = self.clock + cost_time
                        if finish_clock < self.param.simulated_end_time:
                            self.time_line[finish_clock // self.param.timeline_interval].append(Event(5, taxi_id, -1))

                # 服务车调度
                if neighbor_grid_index == -1: continue
                reposition_number2 = int(service_taxi_number * action[j])
                for _ in range(reposition_number2):
                    taxi_id = self.coming_taxi_of_grids[i][self.clock // self.param.step_interval].pop(0) # 取一辆车
                    # 调整列表中的顺位，在到达下一个网格前，还能响应当前网格的出行
                    self.coming_taxi_of_grids[i][self.clock // self.param.step_interval].append(taxi_id)
                    # 重定位目的地路口,即目的网格的随机热点
                    target_node = self.hot_nodes_of_grid[neighbor_grid_index][random.randint(0, len(self.hot_nodes_of_grid[neighbor_grid_index])-1)]
                    self.taxi_set[taxi_id].destination_node = target_node

    def taxi_min_cost_flow(self, candidate_taxis, candidate_nodes, candidate_grids, candidate_number):
        start_nodes = [] # 每条边的开始节点
        end_nodes = [] # 每条边的结束节点
        capacities = [] # 每条边的容量
        unit_costs = [] # 每条边的花费
        supplies = [] # 每个节点的供需，正数为供给，负数为需求
        n1 = len(candidate_taxis)
        n2 = len(candidate_nodes)
        n3 = len(candidate_grids)
        for i in range(n1): # 出租车与路口连边
            supplies.append(0)
            snode = self.taxi_set[candidate_taxis[i]].node
            for j in range(n2):
                start_nodes.append(i)
                end_nodes.append(n1 + j)
                capacities.append(1)
                unit_costs.append(self.road_time_cost[(snode, candidate_nodes[j])])
        for j in range(n2): # 路口与网格连边
            supplies.append(0)
            grid_id = self.grid_of_nodes[candidate_nodes[j]]
            k = candidate_grids.index(grid_id)
            start_nodes.append(n1 + j)
            end_nodes.append(n1 + n2 + k)
            capacities.append(candidate_number[k])
            unit_costs.append(0)
        for i in range(n1): # 源与出租车连边
            start_nodes.append(n1 + n2 + n3)
            end_nodes.append(i)
            capacities.append(1)
            unit_costs.append(0)
        for k in range(n3): 
            supplies.append(-candidate_number[k])
        supplies.append(sum(candidate_number))

        # 构建网络流
        smcf = min_cost_flow.SimpleMinCostFlow()
        # 添加边及其容量和花费
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(np.array(start_nodes), np.array(end_nodes), np.array(capacities), np.array(unit_costs))
        # 添加节点及其供需
        smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)
        status = smcf.solve() # 求解最小花费流
        if status != smcf.OPTIMAL:
            print("There was an issue with the min cost flow input.")
        solution_flows = smcf.flows(all_arcs)
        return solution_flows

    # 执行动作, 对网格车辆进行调度, 最小费用流
    def take_actions(self, actions):
        for i in range(self.grid_number):
            action = actions[i]
            # 空车调度
            idle_taxi_number = len(self.idle_taxi_of_grids[i])
            candidate_taxis = [] # 候选出租车
            candidate_nodes = [] # 候选路口
            candidate_grids = [] # 候选网格
            candidate_number = [] # 候选数量
            for j in range(1,7): # 添加候选网格和数量
                neighbor_grid_index = self.grid_neighbors[i][j]
                reposition_number = int(idle_taxi_number * action[j])
                if neighbor_grid_index == -1: # 调度到在野网格
                    for _ in range(reposition_number):
                        taxi_id = self.idle_taxi_of_grids[i].pop(0) # 取一辆车
                        self.taxi_set[taxi_id].out_grids(self.clock)
                        self.taxi_set[taxi_id].node = -1
                        self.out_taxis.append(taxi_id)
                        self.leaving_taxi_number += 1
                elif reposition_number > 0:
                    candidate_grids.append(neighbor_grid_index) # 加入候选
                    candidate_number.append(reposition_number) # 加入候选
            for taxi in self.idle_taxi_of_grids[i]:
                candidate_taxis.append(taxi)
            # candidate_taxis = copy.deepcopy(self.idle_taxi_of_grids[i]) # 加入候选出租车
            for grid in candidate_grids: # 添加候选路口
                for node in self.hot_nodes_of_grid[grid]:
                    candidate_nodes.append(node)
            if len(candidate_taxis) > 0: # 有车调度
                solution_flows = self.taxi_min_cost_flow(candidate_taxis, candidate_nodes, candidate_grids, candidate_number)
                n1 = len(candidate_taxis)
                n2 = len(candidate_nodes)
                for x in range(n1):
                    for y in range(n2):
                        if solution_flows[x * n2 + y] == 0: continue
                        # 重定位目的地路口,即目的网格的随机热点
                        taxi_id = candidate_taxis[x]
                        target_node = candidate_nodes[y]
                        cost_time = self.road_time_cost[(self.taxi_set[taxi_id].node, target_node)]
                        if cost_time == -1:
                            cost_time = self.param.step_interval
                        self.taxi_set[taxi_id].reposition(target_node)
                        # 赶到下个网格
                        transfer_clock = self.clock + cost_time // 2
                        if transfer_clock < self.param.simulated_end_time:
                            self.time_line[transfer_clock // self.param.timeline_interval].append(Event(4, taxi_id, -1))
                        # 重定位完成
                        finish_clock = self.clock + cost_time
                        if finish_clock < self.param.simulated_end_time:
                            self.time_line[finish_clock // self.param.timeline_interval].append(Event(5, taxi_id, -1))

            # 服务车调度
            service_taxi_number = len(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval])
            candidate_taxis = [] # 候选出租车
            candidate_nodes = [] # 候选路口
            candidate_grids = [] # 候选网格
            candidate_number = [] # 候选数量
            for j in range(1,7):
                neighbor_grid_index = self.grid_neighbors[i][j]
                if neighbor_grid_index == -1: continue
                reposition_number = int(service_taxi_number * action[j])
                if reposition_number > 0:
                    candidate_grids.append(neighbor_grid_index) # 加入候选
                    candidate_number.append(reposition_number) # 加入候选
            candidate_taxis = copy.deepcopy(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval]) # 加入候选出租车
            for grid in candidate_grids: # 添加候选路口
                for node in self.hot_nodes_of_grid[grid]:
                    candidate_nodes.append(node)
            if len(candidate_taxis) > 0: # 有车调度
                solution_flows = self.taxi_min_cost_flow(candidate_taxis, candidate_nodes, candidate_grids, candidate_number)
                n1 = len(candidate_taxis)
                n2 = len(candidate_nodes)
                for x in range(n1):
                    for y in range(n2):
                        if solution_flows[x * n2 + y] == 0: continue
                        taxi_id = candidate_taxis[x]
                        target_node = candidate_nodes[y]
                        self.taxi_set[taxi_id].destination_node = target_node


    def step(self, actions):
        super().taxi_inflow()
        self.take_actions(actions)
        super().step()
        self.update_taxi_state() # 更新环境出租车状态
        self.update_trip_state() # 更新环境出行状态
        self.update_raw_rewards() # 更新平衡度奖励
        self.update_rewards(actions) # 更新联合奖励
        self.update_observations() # 更新观测状态

        return self.observations, self.state, self.raw_reward, self.rewards, self.done
    
    # 更新观测状态
    def update_observations(self):
        for i in range(self.grid_number):
            for j in range(7):
                neighbor_grid_index = self.grid_neighbors[i][j]
                if neighbor_grid_index == -1: # 无该网格
                    self.observations[i][3*j] = 0
                    self.observations[i][3*j+1] = 0
                    self.observations[i][3*j+2] = 0
                else:
                    self.observations[i][3*j] = self.grid_existing_trip[neighbor_grid_index]
                    self.observations[i][3*j+1] = len(self.idle_taxi_of_grids[neighbor_grid_index])
                    self.observations[i][3*j+2] = len(self.coming_taxi_of_grids[neighbor_grid_index][self.clock // self.param.step_interval -1])
    
    # 更新环境出租车状态
    def update_taxi_state(self):
        for i in range(self.grid_number):
            self.state[3*i+1] = len(self.idle_taxi_of_grids[i])
            self.state[3*i+2] = len(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval -1])

    # 更新环境出行状态
    def update_trip_state(self):
        for i in range(self.grid_number):
            self.state[3*i] = self.grid_existing_trip[i]

    def omega(self, s, d, r):
        if s == 0 and d == 0: return r
        if r < 0:
            return r * s
        else: return r * d

    # 更新平衡度奖励
    def update_raw_rewards(self):
        for i in range(self.grid_number):
            trip_number = self.state[3*i]
            taxi_number = self.state[3*i+1] + int(self.state[3*i+2] * self.param.taxi_to_idle_weight)
            if trip_number == 0 and taxi_number == 0: r = 1.0
            else: r = (1.0 - float(abs(trip_number-taxi_number))/max(trip_number,taxi_number,1))
            self.raw_reward[i] = r
    
    # 计算网格联合奖励
    def caculate_r(self, i, j):
        if i == j:
            trip_number = self.state[3*i]
            taxi_number = self.state[3*i+1] + int(self.state[3*i+2] * self.param.taxi_to_idle_weight)
            if trip_number == 0: rate = taxi_number
            else: rate = taxi_number / trip_number
            r = 1.0 - rate
            return self.omega(taxi_number, trip_number, r)
        else:
            trip_number_i = self.state[3*i]
            taxi_number_i = self.state[3*i+1] + int(self.state[3*i+2] * self.param.taxi_to_idle_weight)
            if trip_number_i == 0: rate_i = taxi_number_i
            else: rate_i = taxi_number_i / trip_number_i
            trip_number_j = self.state[3*j]
            taxi_number_j = self.state[3*j+1] +  int(self.state[3*j+2] * self.param.taxi_to_idle_weight)
            if trip_number_j == 0: rate_j = taxi_number_j
            else: rate_j = taxi_number_j / trip_number_j
            r_i = rate_i - 1.0
            r_j = 1.0 - rate_j
            return self.omega(taxi_number_i, trip_number_i, r_i) + self.omega(taxi_number_j, trip_number_j, r_j)
    
    # 更新联合奖励
    def update_rewards(self, actions):
        for i in range(self.grid_number):
            for j in range(self.action_dim):
                neighbor_grid_index = self.grid_neighbors[i][j]
                if neighbor_grid_index == -1: # 无该网格
                    # self.rewards[i][j] = -1000.0
                    self.rewards[i][j] = -1000.0 * actions[i][j]
                else:
                    r = self.caculate_r(i, neighbor_grid_index)
                    self.rewards[i][j] = r


# 一对一调度方式
class OneToOneManner(City):

    def __init__(self, param):
        City.__init__(self, param)
        self.action_dim = 1
        self.state_dim = 1 # 每个网格的状态
        self.state = None # 环境状态
        self.balance = None # 平衡度
        self.rewards = None # 奖励
        
        self.distance = self.grid_distance() # 网格间距离


    def reset(self):
        super().reset()
        self.state = [[0 for _ in range(self.state_dim)] for _ in range(self.grid_number)] # 环境状态
        self.balance = [0 for _ in range(self.grid_number)] # 平衡度
        self.rewards = [0 for _ in range(self.grid_number)] # 奖励
        super().taxi_inflow()
        super().step()
        self.update_state()

        return self.state
    
    # 执行动作, 对网格车辆进行调度
    def take_actions(self, actions):
        for i in range(self.grid_number):
            action = actions[i]
            if action[0] > 1: action[0] = 1 # 避免训练时随机溢出
            target_grid = action[1] # 目的地网格
            if action[0] <= 0 or target_grid == -1: continue # 不用调出车辆
            # 空车调度
            reposition_number1 = int(len(self.idle_taxi_of_grids[i]) * action[0])
            for _ in range(reposition_number1):
                taxi_id = self.idle_taxi_of_grids[i].pop(0) # 取一辆车
                # 调整列表中的顺位，在到达下一个网格前，还能响应当前网格的出行
                self.idle_taxi_of_grids[i].append(taxi_id)
                # 重定位目的地路口,即目的网格的随机热点
                target_node = self.hot_nodes_of_grid[target_grid][random.randint(0, len(self.hot_nodes_of_grid[target_grid])-1)]
                cost_time = self.road_time_cost[(self.taxi_set[taxi_id].node, target_node)]
                if cost_time == -1:  
                    cost_time = self.param.step_interval * self.distance[i][target_grid]
                self.taxi_set[taxi_id].reposition(target_node)
                # 赶到下个网格
                transfer_clock = self.clock + cost_time // 2
                if transfer_clock < self.param.simulated_end_time:
                    self.time_line[transfer_clock // self.param.timeline_interval].append(Event(4, taxi_id, -1))
                # 重定位完成
                finish_clock = self.clock + cost_time
                if finish_clock < self.param.simulated_end_time:
                    self.time_line[finish_clock // self.param.timeline_interval].append(Event(5, taxi_id, -1))
            # 服务车调度
            reposition_number2 = int(len(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval]) * action[0])
            for _ in range(reposition_number2):
                taxi_id = self.coming_taxi_of_grids[i][self.clock // self.param.step_interval].pop(0) # 取一辆车
                # 调整列表中的顺位，在到达下一个网格前，还能响应当前网格的出行
                self.coming_taxi_of_grids[i][self.clock // self.param.step_interval].append(taxi_id)
                # 重定位目的地路口,即目的网格的随机热点
                target_node = self.hot_nodes_of_grid[target_grid][random.randint(0, len(self.hot_nodes_of_grid[target_grid])-1)]
                self.taxi_set[taxi_id].destination_node = target_node
    
    def step(self, actions):
        super().taxi_inflow()
        strategy = self.taxi_dispatch(actions, [len(x) for x in self.idle_taxi_of_grids], self.grid_existing_trip)
        self.take_actions(strategy)
        super().step()
        self.update_state() # 更新环境状态
        self.update_rewards() # 更新奖励

        return self.state, self.rewards, self.balance, strategy, self.done
    
    # 更新环境状态
    def update_state(self):
        for i in range(self.grid_number):
            service_taxi_number = int(len(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval -1]) * self.param.taxi_to_idle_weight)
            self.state[i][0] = len(self.idle_taxi_of_grids[i]) + service_taxi_number - self.grid_existing_trip[i]

    # 更新环境奖励
    def update_rewards(self):
        for i in range(self.grid_number):
            service_taxi_number = int(len(self.coming_taxi_of_grids[i][self.clock // self.param.step_interval -1]) * self.param.taxi_to_idle_weight)
            taxi_number = len(self.idle_taxi_of_grids[i]) + service_taxi_number
            trip_number = self.grid_existing_trip[i]
            if trip_number == 0 and taxi_number == 0: r = 1.0
            else: r = (1.0 - float(abs(trip_number - taxi_number)) / max(trip_number,taxi_number))
            self.rewards[i] = r - self.balance[i]
            self.balance[i] = r

    # Meta 分配供需的策略模块

    # 计算网格间距离
    def grid_distance(self):
        g = igraph.Graph() # 创建一个无向图
        g.add_vertices(len(self.grid_neighbors)) # 添加节点
        for i in range(len(self.grid_neighbors)): # 添加边
            for j in range(1, 7):
                if self.grid_neighbors[i][j] != -1:
                    g.add_edge(i, self.grid_neighbors[i][j])
        return g.distances(None, None, None, 'all')

    # 分配供需
    def taxi_dispatch(self, actions, grid_taxis, grid_trips):
        l_nodes = [] # 二分图左节点集
        r_nodes = [] # 二分图右节点集
        for i in range(self.grid_number):
            if actions[i] * grid_taxis[i] > 1:
                l_nodes.append(i)
            elif actions[i] * grid_trips[i] < -1:
                r_nodes.append(i)

        values = []
        for i in range(len(l_nodes)):
            for j in range(len(r_nodes)):
                x = l_nodes[i]
                y = r_nodes[j]
                d = -1.0 * self.distance[x][y] * abs(abs(actions[x] * grid_taxis[x]) - abs(actions[y] * grid_trips[y]))
                values.append((i, j, d))
        
        matchs = run_kuhn_munkres(values)
        strategy = [[actions[i], -1] for i in range(self.grid_number)]
        for match in matchs:
            x = l_nodes[match[0]]
            y = r_nodes[match[1]]
            strategy[x][1] = y
            strategy[y][1] = x

        return strategy


# KM 算法(递归版) https://github.com/xyxYang/Kuhn_Munkres_Algorithm/tree/master
zero_threshold = 0.00000001

class KMNode(object):
    def __init__(self, id, exception=0, match=None, visit=False):
        self.id = id
        self.exception = exception
        self.match = match
        self.visit = visit


class KuhnMunkres(object):
    def __init__(self):
        self.matrix = None
        self.x_nodes = []
        self.y_nodes = []
        self.minz = float('inf')
        self.x_length = 0
        self.y_length = 0
        self.index_x = 0
        self.index_y = 1

    def __del__(self):
        pass

    def set_matrix(self, x_y_values):
        xs = set()
        ys = set()
        for x, y, value in x_y_values:
            xs.add(x)
            ys.add(y)

        #选取较小的作为x
        if len(xs) < len(ys):
            self.index_x = 0
            self.index_y = 1
        else:
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs

        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length = len(xs)
        self.y_length = len(ys)

        self.matrix = np.zeros((self.x_length, self.y_length))
        for row in x_y_values:
            x = row[self.index_x]
            y = row[self.index_y]
            value = row[2]
            x_index = x_dic[x]
            y_index = y_dic[y]
            self.matrix[x_index, y_index] = value

        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])


    def km(self):
        for i in range(self.x_length):
            while True:
                self.minz = float('inf')
                self.set_false(self.x_nodes)
                self.set_false(self.y_nodes)

                if self.dfs(i):
                    break

                self.change_exception(self.x_nodes, -self.minz)
                self.change_exception(self.y_nodes, self.minz)

    def dfs(self, i):
        x_node = self.x_nodes[i]
        x_node.visit = True
        for j in range(self.y_length):
            y_node = self.y_nodes[j]
            if not y_node.visit:
                t = x_node.exception + y_node.exception - self.matrix[i][j]
                if abs(t) < zero_threshold:
                    y_node.visit = True
                    if y_node.match is None or self.dfs(y_node.match):
                        x_node.match = j
                        y_node.match = i
                        return True
                else:
                    if t >= zero_threshold:
                        self.minz = min(self.minz, t)
        return False

    def set_false(self, nodes):
        for node in nodes:
            node.visit = False

    def change_exception(self, nodes, change):
        for node in nodes:
            if node.visit:
                node.exception += change

    def get_connect_result(self):
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            y_node = self.y_nodes[j]
            x_id = x_node.id
            y_id = y_node.id
            value = self.matrix[i][j]

            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append((x_id, y_id, value))

        return ret

    def get_max_value_result(self):
        ret = 0
        for i in range(self.x_length):
            j = self.x_nodes[i].match
            ret += self.matrix[i][j]

        return ret


def run_kuhn_munkres(x_y_values):
    process = KuhnMunkres()
    process.set_matrix(x_y_values)
    process.km()
    return process.get_connect_result()