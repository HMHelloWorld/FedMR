import numpy as np
from utils.asynchronous_client_config import *


class Clients:
    def __init__(self, args):
        self.args = args
        uncertain_list = [0.2,0.2,0.2,0.2,0.2]
        if args.uncertain_type == 1:
            uncertain_list = [0.5,0.2,0.1,0.1,0.1]
        elif args.uncertain_type == 2:
            uncertain_list = [0.1,0.15,0.5,0.15,0.1]
        elif args.uncertain_type == 3:
            uncertain_list = [0.1,0.1,0.1,0.2,0.5]
        elif args.uncertain_type == 4:
            uncertain_list = [0.4,0.1,0.0,0.1,0.4]
        self.clients_list = generate_asyn_clients(uncertain_list, uncertain_list, args.num_users)
        self.update_list = []  # (idx, version, time)
        self.train_set = set()
        # for i in range(self.args.num_users):
        #     self.clients_list.append(Node(1.4, 1.4, np.random.exponential(), 0, args))

    def train(self, idx, version):
        for i in range(len(self.update_list) - 1, -1, -1):
            if self.update_list[i][0] == idx:
                self.update_list.pop(i)
        client = self.get(idx)
        client.version = version
        client.comm_count += 1
        train_time = client.get_train_time()
        comm_time = client.get_comm_time()
        self.update_list.append([idx, version, train_time + comm_time])
        self.update_list.sort(key=lambda x: x[2])
        self.train_set.add(idx)

    def get_update_byLimit(self, limit):
        lst = []
        for update in self.update_list:
            if update[2] <= limit:
                lst.append(update)
        return lst
        # update = []
        # for i in range(self.args.num_users):
        #     if self.get(i).end_time <= ddl:
        #         update.append((i, self.get(i).end_time))
        # update.sort(key=lambda x: x[1])
        # return update

    def get_update(self, num):
        return self.update_list[0:num]

    def pop_update(self, num):
        res = self.update_list[0:num]
        max_time = self.update_list[num - 1][2]
        for update in self.update_list:
            if update[2] <= max_time:
                self.train_set.remove(update[0])
                client = self.get(update[0])
                client.comm_count += 1
            else:
                update[2] -= max_time
        self.update_list = self.update_list[num::]
        return res

    def get_first_update(self, start_time):
        min_idx = 0
        min_time = 999999999999
        for idx in range(self.args.num_users):
            client = self.get(idx)
            if client.end_time != 0:
                if start_time < client.end_time < min_time:
                    min_time = client.end_time
                    min_idx = idx
        return min_idx

    def get(self, idx):
        return self.clients_list[idx]

    def get_idle(self, num):
        idle = self.get_all_idle()

        if len(idle) < num:
            return []
        else:
            return np.random.choice(idle, num, replace=False)

    def get_all_idle(self):
        idle = set(range(self.args.num_users)).difference(self.train_set)
        return list(idle)
        # idle = []
        # for idx in range(self.args.num_users):
        #     client = self.get(idx)
        #     if not (client.start_time <= time < client.end_time) or client.end_time == 0:
        #         idle.append(idx)
        # return idle


class Node:
    def __init__(self, down_bw, up_bw, computer_ability, version, args):
        self.down_bw = down_bw
        self.up_bw = up_bw
        self.computer_ability = computer_ability
        self.version = version
        self.data_size = args.local_bs
        self.start_time = 0
        self.end_time = 0
        self.args = args
        self.selected = 0
        self.avg = 0

    def get_end_time(self, start_time, version):
        self.version = version
        self.start_time = start_time

        down_time = 10 / (self.down_bw / 8)
        train_time = self.data_size * self.args.local_ep / self.computer_ability
        up_time = 10 / (self.up_bw / 8)
        time = down_time + train_time + up_time

        self.end_time = start_time + time
        self.avg = time if self.selected == 0 else (self.avg * self.selected + time) / (self.selected + 1)
        self.selected += 1

        return self.end_time
