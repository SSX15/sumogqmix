import csv
import pdb
from typing import Union

import torch
import traci
import sumolib
import os
import sys
import pandas as pd
import numpy as np
from utils.tl import trafficlight
from gymnasium import spaces
global att_list
att_list = []
LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import gymnasium
class MALenv(gymnasium.Env):
    CONNECTION_LABEL = 0
    def __init__(
            self, args,
            begin_time=0,
            max_depart_delay=100000,
            waiting_time_memory=1000,
            time_to_teleport=-1,
            sumo_seed: Union[str, int] = 'random',
            sumo_warnings: bool = True,
    ):
        self.args = args
        self.episode_file = args.csv_name
        self.sim_file = args.csv_name
        self.net = args.net
        self.route = args.route
        self.use_gui = args.gui
        if self.use_gui or self.render_mode is not  None:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')
        self.min_green = args.min_green
        self.max_green = args.max_green
        self.yellow_time = args.yellow_time
        self.begin_time = begin_time
        self.sim_max_time = args.num_seconds
        self.delta_time = args.delta_time
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.sumo_warnings = False
        self.sumo_seed = args.seed
        self.sumo = None
        self.label = str(MALenv.CONNECTION_LABEL)
        MALenv.CONNECTION_LABEL += 1
        self.ac_speed = 0
        self.ac_queue = 0
        self.vehicles = {}
        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self.net])
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self.net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        self.tl_ids = list(conn.trafficlight.getIDList())
        self.agent_id = self.tl_ids
        conn.close()

        self.run = 0
        self.metrics = []
        self.action_save = []
        self.states = {agent: None for agent in self.agent_id}
        self.gl_s = None
        self.rewards = {agent: None for agent in self.agent_id}
        self.train_freq = args.train_freq

        self.nullphase_hot = [0,0,0,0,0,0]
        self.curphase_hot = {a_id: [1,0,0,0,0,0] for a_id in self.agent_id}
        self.curphasenum = 0
        self.node_map = {}
        self.node_map['0'] = ['0', '1', '3']
        self.node_map['1'] = ['1', '2', '0']
        self.node_map['2'] = ['2', '3', '1']
        self.node_map['3'] = ['3', '0', '2']
        self.mapping = {}
        self.mapping['0'] = ['0j1', '0j2']
        self.mapping['1'] = ['1j4', '1j4']
        self.mapping['2'] = ['2j3', '2j4']
        self.mapping['3'] = ['3j2', '3j3']
        self.cur_phase = {a_id: 0 for a_id in self.agent_id}

    def start(self):
        xml_file = self.args.param_file + "/xml/" + f"{self.run}.xml"
        sumo_cmd = [self.sumo_binary,
                    '-n', self.net,
                    '-r', self.route,
                    '--max-depart-delay', str(self.max_depart_delay),
                    '--time-to-teleport', '-1',
                    '--waiting-time-memory', str(self.waiting_time_memory),
                    "--duration-log.statistics",
                    "--statistic-output", xml_file]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

    @property
    def sim_step(self):
        return traci.simulation.getTime()

    def step_dqn(self, actions):
        #pdb.set_trace()
        for a_id, action in actions.items():
            if self.trafficlights[a_id].time_to_act:
                #self.apply_action(a_id=a_id, action=action) #custom_apply_aciton
                self.trafficlights[a_id].set_next_phase(action)
        time_to_act = False
        while not time_to_act:
            self.sumo.simulationStep()
            for a_id in self.agent_id:
                self.trafficlights[a_id].update()
                if self.trafficlights[a_id].time_to_act:
                    time_to_act = True

        st = self.compute_st()
        r = self.compute_r()
        gl_s = self.compute_gl_s()
        dones = self.compute_done()
        self.compute_info_dqn(r=r)
        return st, r, gl_s, dones

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.run += 1
        self.metrics = []
        self.action_save = []
        self.ac_queue = 0
        self.ac_speed = 0
        self.ac_reward = {a_id: 0 for a_id in self.agent_id}
        self.vehicles = {}
        if seed is not None:
            self.sumo_seed = seed
        self.start()
        self.trafficlights = {a_id: trafficlight(env=self, args=self.args,
                                               tl_id=a_id,
                                               delta_time=self.delta_time,
                                               yellow_time=self.yellow_time,
                                               min_green=self.min_green,
                                               max_green=self.max_green,
                                               begin_time=self.begin_time,
                                               sumo=self.sumo,
                                               agent_n=len(self.agent_id)) for a_id in self.agent_id}
        self.compute_info_dqn()
        return self.compute_st(), self.compute_gl_s()

    def compute_st(self): #single ob
        self.states.update({a_id: self.trafficlights[a_id].get_state() for a_id in self.agent_id})
        return {a_id: self.states[a_id] for a_id in self.agent_id if self.trafficlights[a_id].time_to_act}

    def compute_gl_s(self):
        gl_s = []
        for a_id in self.agent_id:
            queue = self.trafficlights[a_id].get_lane_queue()
            speed = self.trafficlights[a_id].get_lane_speed()
            gl_s.append(sum(queue) / len(queue))
            gl_s.append(sum(speed) / len(speed))
        gl_s = np.array(gl_s, dtype=np.float32)
        return gl_s

    def compute_r(self):
        self.rewards.update({a_id: self.trafficlights[a_id].get_reward() for a_id in self.agent_id})
        return {a_id: self.rewards[a_id] for a_id in self.agent_id if self.trafficlights[a_id].time_to_act}

    def compute_done(self):
        dones = {a_id: False for a_id in self.agent_id}
        dones['__all__'] = self.sim_step > self.sim_max_time
        return dones


    def compute_info_dqn(self, r={}):
        info = {'step': self.sim_step}
        info.update(self.compute_sim_info(r))
        self.compute_agent_info(info)
        self.metrics.append(info)

    def compute_sim_info(self, r={}):
        vehicle = traci.vehicle.getIDList()
        speed = [traci.vehicle.getSpeed(v_id) for v_id in vehicle]
        mean_speed = np.mean(speed)
        sum_queue = sum([int(s < 0.05) for s in speed])
        info1 = {
            'mean_speed': mean_speed,
            'sum_queue': sum_queue,
        }
        reward = {}
        if r == {}:
            for a_id in self.agent_id:
                reward.update({a_id: 0})
        else:
            for a_id in self.agent_id:
                if r[a_id] == None:
                    reward.update({a_id: 0})
                else:
                    reward.update({a_id: r[a_id]})
        info1.update(reward)
        return info1

    def compute_agent_info(self, info):
        for a_id in self.agent_id:
            #self.ac_reward[tl] += info[tl]
            self.ac_reward.update({a_id: self.ac_reward[a_id] + info[a_id]})
        if np.isnan(info['mean_speed']):
            self.ac_speed += 0
        else:
            self.ac_speed += info['mean_speed']
        self.ac_queue += info['sum_queue']

    def save_sim_info(self):
        #simfile = self.sim_file+'{}'.format(self.run)
        simfile = self.sim_file
        if simfile is not None:
            df = pd.DataFrame(self.metrics)
            df_action = pd.DataFrame(self.action_save)
            os.makedirs(os.path.dirname(simfile), exist_ok=True)
            #file_path = os.path.join(self.output_file, '.csv')
            with open(simfile+'.csv', 'a') as f:
                df.to_csv(f, index=False, header=f.tell()==0)
            with open(simfile+'_action.csv', 'a') as fa:
                df_action.to_csv(fa, index=False, header=fa.tell()==0)

    def save_episode_info(self, agents):
        epfile = self.episode_file+'_ep.csv'
        fileexist = os.path.isfile(epfile)

        with open(epfile, 'a') as f:
            cur_ep = {'ac_queue': self.ac_queue,
                      'ac_speed': self.ac_speed}
            for a_id in self.agent_id:
                cur_ep.update({a_id:self.ac_reward[a_id]})
            if agents.loss == []:
                cur_ep.update({'loss': 0})
            else:
                cur_ep.update({'loss': np.mean(agents.loss)})
            agents.loss = []
            f_writer = csv.DictWriter(f, fieldnames=cur_ep.keys())
            if not fileexist:
                f_writer.writeheader()
            #print(cur_ep)
            f_writer.writerow(cur_ep)

    def ob_space(self):
        return self.trafficlights[self.agent_id[0]].ob_space
    def action_space(self):
        return self.trafficlights[self.agent_id[0]].action_space

    def close(self):
        #pdb.set_trace()
        self.sumo.close()

    def rollout(self, agents, ep, max_ep):
        actions = {}
        nums_collect = 0
        done = {'__all__': False}
        while nums_collect < self.train_freq and not done['__all__']:
            if ep == max_ep:
                self.write_eval()
            if self.args.gat:
                if ep != max_ep:
                    agents.get_gat_state()
                else:
                    agents.get_gat_state(eval=True)
            for id in self.agent_id:
                action = agents.choose_action(agent_id=id)
                action = action.item()
                actions.update({id: action})
            self.action_save.append(actions)
            next_st, r, gl_ns, done = self.step_dqn(actions=actions)
            if done['__all__']:
                if ep == max_ep:
                    self.output_att()
                return done
            nums_collect += 1
            agents.timestep_plus()
            agents.buffer_push(action=actions, next_st=next_st, r=r, gl_ns=gl_ns)
            agents.update_state(next_st=next_st, gl_ns=gl_ns)
            agents.update_target()
        return done

    def evaluate(self, agents):
        actions = {}
        done = {'__all__': False}
        while not done['__all__']:
            self.write_eval()
            if self.args.gat:
                agents.get_gat_state_eval()
            for id in self.agent_id:
                action = agents.choose_action(agent_id=id)
                action = action.item()
                actions.update({id: action})
            self.action_save.append(actions)
            next_st, r, gl_ns, done = self.step_dqn(actions=actions)
            agents.update_state(next_st=next_st, gl_ns=gl_ns)
        self.output_att()

    def write_eval(self):
        eva_lights = ["intersection_2_3", "intersection_2_2", "intersection_2_4", "intersection_1_3",
                      "intersection_3_3"]
        eva_speed = {}
        eva_queue = {}
        eva_carnums = {}
        eva_occup = {}
        data_eva = [eva_speed, eva_queue, eva_carnums, eva_occup]
        f_eva_speed = self.episode_file + 'eva_speed.csv'
        f_eva_queue = self.episode_file + 'eva_queue.csv'
        f_eva_carnums = self.episode_file + 'eva_carnums.csv'
        f_eva_occup = self.episode_file + 'eva_occup.csv'
        f_eva = [f_eva_speed, f_eva_queue, f_eva_carnums, f_eva_occup]

        for tl in eva_lights:
            for lane in traci.trafficlight.getControlledLanes(tl):
                eva_speed.update({lane: traci.lane.getLastStepMeanSpeed(lane)})
                eva_queue.update({lane: traci.lane.getLastStepHaltingNumber(lane)})
                eva_carnums.update({lane: traci.lane.getLastStepVehicleNumber(lane)})
                eva_occup.update({lane: traci.lane.getLastStepOccupancy(lane)})
        for index, file in enumerate(f_eva):
            fileexist = os.path.isfile(file)
            with open(file, 'a') as f:
                f_writer = csv.DictWriter(f, fieldnames=data_eva[index].keys())
                if not fileexist:
                    f_writer.writeheader()
                f_writer.writerow(data_eva[index])

    def output_att(self):
        f_att = self.episode_file + '_att.csv'
        att_result= []
        tmp = {}
        for att in att_list:
            for i in range(16):
                for j in range(16):
                    tmp.update({f"{i}_{j}": att[i*16+j]})
            att_result.append(tmp)
        df = pd.DataFrame(att_result)
        with open(f_att, 'a') as f:
            df.to_csv(f, index=False, header=f.tell()==0)

    def compute_adjmatrix(self):
        n = len(self.agent_id)
        adj = torch.zeros((n, n))
        it_sets = []
        for it in self.agent_id:
            links = traci.trafficlight.getControlledLinks(it)
            it_set = set()
            for link in links:
                it_set.add(link[0][0])
                it_set.add(link[0][1])
            it_sets.append(it_set)
        for i, _ in enumerate(self.agent_id):
            for j, _ in enumerate(self.agent_id):
                if i == j:
                    adj[i][j] = 1
                elif len(it_sets[i] & it_sets[j]) != 0:
                    adj[i][j] = 1
        return adj

