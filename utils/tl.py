import numpy as np
import traci
import pdb
from gymnasium import spaces
class trafficlight:
    def __init__(self,
                env,
                args,
                tl_id,
                delta_time,
                yellow_time,
                min_green,
                max_green,
                begin_time,
                sumo,
                agent_n):
        #pdb.set_trace()
        #print(env)
        self.args = args
        self.agent_n = agent_n
        self.id = tl_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.begin_time = begin_time
        self.green_phase = 0
        self.sumo = sumo
        self.next_action_time = begin_time
        self.time_since_last_change = 0
        self.is_yellow = False
        self.all_lanes = traci.lane.getIDList()
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        #pdb.set_trace()

        #self.lanes_length = {lane: traci.lane.getLength()}
        self.init_phase()
        self.ob_space = spaces.Box(low=np.zeros(self.num_green_phases+1+2*len(self.lanes)+self.agent_n, dtype=np.float32), high=np.ones(self.num_green_phases+1+2*len(self.lanes)++self.agent_n, dtype=np.float32))
        self.action_space = spaces.Discrete(self.num_green_phases)
        self.reward = None
        self.last_measure = 0.0
    def init_phase(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i,j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases

        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    def get_state(self):
        phase = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]
        min_green = [0 if self.time_since_last_change < self.min_green + self.yellow_time else 1]
        density = self.get_lane_density()
        queue = self.get_lane_queue()
        state = np.array(phase + min_green + density + queue, dtype=np.float32)
        return state

    def get_reward(self):
        if self.args.reward == 'queue':
            #self.reward = -sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)
            self.reward = -sum(traci.lane.getLastStepHaltingNumber(lane) for lane in self.all_lanes)
        elif self.args.reward == 'speed':
            self.reward = self.get_average_speed()
        elif self.args.reward == 'pressure':
            self.reward = self.get_pressure()
        elif self.args.reward == 'diff_time':
            self.reward = self.diff_waiting_time_reward()
        return self.reward

    def diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def get_accumulated_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self):
        avg_speed = 0.0
        vehs = self.get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_veh_list(self):
        veh_list = []
        #for lane in self.lanes:
        for lane in self.all_lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    def get_pressure(self):
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)

    def get_lane_density(self):
        density = [traci.lane.getLastStepVehicleNumber(lane_id) * (traci.lane.getLastStepLength(lane_id) + 2.5) / traci.lane.getLength(lane_id) for lane_id in self.lanes]
        return density

    def get_lane_queue(self):
        lane_queue = [traci.lane.getLastStepHaltingNumber(lane_id) * (traci.lane.getLastStepLength(lane_id) + 2.5) / traci.lane.getLength(lane_id) for lane_id in self.lanes]
        return lane_queue

    def set_next_phase(self, newphase):
        if self.green_phase == newphase or self.time_since_last_change < self.min_green + self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[(self.green_phase, newphase)]].state)
            self.green_phase = newphase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_change = 0

    def update(self):
        self.time_since_last_change += 1
        if self.is_yellow and self.time_since_last_change == self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False
