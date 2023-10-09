import json
import os, sys, argparse
import time
from datetime import datetime
import pdb
import torch
import numpy as np
import random

#os.environ['SUMO_HOME'] = '/home/ssx/sumo'
os.environ['LIBSUMO_AS_TRACI'] = '1'

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from env.env_MAL import MALenv

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    prs.add_argument("-route", dest="route", type=str, default='../nets/hangzhou/hangzhou_4x4_gudang_18041610_1h.rou.xml')
    prs.add_argument("-net", dest="net",default='../nets/hangzhou/hangzhou_4x4_gudang_18041610_1h.net.xml')
    prs.add_argument("-lr", dest="lr", default=0.001)
    prs.add_argument("-gamma", dest="gamma", default=0.99)
    prs.add_argument("-gradient_step", dest="gradient_step", default=1)
    prs.add_argument("-train_freq", dest="train_freq", default=100)
    prs.add_argument("-target_update_freq", dest="target_update_freq", default=3600)
    prs.add_argument("-savefile", dest="file", default=True)
    prs.add_argument("-saveparam", dest="save_param", default=False)
    prs.add_argument("-load", dest="load", default=False)
    prs.add_argument("-gui", dest="gui", default=False)
    prs.add_argument("-batch_size", dest="batch_size", default=32)
    prs.add_argument("-delta_time", dest="delta_time", default=5)
    prs.add_argument("-buffer_size", dest="buffer_size", default=36000)
    prs.add_argument("-start_size", dest="start_size", default=300)
    prs.add_argument("-reward", dest="reward", default="queue") #"queue", "pressure", "diffwait", "speed"
    prs.add_argument("-alg", dest="alg", default="qmix") #"idqn", "vdn", "qmix"

    args = prs.parse_args()
    args.seed = 'random'
    exprimenttime = str(datetime.now()).split('.')[0]
    csv_name = '../output/hangzhou/{}_single_st_{}_{}/{}_{}_{}_{}'.format(args.alg, args.reward, exprimenttime,
                                                                      args.lr,
                                                                      args.gradient_step,
                                                                      args.train_freq,
                                                                      args.target_update_freq)
    param_file = '../output/hangzhou/{}_single_st_{}_{}/'.format(args.alg, args.reward, exprimenttime)
    xml_file = "../output/hangzhou/xml/"
    #args.xml_file = xml_file
    os.makedirs(os.path.dirname(csv_name), exist_ok=True)
    os.makedirs(os.path.dirname(xml_file), exist_ok=True)
    args.param_file = param_file
    args.min_green = 10
    args.max_green = 60
    args.num_seconds = 3600
    args.csv_name = csv_name
    args.epsilon_init = 0
    args.yellow_time = 3
    with open(param_file + 'args.json', 'w') as file:
        json.dump(vars(args), file)

    if args.seed != 'random':
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        seed_torch(seed)

    #pdb.set_trace()
    env = MALenv(args)
    init_st = env.reset()
    args.agent_ids = env.agent_id
    args.ob_space = env.ob_space()
    args.action_space = env.action_space()
    args.init_state = init_st
    if args.alg == "idqn":
        from agent.IDQNagent import Agent
    elif args.alg == "vdn":
        from agent.VDNagent import Agent
    elif args.alg == "qmix":
        from agent.QMIXagent import Agent
    agents = Agent(args)


    max_episode = 200
    start_time = time.time()
    for ep in range(max_episode):
        print(f"episode: {ep}/{max_episode}")
        ep_start = time.time()
        agents.update_epsilon(episode=ep, max_episode=max_episode)
        done = {'__all__': False}
        while not done['__all__']:
            done = env.rollout(agents=agents)
            agents.train()
        if args.file:
            env.save_sim_info()
            env.save_episode_info(agents=agents)
        env.close()
        if ep != max_episode-1:
            new_st = env.reset()
            agents.reset_st(state=new_st)
        print(f"ep{ep} cost {time.time()-ep_start}")
    if args.save_param:
        agents.save_parameters()
    end_time = time.time()
    print('cost time:', end_time-start_time, 's')

