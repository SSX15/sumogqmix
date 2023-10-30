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

def run(test, lr=None, tf=None, bs=None, bas=None, gs=None):
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-route", dest="route", type=str,
                     default='../nets/hangzhou/hangzhou_4x4_gudang_18041610_1h.rou.xml')
    prs.add_argument("-net", dest="net", default='../nets/hangzhou/hangzhou_4x4_gudang_18041610_1h.net.xml')
    prs.add_argument("-lr", dest="lr", default=0.001)
    prs.add_argument("-gamma", dest="gamma", default=0.99)
    prs.add_argument("-gradient_step", dest="gradient_step", default=1)
    prs.add_argument("-train_freq", dest="train_freq", default=1)
    prs.add_argument("-target_update_freq", dest="target_update_freq", default=720)
    prs.add_argument("-savefile", dest="file", default=True)
    prs.add_argument("-saveparam", dest="save_param", default=False)
    prs.add_argument("-load", dest="load", default=False)
    prs.add_argument("-gui", dest="gui", default=False)
    prs.add_argument("-batch_size", dest="batch_size", default=32)
    prs.add_argument("-delta_time", dest="delta_time", default=5)
    prs.add_argument("-buffer_size", dest="buffer_size", default=36000)
    prs.add_argument("-start_size", dest="start_size", default=300)
    prs.add_argument("-reward", dest="reward", default="queue")  # "queue", "pressure", "diffwait", "speed"
    prs.add_argument("-alg", dest="alg", default="qmix")  # "idqn", "vdn", "qmix"
    prs.add_argument("-rnn", dest="rnn", default=False)
    prs.add_argument("-seq", dest="seq_len", default=8)
    prs.add_argument("-GAT", dest="gat", default=True)

    args = prs.parse_args()
    if test:
        args.lr = lr
        args.train_freq = tf
        args.buffer_size = bs
        args.batch_size = bas
        args.gradient_step = gs
    args.seed = 'random'
    exprimenttime = str(datetime.now()).split('.')[0].replace(':', '-')
    csv_name = '../output/hangzhou/{}_rnn{}_GAT{}_single_st_{}_{}/{}_{}_{}_{}'.format(args.alg, args.rnn, args.gat, args.reward, exprimenttime,
                                                                      args.lr,
                                                                      args.gradient_step,
                                                                      args.train_freq,
                                                                      args.target_update_freq)
    param_file = '../output/hangzhou/{}_rnn{}_GAT{}_single_st_{}_{}/'.format(args.alg, args.rnn, args.gat, args.reward, exprimenttime)
    xml_file = param_file + 'xml/'

    os.makedirs(os.path.dirname(csv_name), exist_ok=True)
    os.makedirs(os.path.dirname(xml_file), exist_ok=True)
    args.param_file = param_file
    args.min_green = 10
    args.max_green = 60
    args.num_seconds = 3600
    args.csv_name = csv_name
    args.epsilon_init = 0
    args.yellow_time = 3
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(param_file + 'args.json', 'w') as file:
        json.dump(vars(args), file)

    if args.seed != 'random':
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        seed_torch(seed)

    # pdb.set_trace()
    env = MALenv(args)
    init_st, init_gl_s = env.reset()
    args.agent_ids = env.agent_id
    args.tl_ob_dim = env.ob_space().shape[0]
    args.ob_dim = args.tl_ob_dim
    args.gl_s_dim = 2 * 16
    args.action_space = env.action_space()
    args.init_state = init_st
    args.init_gl_s = init_gl_s
    if args.gat:
        args.adj = env.compute_adjmatrix()
        args.gat_dim = 64
        args.ob_dim = args.gat_dim
    if args.alg == "idqn":
        from agent.IDQNagent import Agent
    elif args.alg == "vdn":
        from agent.VDNagent import Agent
    elif args.alg == "qmix":
        from agent.QMIXagent import Agent
    agents = Agent(args)

    max_episode = 150
    start_time = time.time()
    for ep in range(max_episode):
        print(f"episode: {ep}/{max_episode}")
        ep_start = time.time()
        agents.update_epsilon(episode=ep, max_episode=max_episode)
        if args.rnn:
            agents.init_rnn()
        done = {'__all__': False}
        while not done['__all__']:
            done = env.rollout(agents=agents)
            agents.train()
        if args.rnn:
            agents.episode_push()
        if args.file:
            env.save_sim_info()
            env.save_episode_info(agents=agents)
        env.close()
        if ep != max_episode - 1:
            new_st, gl_s = env.reset()
            agents.reset_st(state=new_st, gl_s=gl_s)
        print(f"ep{ep} cost {time.time() - ep_start}")
    if args.save_param:
        agents.save_parameters()
    end_time = time.time()
    print('cost time:', end_time - start_time, 's')


if __name__ == '__main__':
    test_arg = False
    lr_list = [0.0005, 0.001, 0.005]
    train_freq_list = [1, 4, 8]
    buffer_size_list = [3600, 10800, 36000]
    batch_size_list = [32, 64]
    gradient_step_list = [1, 5, 10]

    if test_arg:
        for test_lr in lr_list:
            for test_tf in train_freq_list:
                for test_bs in buffer_size_list:
                    for test_bas in batch_size_list:
                        for test_gs in gradient_step_list:
                            run(test_arg, test_lr, test_tf, test_bs, test_bas, test_gs)
    else:
        run(test_arg)

