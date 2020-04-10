#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:42:06 2020

@author: omarschall
"""

if os.environ['HOME'] == '/Users/omarschall':
    params = {'mu': 0.8, 'clip_norm': 0.1, 'L2_reg': 0.001}
    i_job = 0
    save_dir = '/Users/omarschall/vanilla-rtrl/library'

    #np.random.seed(1)

#with open('notebooks/good_ones/another_try', 'rb') as f:
#    sim = pickle.load(f)
#
#sim.checkpoint_model()
#
#task = Flip_Flop_Task(3, 0.05)
#data = task.gen_data(1000, 8000)
#
#result = analyze_all_checkpoints(sim.checkpoints, find_KE_minima, data,
#                                 verbose_=True, N=20, N_iters=1000)

random_FPs = [np.random.normal(0, 1, (128)) for _ in range(8)]
noisy_points = np.array([random_FPs[np.random.randint(8)] + np.random.normal(0, 0.3, (128)) for _ in range(1000)])
random_FPs = np.array(random_FPs)
U, S, V = np.linalg.svd(noisy_points)
PCs = V[:,:3]
proj = random_FPs.dot(PCs)
noisy_proj = noisy_points.dot(PCs)
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.plot(proj[:,0], proj[:,1], proj[:,2], '.')
ax.plot(noisy_proj[:,0], noisy_proj[:,1], noisy_proj[:,2], '.', alpha=0.2)
for x in proj:
    for y in proj:
        
        ax.plot([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color='C0', alpha=0.1)

task = Flip_Flop_Task(3, 0.05, tau_task=1)
data = task.gen_data(10000, 6000)

n_in = task.n_in
n_hidden = 128
n_out = task.n_out

W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
b_rec = np.zeros(n_hidden)
b_out = np.zeros(n_out)

alpha = 1

rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
          activation=tanh,
          alpha=alpha,
          output=identity,
          loss=mean_squared_error)

optimizer = SGD_Momentum(lr=0.001, mu=params['mu'],
                         clip_norm=params['clip_norm'])
learn_alg = Efficient_BPTT(rnn, 10, L2_reg=params['L2_reg'])
#learn_alg = RFLO(rnn, alpha=alpha)
#learn_alg = Only_Output_Weights(rnn)
#learn_alg = RTRL(rnn, M_decay=0.7)
#learn_alg = RFLO(rnn, alpha=alpha)

comp_algs = []
monitors = ['learn_alg.rec_grads-norm']
monitors = []

sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)
optimizer = SGD_Momentum(lr=0.0001, mu=params['mu'],
                         clip_norm=params['clip_norm'])
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)
optimizer = SGD_Momentum(lr=0.00001, mu=params['mu'],
                         clip_norm=params['clip_norm'])
sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
        comp_algs=comp_algs,
        monitors=monitors,
        verbose=True,
        report_accuracy=False,
        report_loss=True,
        checkpoint_interval=None)
sim.checkpoint_model()

fixed_points = find_KE_minima(sim.checkpoints[-1], data, N=20,
                              verbose=True, parallelize=True)

test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_'],
             verbose=False)
test_loss = np.mean(test_sim.mons['rnn.loss_'])
processed_data = {'test_loss': test_loss}

#plt.figure()
#x = configs_array['T_horizon']
#mean_results = results_array.mean(-1)
#ste_results = results_array.std(-1)/np.sqrt(20)
#for i in range(4):
#    col = 'C{}'.format(i)
#    mu = mean_results[:,i]
#    ste = ste_results[:, i]
#    plt.plot(x, mu, color=col)
#    plt.fill_between(x, mu - ste, mu + ste, alpha=0.3, color=col)
#plt.legend([str(lr) for lr in configs_array['LR']])
#plt.xticks(x)


# test_sim = Simulation(rnn)
# test_sim.run(data,
#              mode='test',
#              monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
#              verbose=False)

# plt.figure()
# plt.plot(test_sim.mons['rnn.y_hat'][:, 0])
# #plt.plot(data['test']['X'][:, 0])
# plt.plot(data['test']['Y'][:, 0])

# Load network
network_name = 'j_boxman'
with open(os.path.join('notebooks/good_ones', network_name), 'rb') as f:
    rnn = pickle.load(f)

task = Flip_Flop_Task(3, 0.05)
np.random.seed(0)
n_test = 10000
data = task.gen_data(0, n_test)
#test_sim = deepcopy(sim)
test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
             verbose=False)

find_slow_points_ = partial(find_slow_points, N_iters=10000, return_period=100,
                            N_seed_2=1)
#results = find_slow_points_([test_sim, 0, 0])
pool = mp.Pool(mp.cpu_count())
N_seed_1 = 8
results = pool.map(find_slow_points_, zip([test_sim]*N_seed_1,
                                          range(N_seed_1),
                                          [i_job]*N_seed_1))
pool.close()
A = [results[i][0] for i in range(N_seed_1)]
speeds = [results[i][1] for i in range(N_seed_1)]
LR_drop_times = [results[i][2] for i in range(N_seed_1)]
result = {'A': A, 'speeds': speeds}

task = Flip_Flop_Task(3, 0.05)
np.random.seed(0)
n_test = 10000
data = task.gen_data(0, n_test)
#test_sim = deepcopy(sim)
test_sim = Simulation(rnn)
test_sim.run(data,
             mode='test',
             monitors=['rnn.loss_', 'rnn.y_hat', 'rnn.a'],
             verbose=False)

ssa = State_Space_Analysis(test_sim.mons['rnn.a'], n_PCs=3)
ssa.plot_in_state_space(test_sim.mons['rnn.a'], '.', alpha=0.002)
ssa.fig.axes[0].set_xlim([-0.6, 0.6])
ssa.fig.axes[0].set_ylim([-0.6, 0.6])
ssa.fig.axes[0].set_zlim([-0.8, 0.8])

data_path = '/Users/omarschall/cluster_results/vanilla-rtrl/slow_points_2'

all_speeds = []

for i_job in range(30):
    try:
        with open(os.path.join(data_path, 'result_{}'.format(i_job)), 'rb') as f:
            result = pickle.load(f)
            A = result['A']
            all_speeds.append(result['speeds'][-1])
    except FileNotFoundError:
        continue
    for i in range(20):
        col = 'C1'
        #ssa.plot_in_state_space(A[i][:-1,:], color=col)
        slowness = np.minimum(1/np.sqrt(result['speeds'][i][-1]), 4)
        ssa.plot_in_state_space(A[i][-1,:].reshape((1,-1)), 'x', color=col, alpha=0.3,
                                markersize=slowness)
    


#task = Flip_Flop_Task(3, 0.5)
#data = task.gen_data(100000, 5000)
#
#n_in = task.n_in
#n_hidden = 32
#n_out = task.n_out
#
#W_in  = np.random.normal(0, np.sqrt(1/(n_in)), (n_hidden, n_in))
#W_rec = np.linalg.qr(np.random.normal(0, 1, (n_hidden, n_hidden)))[0]
#W_out = np.random.normal(0, np.sqrt(1/(n_hidden)), (n_out, n_hidden))
#W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_hidden))
#b_rec = np.zeros(n_hidden)
#b_out = np.zeros(n_out)
#
#alpha = 1
#
#rnn = RNN(W_in, W_rec, W_out, b_rec, b_out,
#          activation=tanh,
#          alpha=alpha,
#          output=identity,
#          loss=mean_squared_error)
#
#optimizer = Stochastic_Gradient_Descent(lr=0.001)
#SG_optimizer = Stochastic_Gradient_Descent(lr=0.001)
#
#if params['algorithm'] == 'Only_Output_Weights':
#    learn_alg = Only_Output_Weights(rnn)
#if params['algorithm'] == 'RTRL':
#    learn_alg = RTRL(rnn, L2_reg=0.01)
#if params['algorithm'] == 'UORO':
#    learn_alg = UORO(rnn)
#if params['algorithm'] == 'KF-RTRL':
#    learn_alg = KF_RTRL(rnn)
#if params['algorithm'] == 'R-KF-RTRL':
#    learn_alg = Reverse_KF_RTRL(rnn)
#if params['algorithm'] == 'BPTT':
#    learn_alg = Future_BPTT(rnn, params['T_horizon'])
#if params['algorithm'] == 'DNI':
#    learn_alg = DNI(rnn, SG_optimizer)
#if params['algorithm'] == 'DNIb':
#    J_lr = 0.001
#    learn_alg = DNI(rnn, SG_optimizer, use_approx_J=True, J_lr=J_lr,
#                    SG_label_activation=tanh, W_FB=W_FB)
#    learn_alg.name = 'DNIb'
#if params['algorithm'] == 'RFLO':
#    learn_alg = RFLO(rnn, alpha=alpha)
#if params['algorithm'] == 'KeRNL':
#    sigma_noise = 0.0000001
#    base_learning_rate = 0.01
#    kernl_lr = base_learning_rate/sigma_noise
#    KeRNL_optimizer = Stochastic_Gradient_Descent(kernl_lr)
#    learn_alg = KeRNL(rnn, KeRNL_optimizer, sigma_noise=sigma_noise)
#
#comp_algs = []
#monitors = []
#
#sim = Simulation(rnn)
#sim.run(data, learn_alg=learn_alg, optimizer=optimizer,
#        comp_algs=comp_algs,
#        monitors=monitors,
#        verbose=True,
#        check_accuracy=False,
#        check_loss=True)