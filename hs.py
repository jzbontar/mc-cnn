#! /usr/bin/env python3

import os
import random
import subprocess
import sys

method, dataset, arch, action, net_fname = sys.argv[1:]
assert(method in {'random', 'hillclimb_slow', 'hillclimb_fast', 'hillclimb_dim'})
assert(dataset in {'mb', 'kitti', 'kitti2015'})
assert(arch in {'fast', 'slow', 'ad', 'census'})
assert(action in {'test_te', 'train_tr', 'da'})

if action == 'da':
    action = 'train_tr'
    params = [
        ('hflip', [0]),
        ('vflip', [0]),
        ('rotate', [0, 3, 7, 14, 21, 28]),
        ('hscale', [1, 0.9, 0.8, 0.7]),
        ('scale', [1, 0.9, 0.8, 0.7]),
        ('trans', [0]),
        ('hshear', [0, 0.1, 0.2, 0.3]),
        ('brightness', [0, 0.5, 0.7, 1, 1.3]),
        ('contrast', [1, 1.1, 1.2, 1.3, 1.4, 1.5]),
        ('d_vtrans', [0, 0.5, 1, 1.5, 2]),
        ('d_rotate', [0, 3, 5]),
        ('d_hscale', [1, 0.9, 0.8]),
        ('d_hshear', [0, 0.1, 0.2, 0.3]),
        ('d_brightness', [0, 0.2, 0.3, 0.5, 0.7, 0.9]),
        ('d_contrast', [1, 1.1, 1.2])
    ]

    def valid(ps):
        return True

elif (dataset == 'kitti' or dataset == 'kitti2015') and action == 'train_tr' and arch == 'slow':
    params = [
        ('l1', [3, 4, 5]),
        ('fm', [4, 5, 6, 7, 8]),
        ('l2', [3, 4, 5, 6]),
        ('nh2', [200, 300, 400, 500]),
        ('lr', [0.001, 0.003, 0.01]),
#        ('true1', [0.5, 1, 1.5]),
#        ('false1', [2, 3, 4, 5]),
#        ('false2', [4, 6, 8, 10, 12]),
    ]

    def valid(ps):
#        if ps['true1'] > ps['false1']: return False
        return True

elif (dataset == 'kitti' or dataset == 'kitti2015') and action == 'test_te' and arch == 'slow':
    params = [
        ('L1', [0, 1, 2, 3, 4, 5, 6]),
        ('cbca_i1', [0, 2, 4, 6, 8]),
        ('cbca_i2', [0, 2, 4, 6, 8]),
        ('tau1', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('pi1', [0.25, 0.33, 0.44, 0.57, 0.76, 1.0, 1.32, 1.74, 2.3, 3.03, 4.0]),
        ('pi2', [8.0, 10.56, 13.93, 18.38, 24.25, 32.0, 42.22, 55.72, 73.52, 97.01, 128.0]),
        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
        ('alpha1', [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]),
        ('tau_so', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('blur_sigma', [1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0]),
        ('blur_t', [1, 2, 3, 4, 5, 6, 7]),
    ]

    def valid(ps):
        if ps['pi1'] > ps['pi2']: return False
        return True

elif (dataset == 'kitti' or dataset == 'kitti2015') and action == 'test_te' and (arch == 'ad' or arch == 'census'):
    params = [
        ('L1', [0, 1, 2, 3, 4, 5, 6]),
        ('cbca_i1', [0, 2, 4, 6, 8]),
        ('cbca_i2', [0, 2, 4, 6, 8]),
        ('tau1', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('pi1', [0.25, 0.33, 0.44, 0.57, 0.76, 1.0, 1.32, 1.74, 2.3, 3.03, 4.0]),
        ('pi2', [8.0, 10.56, 13.93, 18.38, 24.25, 32.0, 42.22, 55.72, 73.52, 97.01, 128.0]),
        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
        ('alpha1', [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]),
        ('tau_so', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('blur_sigma', [1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0]),
        ('blur_t', [1, 2, 3, 4, 5, 6, 7]),
    ]

    def valid(ps):
        if ps['pi1'] > ps['pi2']: return False
        return True

elif (dataset == 'kitti' or dataset == 'kitti2015') and action == 'test_te' and arch == 'fast':
    params = [
        ('pi1', [0.25, 0.33, 0.44, 0.57, 0.76, 1.0, 1.32, 1.74, 2.3, 3.03, 4.0]),
        ('pi2', [8.0, 10.56, 13.93, 18.38, 24.25, 32.0, 42.22, 55.72, 73.52, 97.01, 128.0]),
        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
        ('alpha1', [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]),
        ('tau_so', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('blur_sigma', [1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0]),
        ('blur_t', [1, 2, 3, 4, 5, 6, 7]),
    ]

    def valid(ps):
        if ps['pi1'] > ps['pi2']: return False
        return True

elif dataset == 'mb' and action == 'train_tr' and arch == 'slow':
    params = [
        ('l1', [3, 4, 5]),
        ('fm', [4, 5, 6, 7, 8]),
        ('l2', [2, 3, 4, 5]),
        ('nh2', [100, 200, 300, 400]),
        ('lr', [0.0003, 0.001, 0.003, 0.01]),
#        ('true1', [0.5, 1, 1.5]),
#        ('false1', [1, 1.5, 2, 2.5]),
#        ('false2', [12, 15, 18, 21, 24, 27]),
    ]

    def valid(ps):
#        if ps['true1'] > ps['false1']: return False
        return True

elif action == 'train_tr' and arch == 'fast':
    params = [
        ('l1', [2, 3, 4, 5, 6]),
        ('fm', [64, 80, 96]),
        ('lr', [0.001, 0.002, 0.005, 0.01, 0.02]),
    ]

    def valid(ps):
        return True

elif dataset == 'mb' and action == 'test_te':
    params = [
#        ('L1', [0, 1, 2, 3, 4, 5, 6]),
#        ('cbca_i1', [0, 2, 4, 6, 8]),
#        ('cbca_i2', [0, 2, 4, 6, 8]),
#        ('tau1', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('pi1', [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.3, 3.0, 4.0]),
        ('pi2', [2.0, 2.6, 3.5, 4.6, 6.1, 8.0, 10.6, 13.9, 18.4, 24.3, 32.0]),
        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
        ('alpha1', [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]),
        ('tau_so', [0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.22, 0.36, 0.6, 1.0]),
        ('blur_sigma', [1.0, 1.29, 1.67, 2.15, 2.78, 3.59, 4.64, 5.99, 7.74, 10.0]),
        ('blur_t', [1, 2, 3, 4, 5]),
    ] 

    def valid(ps):
        if ps['pi1'] > ps['pi2']: return False
        return True

while True:
    if method == 'random':
        x = [random.randint(0, len(vals) - 1) for _, vals in params]
    elif method.startswith('hillclimb'):
        results = []
        for fname in os.listdir():
            if fname.startswith('hs.sh.'):
                for line in open(fname):
                    try:
                        score, dataset_, arch_, action_, ps_str = line.strip().split(' ', 4)
                        if dataset_ == dataset and arch_ == arch and action_ == action:
                            results.append((float(score), ps_str))
                    except ValueError:
                        pass
        score, ps_str = min(results)

        # recover x from ps_str
        ps_str = ps_str.split()
        x = []
        for i in range(len(params)):
            assert(params[i][0] == ps_str[2 * i][1:])
            val = float(ps_str[2 * i + 1])
            val_ind = min([(abs(val - v), j) for j, v in enumerate(params[i][1])])[1]
            x.append(val_ind)

        # random neighbor
        if method == 'hillclimb_dim':
            i = random.randrange(len(params))
            x[i] = random.randrange(len(params[i][1]))
        else:
            if method == 'hillclimb_fast':
                ps = range(len(params))
            elif method == 'hillclimb_slow':
                ps = [random.randint(0, len(params) - 1)]
            for i in ps:
                ns = [x[i]]
                if x[i] - 1 >= 0:
                    ns.append(x[i] - 1)
                if x[i] + 1 < len(params[i][1]):
                    ns.append(x[i] + 1)
                x[i] = random.choice(ns)
            
    # list of (param name, param value) tuples
    ps = [(params[i][0], params[i][1][x[i]]) for i in range(len(params))]

    if not valid(dict(ps)):
        continue

    ps_str = ' '.join('-{} {}'.format(*p) for p in ps)
    if action == 'test_te':
        if arch == 'slow':
            ps_str += ' -use_cache'
        else:
            ps_str += ' -net_fname {}'.format(net_fname)
    o = subprocess.check_output('./main.lua {} {} -a {} {}'.format(dataset, arch, action, ps_str), shell=True)
    new_score = float(o.split()[-1])
    print(new_score, dataset, arch, action, ps_str)
    sys.stdout.flush()
