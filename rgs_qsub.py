#!/usr/bin/env python3

import sys

method, dataset, action, njobs = sys.argv[1:]
assert(method in {'steep', 'explore'})
assert(dataset in {'mb', 'kitti'})
assert(action in {'test_te', 'train_tr'})
njobs = int(njobs)

if dataset == 'kitti' and action == 'train_tr':
    params = [
        ('l1', [3, 4, 5]),
        ('fm_s', [1, 2, 3, 4, 5, 6, 7]),
        ('fm_t', [4, 5, 6, 7, 8, 9, 10]),
        ('l2', [3, 4, 5]),
        ('nh2', [200, 300, 400]),
        ('lr', [0.001, 0.003, 0.01]),
#        ('true1', [0.5, 1, 1.5]),
#        ('false1', [2, 3, 4, 5]),
#        ('false2', [4, 6, 8, 10, 12]),
    ]

    def valid(ps):
        if ps['fm_s'] > ps['fm_t']: return False
#        if ps['true1'] > ps['false1']: return False
        return True

if dataset == 'kitti' and action == 'test_te':
    params = [
        ('L1', range(0, 10)),
        ('cbca_i1', [0, 2, 4, 6, 8]),
        ('cbca_i2', [0, 2, 4, 6, 8]),
        ('tau1', [2**(i/2.) for i in range(-13,-4)]),
        ('pi1', [2**i for i in range(-3, 4)]),
        ('pi2', [2**i for i in range(2, 9)]),
        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
        ('alpha1', [1 + i/4. for i in range(0, 8)]),
        ('tau_so', [2**(i/2.) for i in range(-10,0)]),
        ('blur_sigma', [2**(i/2.) for i in range(0, 8)]),
        ('blur_t', range(1, 8)),
    ]

    def valid(ps):
        if ps['pi1'] > ps['pi2']: return False
        return True

if dataset == 'mb' and action == 'train_tr':
    params = [
        ('l1', [3, 4, 5]),
        ('fm_s', [1, 2, 3, 4, 5, 6]),
        ('fm_t', [1, 2, 3, 4, 5, 6]),
        ('l2', [2, 3, 4, 5]),
        ('nh2', [100, 150, 200]),
        ('lr', [0.001, 0.003, 0.01]),
#        ('true1', [0.5, 1, 1.5]),
#        ('false1', [1, 1.5, 2, 2.5]),
#        ('false2', [12, 15, 18, 21, 24, 27]),
    ]

    def valid(ps):
        if ps['fm_s'] > ps['fm_t']: return False
#        if ps['true1'] > ps['false1']: return False
        return True

if dataset == 'mb' and action == 'test_te':
    params = [
        ('L1', range(0, 10)),
        ('cbca_i1', [0, 2, 4, 6, 8]),
        ('cbca_i2', [0, 2, 4, 6, 8]),
        ('tau1', [2**(i/2.) for i in range(-13,-4)]),
        ('pi1', [2**i for i in range(-3, 4)]),
        ('pi2', [2**i for i in range(2, 9)]),
        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
        ('alpha1', [1 + i/4. for i in range(0, 8)]),
        ('tau_so', [2**(i/2.) for i in range(-10,0)]),
        ('blur_sigma', [2**(i/2.) for i in range(0, 8)]),
        ('blur_t', range(1, 8)),
    ]

    def valid(ps):
        if ps['pi1'] > ps['pi2']: return False
        return True

###
import os
import random
import signal
import subprocess
import sys
import time

def signal_handler(signal, frame):
    for id in jobs:
        print('Stopping job {}'.format(id))
        subprocess.call('qdel {}'.format(id).split())
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

oe = 'oe.{}_{}'.format(dataset, action)
try:
    os.mkdir(oe)
except OSError:
    pass

jobs = {}
results = []
visited = set()
while True:
    # start new jobs
    while len(jobs) < njobs:
        level = random.randint(0, max([r[3] for r in results])) if results else 0
        if level == 0:
            ps = tuple((name, tuple(vals), random.randint(0, len(vals) - 1)) for name, vals in params)
        else:
            ps_min = min([r for r in results if r[3] == level])[2]
            ps = []
            for name, vals, i in ps_min:
                xs = [i]
                if i - 1 >= 0:
                    xs.append(i - 1)
                if i + 1 < len(vals):
                    xs.append(i + 1)
                ps.append((name, vals, random.choice(xs)))
            ps = tuple(ps)

        if not valid({name: vals[i] for name, vals, i in ps}):
            continue

        if ps in visited:
            continue
        visited.add(ps)

        ps_str = ' '.join('-{} {}'.format(name, vals[i]) for name, vals, i in ps)
        args = '{} -a {} {}'.format(dataset, action, ps_str)
        if action == 'test_te':
            args = args + ' -use_cache'
        cmd = 'args="{}" qsub -o {} -V main.sh'.format(args, oe)
        id = int(subprocess.check_output(cmd, shell=True).split()[-1])
        jobs[id] = ps_str, ps, level + 1

    # collect results
    o = subprocess.check_output('qstat -u jz1640'.split()).decode()
    for line in o.splitlines()[5:]:
        line = line.split()
        id, status = int(line[0]), line[9]
        if id in jobs and status == 'C':
            ps_str, ps, level = jobs.pop(id)
            fname = '{}/main.sh.o{}'.format(oe, id)

            score = float(open(fname).readlines()[-1])
            results.append((score, ps_str, ps, level))
            print(score, ps_str)
            sys.stdout.flush()
    time.sleep(10)
