#!/usr/bin/env python3

import sys

dataset, action = sys.argv[1:]
assert(dataset in {'mb', 'kitti'})
assert(action in {'test_te', 'train_tr'})

workers = [
    ('localhost', '-gpu 1'),
    ('localhost', '-gpu 2'),
    ('localhost', '-gpu 3'),
    ('localhost', '-gpu 4'),
]

if dataset == 'kitti' and action == 'train_tr':
    params = [
        ('l1', [3, 4, 5]),
        ('fm_s', [1, 2, 3, 4, 5, 6, 7]),
        ('fm_t', [4, 5, 6, 7, 8, 9, 10]),
        ('l2', [3, 4, 5]),
        ('nh2', [200, 300, 400]),
        ('lr', [0.001, 0.003, 0.01]),
        ('true1', [0.5, 1, 1.5]),
        ('false1', [2, 3, 4, 5]),
        ('false2', [4, 6, 8, 10, 12]),
    ]

    def valid(ps):
        if ps['fm_s'] > ps['fm_t']: return False
        if ps['true1'] > ps['false1']: return False
        return True

if dataset == 'mb' and action == 'train_tr':
    params = [
        ('l1', [3, 4, 5]),
        ('fm_s', [1, 2, 3, 4, 5, 6]),
        ('fm_t', [1, 2, 3, 4, 5, 6]),
        ('l2', [3, 4, 5]),
        ('nh2', [100, 150, 200]),
        ('lr', [0.001, 0.003, 0.01]),
        ('true1', [0.5, 1, 1.5]),
        ('false1', [1, 1.5, 2, 2.5, 3]),
        ('false2', [4, 6, 8, 10, 12]),
    ]

    def valid(ps):
        if ps['fm_s'] > ps['fm_t']: return False
        if ps['true1'] > ps['false1']: return False
        return True

if dataset == 'mb' and action == 'test_te':
    params = [
#        ('L1', range(0, 10)),
#        ('cbca_i1', [0, 2, 4, 6, 8]),
#        ('cbca_i2', [0, 2, 4, 6, 8]),
        ('tau1', [2**(i/2.) for i in range(-13,-4)]),
#        ('pi1', [2**i for i in range(-3, 4)]),
#        ('pi2', [2**i for i in range(2, 9)]),
#        ('sgm_q1', [3, 3.5, 4, 4.5, 5]),
#        ('sgm_q2', [2, 2.5, 3, 3.5, 4, 4.5]),
#        ('alpha1', [1 + i/4. for i in range(0, 8)]),
        ('tau_so', [2**(i/2.) for i in range(-10,0)]),
#        ('blur_sigma', [2**(i/2.) for i in range(0, 8)]),
#        ('blur_t', range(1, 8)),
    ]

    def valid(ps):
#        if ps['pi1'] > ps['pi2']: return False
        return True

###
import random
import threading
import multiprocessing
import subprocess
import sys
 
def start_job(ps, level):
    worker = multiprocessing.current_process()._identity[0] - 1
    host, args = workers[worker]
    ps_str = ' '.join('-%s %r' % (name, vals[i]) for name, vals, i in ps)
    if action == 'test_te':
        ps_str += ' -use_cache'
    cmd = "ssh %s 'cd devel/mc-cnn;TERM=xterm ./main.lua %s -a %s %s %s'" % (host, dataset, action, args, ps_str)
    try:
        o = subprocess.check_output(cmd, shell=True)
        return float(o.split()[-1]), ps_str, ps, level
    except:
        print('Exception!')
        return 1, ps_str, ps, level
 
def stop_job(res):
    results.append(res)
    #print(min(results)[:2])
    for r in sorted(results, reverse=True)[-50:]:
        print(r[:2])
    print(res[:2])
    print('--')
    sem.release()

for worker in set(w[0] for w in workers):
    subprocess.call("ssh {} 'pkill luajit'".format(worker), shell=True)

pool = multiprocessing.Pool(len(workers))
sem = threading.Semaphore(len(workers))

results = []
visited = set()
while True:
    # get level
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

    sem.acquire()
    pool.apply_async(start_job, (ps, level + 1), callback=stop_job)
