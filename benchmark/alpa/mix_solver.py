from pathlib import Path
import numpy as np
import json
from dp import get_submesh_choices, VirtualPhysicalMesh, get_all_submesh_autosharding_config_choices, dp_hete
import itertools
import os
import os.path
import time
import multiprocessing
import pickle

def slice_cluster(cluster, homogeneous=True):
    single = {cname: [] for cname in cluster}
    for cname, num_gpu in cluster.items():
        i = 0
        while i <= num_gpu:
            single[cname].append(i)
            if i == 0:
                i = i + 1
            elif i <= 8:
                i = i * 2
            else:
                i = i + 8
    res = list(itertools.product(*list(single.values())))[1:]
    if homogeneous:
        res = [e for e in res if e[0] == 0 or e[1] == 0]
    # res = [{list(cluster.keys())[0]: e[0], list(cluster.keys())[1]: e[1]} for e in res]
    return res
    



if __name__ == '__main__':
    gpu_name = ['rtx', 'a100']
    cluster = {gpu_name[0]:8, gpu_name[1]:8}
    job_list = ['wresnet-6.5b', 'gpt-6.7b']
    num_micro_batches_dict = {job_list[0]: 38, job_list[1]: 128}
    batch_size = 1536

    """ spec """
    num_hosts_profiled = 1
    num_devices_per_host = 8
    submesh_physical_shape_space: str = "power_of_two"
    submesh_logical_shape_space: str = "single_node_model_parallel"
    submesh_choices = get_submesh_choices(num_hosts_profiled, num_devices_per_host, submesh_physical_shape_space)

    virtual_mesh = VirtualPhysicalMesh(host_ids=list(range(num_hosts_profiled)),
                                       host_info=None,
                                       head_ip=None,
                                       num_devices_per_host=num_devices_per_host,
                                       parent=None)
    autosharding_configs = get_all_submesh_autosharding_config_choices(
        virtual_mesh, submesh_choices,
        submesh_logical_shape_space, batch_size)
    # print('autosharding_configs:')
    # for i in range(len(autosharding_configs)):
    #     print([(autosharding_configs[i][j][0].shape, autosharding_configs[i][j][1]) for j in range(len(autosharding_configs[i])) if autosharding_configs[i][j] is not None])
    num_autosharding_configs = len(autosharding_configs[0])


    """ prepare propfile"""
    compute_cost_dict = {}
    max_n_succ_stages_dict = {}
    for job in job_list:
        profile_1 = np.load(f'./compute-cost-{job}-{gpu_name[0]}.npy')
        compute_cost_1 = profile_1[0, :, :, :, :]
        max_n_succ_stages_1 = profile_1[1, :, :, :, :]  
        is_profiled_1 = profile_1[2, :, :, :, :]

        profile_2 = np.load(f'./compute-cost-{job}-{gpu_name[1]}.npy')
        compute_cost_2 = profile_2[0, :, :, :, :]
        max_n_succ_stages_2 = profile_2[1, :, :, :, :]  
        is_profiled_2 = profile_2[2, :, :, :, :]

        compute_cost_h = np.zeros((2, compute_cost_1.shape[0], compute_cost_1.shape[1], compute_cost_1.shape[2], compute_cost_1.shape[3]))
        compute_cost_h[0, :] = compute_cost_1
        compute_cost_h[1, :] = compute_cost_2
        max_n_succ_stages_h = np.zeros((2, max_n_succ_stages_1.shape[0], max_n_succ_stages_1.shape[1], max_n_succ_stages_1.shape[2], max_n_succ_stages_1.shape[3]))
        max_n_succ_stages_h[0, :] = max_n_succ_stages_1
        max_n_succ_stages_h[1, :] = max_n_succ_stages_2

        compute_cost_dict[job] = compute_cost_h
        max_n_succ_stages_dict[job] = max_n_succ_stages_h


    """homogeneous allocation"""
    sub_clusters = slice_cluster(cluster, homogeneous=True)
    # print(valid_allocations_homo)

    alloc_list = list(itertools.product(*[job_list, sub_clusters]))
    # print(valid_alloc)

    # get throughput matrix
    throughput_cache_path = f'./throughput_matrix.txt'
    if Path(throughput_cache_path).is_file():
        throughput_dict = json.load(open(throughput_cache_path))
    else:   
        throughput_dict = {}
    for jn in job_list:
        if jn not in throughput_dict:
            throughput_dict[jn] = {}
        for c in sub_clusters:
            key = f'{c[0]}_{gpu_name[0]}_{c[1]}_{gpu_name[1]}'
            if key not in throughput_dict[jn]:
                throughput_dict[jn][key] = -1

    def get_throughput(job, cluster, send_end):
        lines = []
        print(f"------ Start {job}, {cluster} ------")
        lines.append(f"------ Start {job}, {cluster} ------\n")
        start = time.time()
        compute_cost_h = compute_cost_dict[job]
        max_n_succ_stages_h = max_n_succ_stages_dict[job]
        num_layers = compute_cost_h.shape[1]
        num_micro_batches = num_micro_batches_dict[job]
        iter_time, solution, f, f_stage_max, f_argmin = dp_hete(num_layers, cluster,
                    num_micro_batches, submesh_choices,
                    num_autosharding_configs, compute_cost_h,
                    max_n_succ_stages_h)
        end = time.time()
        lines.append(f'dp search time: {end-start}\n')

        
        if solution is None:
            lines.append('no solution found\n')
            res = (job, cluster, np.inf)
        else:
            gpu_used_1 = 0
            gpu_used_2 = 0
            lines.append('best solution:\n')

            for i, s in enumerate(solution):
                lines.append(f'stage: {i}, solution: {s}: layers: {s[0][0]}-{s[0][1]-1}, shard: ({autosharding_configs[s[1]][s[2]][0].shape}, {autosharding_configs[s[1]][s[2]][1]}), gpu: {gpu_name[s[3]]}, time: {compute_cost_h[s[3], s[0][0], s[0][1]-1, s[1], s[2]]}\n')
                if s[3]==0:
                    gpu_used_1 += np.prod(submesh_choices[s[1]])
                else:
                    gpu_used_2 += np.prod(submesh_choices[s[1]])
            throughput = 1/iter_time
            lines.append(f'used {gpu_used_1} {gpu_name[0]} + {gpu_used_2} {gpu_name[1]}\n')
            lines.append(f'best_iter_time: {iter_time}, throughput: {throughput}\n')
            res = (job, cluster, iter_time)

        with open(f'./logs_{job}/{cluster[0]}_{gpu_name[0]}_{cluster[1]}_{gpu_name[1]}.txt', 'a') as f:
            f.writelines(lines)
        print(f"------ Start {job}, {cluster} ------")
        send_end.send(res)


    for job in job_list:
        log_dir = f'./logs_{job}/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    processes = []
    pipe_list = []
    for (job, c) in alloc_list:
        key = f'{c[0]}_{gpu_name[0]}_{c[1]}_{gpu_name[1]}'
        if throughput_dict[job][key] != -1:
            print(f"{job}, {key} cache hit")
            continue
        recv_end, send_end = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=get_throughput, args=(job, c, send_end))
        processes.append(p)
        pipe_list.append(recv_end)
        p.start()

    for i in range(len(processes)):
        (job, c, time) = pipe_list[i].recv()
        key = f'{c[0]}_{gpu_name[0]}_{c[1]}_{gpu_name[1]}'
        throughput_dict[job][key] =time
        processes[i].join()

    print(throughput_dict)

    with open(throughput_cache_path, 'w') as file:
        file.write(json.dumps(throughput_dict))




        






    """heterogeneosu allocation"""
