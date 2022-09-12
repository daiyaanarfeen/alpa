from cmath import inf
from concurrent.futures import process
from re import L
from time import sleep
from alpa.timer import timers
import numpy as np
from alpa.pipeline_parallel.stage_construction import get_submesh_choices, get_all_submesh_autosharding_config_choices
# from alpa.device_mesh import VirtualPhysicalMesh
from typing import Any, List, Union, Sequence, Tuple, Optional
from alpa.pipeline_parallel.stage_construction import dp, dp_impl
from alpa.util import maybe_numba_jit
import json
import time
from functools import cmp_to_key
import threading
import multiprocessing
from multiprocessing import Process
import sys

class LogicalDeviceMesh:

    def __init__(self, physical_mesh, id_mesh, mesh_alpha=None, mesh_beta=None):
        self.physical_mesh = physical_mesh
        self.id_mesh = np.array(id_mesh)
        self.flatten_ids = tuple(int(x) for x in self.id_mesh.flatten())

        # coefficient for alpha-beta communication model
        if mesh_alpha is None:
            mesh_alpha = [1] * len(self.id_mesh.shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self.id_mesh.shape)
        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)

    @property
    def shape(self):
        return self.id_mesh.shape

class VirtualPhysicalMesh:
    def __init__(self,
                 host_ids: Sequence[int],
                 host_info: Sequence[dict],
                 head_ip,
                 num_devices_per_host,
                 parent: "VirtualPhysicalMesh" = None,
                 devices: Sequence[Sequence[int]] = None):
        # host_ids are the indices of hosts in the global DeviceCluster
        self.host_ids = host_ids
        self.host_info = host_info
        self.head_ip = head_ip
        self.num_devices_per_host = num_devices_per_host
        self.parent = parent

        self.launched_physical_mesh = None
        self.launched_physical_mesh_group = None

        if devices is not None:
            if len(devices) != len(host_ids):
                raise RuntimeError(
                    "Please specify the gpu IDs used on each host.")
            if not all(len(ids) == num_devices_per_host for ids in devices):
                raise RuntimeError(
                    "Device IDs specified for each host does not align "
                    "with `num_devices_per_host`.")
        else:
            devices = [list(range(num_devices_per_host)) for _ in host_ids]

        self.devices = devices

    def slice_2d(self, host_indices, device_indices):
        host_ids = [self.host_ids[x] for x in host_indices]
        # host_info = [self.host_info[x] for x in host_indices]

        # Check the validity of device_indices
        for i in range(len(device_indices)):
            for x in device_indices[i]:
                assert x in self.devices[i]

        return VirtualPhysicalMesh(host_ids=host_ids,
                                   host_info=None,
                                   head_ip=self.head_ip,
                                   num_devices_per_host=len(device_indices[0]),
                                   parent=self,
                                   devices=device_indices)

    @property
    def shape(self):
        return (len(self.host_ids), self.num_devices_per_host)

    @property
    def num_devices(self):
        """Return the total number of GPUs on this mesh."""
        return len(self.host_ids) * self.num_devices_per_host

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    def get_logical_mesh(self,
                         mesh_shape: Optional[Sequence[int]] = None,
                         mesh_alpha: Optional[float] = None,
                         mesh_beta: Optional[float] = None):
        """
        Return a logical mesh and parameters of the alpha-beta communication
        cost model. The logical view is used for auto-sharding.
        """
        if mesh_shape is None:
            mesh_shape = (self.num_hosts, self.num_devices_per_host)

        id_mesh = np.arange(self.num_devices).reshape(mesh_shape)
        mesh_alpha = mesh_alpha or (1, 1)
        mesh_beta = mesh_beta or (1, 0.1)
        return LogicalDeviceMesh(None, id_mesh, mesh_alpha, mesh_beta) 

def get_cluster_cost(num_gpu, price):
    return num_gpu[0] * price[0] + num_gpu[1] * price[1]

def get_mix_cluster(num_gpu, ratio):
    res = []
    res.append((0, num_gpu * ratio))
    i = 1
    while i <= num_gpu:
        res.append((i, (num_gpu-i)*ratio))
        i = i * 2 if i <= 8 else i + 8
    return res[::-1]

def get_homo_cluster(num_gpu):
    res = [0]
    i = 1
    while i <= num_gpu:
        res.append(i)
        i = i * 2 if i <=8 else i + 8
    return res

def get_all_mix_cluster(gpu_price, budget, lb=0):
    res = []
    for num_gpu_1 in range(0, int(budget/gpu_price[0]) + 1):
        if num_gpu_1 % 8 != 0 and num_gpu_1 != 0 and num_gpu_1 != 1 and num_gpu_1 != 2 and num_gpu_1 != 4:
            continue
        for num_gpu_2 in range(0, int(budget/gpu_price[1]) + 1):
            if num_gpu_2 % 8 != 0 and num_gpu_2 != 0 and num_gpu_2 != 1 and num_gpu_2 != 2 and num_gpu_2 != 4:
                continue
            if num_gpu_1+num_gpu_2 > 0 and get_cluster_cost([num_gpu_1, num_gpu_2], gpu_price) <= budget and get_cluster_cost([num_gpu_1, num_gpu_2], gpu_price) >= lb:
                res.append((num_gpu_1, num_gpu_2))
    return res

def dp_hete(num_layers, num_devices, num_microbatches, submesh_choices,
       num_autosharding_configs, compute_cost, max_n_succ_stages):
    """Auto stage dynamic programming."""
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    best_f = None
    best_f_argmin = None
    best_f_stage_max = None
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."
    # print(f"num of possible stage costs: {len(all_possible_stage_costs)} max: {np.max(all_possible_stage_costs)}")
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        # print(f'max_stage_cost: {max_stage_cost}')    
        cost, solution, f, f_stage_max, f_argmin = dp_impl_hete(num_layers, num_devices, num_microbatches,
                                 submesh_choices, num_autosharding_configs,
                                 compute_cost, max_n_succ_stages,
                                 max_stage_cost)

        if cost < best_cost:
            best_cost = cost
            best_solution = solution
            best_f = f
            best_f_stage_max = f_stage_max
            best_f_argmin = f_argmin
            # print(f'new cost: {cost}')
            # for s in solution:
            #     print(s[0], submesh_choices[s[1]], s[2], s[3])
        last_max_stage_cost = max_stage_cost

    return best_cost, best_solution, best_f, best_f_stage_max, best_f_argmin


@maybe_numba_jit
def dp_impl_hete(num_layers, num_devices, num_microbatches, submesh_choices,
            num_autosharding_configs, compute_cost, max_n_succ_stages,
            max_stage_cost):
    f = np.full((num_layers + 1, num_layers + 1, num_devices[0] + 1, num_devices[1] + 1),
                np.inf,
                dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, num_devices[0] + 1, num_devices[1] + 1),
                          0.0,
                          dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices[0] + 1, num_devices[1] + 1, 4),
                       -1,
                       dtype=np.int32)
    f[0, num_layers, 0, 0] = 0
    for s in range(1, num_layers + 1):  # pylint: disable=too-many-nested-blocks
        for i in range(num_layers - 1, -1, -1):
            for k in range(num_layers, i, -1):
                for num_gpu_1 in range(0, num_devices[0]+1):
                    for num_gpu_2 in range(0, num_devices[1]+1):
                        for gpu_id in range(len(num_devices)):
                            num_gpu_aval = num_gpu_1 if gpu_id==0 else num_gpu_2
                            if num_gpu_aval == 0:
                                continue
                            for m, submesh in enumerate(submesh_choices):
                                n_submesh_devices = np.prod(np.array(submesh))
                                if n_submesh_devices <= num_gpu_aval:
                                    for n_config in range(num_autosharding_configs):
                                        if s-1 <= max_n_succ_stages[gpu_id,i,k-1,m,n_config]:
                                            stage_cost = compute_cost[gpu_id,i,k-1,m, n_config]
                                            new_cost = (f[s-1,k,num_gpu_1-n_submesh_devices,num_gpu_2] if gpu_id==0 else f[s-1,k,num_gpu_1,num_gpu_2-n_submesh_devices]) + stage_cost
                                            if stage_cost <= max_stage_cost and new_cost < f[s, i,num_gpu_1,num_gpu_2]:
                                                f[s,i,num_gpu_1,num_gpu_2] = new_cost
                                                f_stage_max[s,i,num_gpu_1,num_gpu_2] = max(stage_cost, f_stage_max[s-1,k,num_gpu_1-n_submesh_devices,num_gpu_2] if gpu_id==0 else f_stage_max[s-1,k,num_gpu_1,num_gpu_2-n_submesh_devices])
                                                f_argmin[s,i,num_gpu_1,num_gpu_2] = (gpu_id,k,m,n_config)
    best_s = -1
    best_total_cost = np.inf
    best_num_gpu_1 = -1
    best_num_gpu_2 = -1
    for s in range(1, num_layers + 1):
        for num_gpu_1 in range(0, num_devices[0]+1):
            for num_gpu_2 in range(0, num_devices[1]+1):
                if f[s, 0, num_gpu_1, num_gpu_2] < best_total_cost:
                    best_s = s
                    best_total_cost = f[s, 0, num_gpu_1, num_gpu_2]
                    best_num_gpu_1 = num_gpu_1
                    best_num_gpu_2 = num_gpu_2
        
    if np.isinf(best_total_cost):
        return np.inf, None, f, f_stage_max, f_argmin

    # assert best_num_gpu_1==num_devices[0] and best_num_gpu_2==num_devices[1]
    
    total_cost = f[best_s, 0, best_num_gpu_1, best_num_gpu_2] + (
        num_microbatches - 1) * f_stage_max[best_s, 0, best_num_gpu_1, best_num_gpu_2]

    current_s = best_s
    current_layer = 0
    current_devices = [best_num_gpu_1, best_num_gpu_2]

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices[0] + current_devices[1] > 0:
        used_gpu_id, next_start_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices[0], current_devices[1]])
        assert next_start_layer != -1 and current_devices[0] != -1 and current_devices[1] != -1
        res.append(((current_layer, next_start_layer), submesh_choice,
                    autosharding_choice, used_gpu_id))
        current_s -= 1
        current_layer = next_start_layer
        current_devices[used_gpu_id] -= np.prod(np.array(submesh_choices[submesh_choice]))

    assert (current_s == 0 and current_layer == num_layers and current_devices[0] == 0 and current_devices[1] == 0)

    return total_cost, res, f, f_stage_max, f_argmin


if __name__ == '__main__':
    model_size = "6.5b"
    batch_size = 1536 if model_size=='2b' else 1520
    num_micro_batches = 24 if model_size=='2b' else 38

    num_hosts_profiled = 1
    num_devices_per_host = 8
    submesh_physical_shape_space: str = "power_of_two"
    submesh_logical_shape_space: str = "single_node_model_parallel"
    submesh_choices = get_submesh_choices(num_hosts_profiled, num_devices_per_host, submesh_physical_shape_space)
    # print(f'submesh_choices: {submesh_choices}')

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


    """read profile"""
    # homogenesou
    gpu = "a100"
    # profile = np.load(f'./compute-cost-wresnet-{model_size}-{gpu}.npy')
    # compute_cost = profile[0, :, :, :, :]
    # max_n_succ_stages = profile[1, :, :, :, :]  
    # is_profiled = profile[2, :, :, :, :] 

    # heterogeneous
    gpu_name = ['rtx', 'a100']
    gpu_price = [22.032, 27.197]
    profile_1 = np.load(f'./compute-cost-wresnet-{model_size}-{gpu_name[0]}.npy')
    compute_cost_1 = profile_1[0, :, :, :, :]
    num_layers = compute_cost_1.shape[0]
    max_n_succ_stages_1 = profile_1[1, :, :, :, :]  
    is_profiled_1 = profile_1[2, :, :, :, :]
    profile_2 = np.load(f'./compute-cost-wresnet-{model_size}-{gpu_name[1]}.npy')
    compute_cost_2 = profile_2[0, :, :, :, :]
    max_n_succ_stages_2 = profile_2[1, :, :, :, :]  
    is_profiled_2 = profile_2[2, :, :, :, :]
    # make sure each gpu profile the same set of substages
    # for i in range(is_profiled_1.shape[0]):
    #     for j in range(is_profiled_1.shape[1]):
    #         for k in range(is_profiled_1.shape[2]):
    #             for l in range(is_profiled_1.shape[3]):
    #                 if not is_profiled_1[i,j,k,l]:
    #                     compute_cost_2[i,j,k,l] = np.inf
    #                     max_n_succ_stages_2[i,j,k,l] = -1

    compute_cost_h = np.zeros((2, compute_cost_1.shape[0], compute_cost_1.shape[1], compute_cost_1.shape[2], compute_cost_1.shape[3]))
    compute_cost_h[0, :] = compute_cost_1
    compute_cost_h[1, :] = compute_cost_2
    max_n_succ_stages_h = np.zeros((2, max_n_succ_stages_1.shape[0], max_n_succ_stages_1.shape[1], max_n_succ_stages_1.shape[2], max_n_succ_stages_1.shape[3]))
    max_n_succ_stages_h[0, :] = max_n_succ_stages_1
    max_n_succ_stages_h[1, :] = max_n_succ_stages_2


    """search cost"""
    manager = multiprocessing.Manager()
    res_dict = manager.dict()

    budget = 64 * gpu_price[0]
    mix_cluster = get_all_mix_cluster(gpu_price, budget, lb=48*gpu_price[0])
    mix_cluster = [c for c in mix_cluster if c[0] + c[1] >= 48]
    def compare(c1, c2):
        return get_cluster_cost(c1, gpu_price) - get_cluster_cost(c2, gpu_price)
    mix_cluster = sorted(mix_cluster, key=cmp_to_key(compare))
    print(mix_cluster)


    def solve_cluster(cluster, res_dict):
        lines = []
        print(f"------ Start {cluster[0]} {gpu_name[0]} + {cluster[1]} {gpu_name[1]}, {i+1}/{len(mix_cluster)} ------")
        lines.append(f"------ {cluster[0]} {gpu_name[0]} + {cluster[1]} {gpu_name[1]}, {i+1}/{len(mix_cluster)} ------\n")
        start = time.time()
        iter_time, solution, f, f_stage_max, f_argmin = dp_hete(num_layers, cluster,
                            num_micro_batches, submesh_choices,
                            num_autosharding_configs, compute_cost_h,
                            max_n_succ_stages_h)
        end = time.time()
        lines.append(f'dp search time: {end-start}\n')

        if solution is None:
            lines.append('no solution found\n')
            res_dict[str(cluster)] = (None, None, None)
        else:            
            gpu_used_1 = 0
            gpu_used_2 = 0
            lines.append('best solution:\n')
            for s in solution:
                lines.append(f'({s[0]}, {submesh_choices[s[1]]}, {s[2]}, {gpu_name[0] if s[3]==0 else gpu_name[1]})\t')
                if s[3]==0:
                    gpu_used_1 += submesh_choices[s[1]][0] * submesh_choices[s[1]][1]
                else:
                    gpu_used_2 += submesh_choices[s[1]][0] * submesh_choices[s[1]][1]
            lines.append('\n')
            throughput = 1/iter_time
            cost = get_cluster_cost([gpu_used_1, gpu_used_2], gpu_price)
            res_dict[str(cluster)] = (iter_time, cost, (gpu_used_1, gpu_used_2))
            lines.append(f'used {gpu_used_1} {gpu_name[0]} + {gpu_used_2} {gpu_name[1]}\n')
            lines.append(f'best_iter_time: {iter_time}, throughput: {throughput}, cost: {cost}\n')
            lines.append('\n')

        with open(f'./logs_6.5b/{cluster[0]}_{cluster[1]}.txt', 'a') as f:
            f.writelines(lines)
        print(f"------ END {cluster[0]} {gpu_name[0]} + {cluster[1]} {gpu_name[1]}, {i+1}/{len(mix_cluster)} ------")
        # return cluster, iter_time, (gpu_used_1, gpu_used_2), cost

    processes = []
    mix_cluster = [(96,0)]
    print(mix_cluster)
    for i, cluster in enumerate(mix_cluster):
        p = Process(target=solve_cluster, args=[cluster, res_dict])
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    dict = {}
    for k, v in res_dict.items():
        dict[k] = v
    with open('hete-result-homegeneous-a100.txt', 'w') as file:
        file.write(json.dumps(dict))

    print(res_dict)









    # for i, cluster in enumerate(mix_cluster[:2]):
    #     print(f"------ {cluster[0]} {gpu_name[0]} + {cluster[1]} {gpu_name[1]}, {i+1}/{len(mix_cluster)} ------")

    #     start = time.time()
    #     iter_time, solution, f, f_stage_max, f_argmin = dp_hete(num_layers, cluster,
    #                         num_micro_batches, submesh_choices,
    #                         num_autosharding_configs, compute_cost_h,
    #                         max_n_succ_stages_h)
    #     end = time.time()
    #     print(f'dp search time: {end-start}')

    #     if solution is None:
    #         print("no solution found")
    #     else:            
    #         gpu_used_1 = 0
    #         gpu_used_2 = 0
    #         print('best solution:')
    #         for s in solution:
    #             print('(', s[0], submesh_choices[s[1]], s[2], gpu_name[0] if s[3]==0 else gpu_name[1], ')', end='\t')
    #             if s[3]==0:
    #                 gpu_used_1 += submesh_choices[s[1]][0] * submesh_choices[s[1]][1]
    #             else:
    #                 gpu_used_2 += submesh_choices[s[1]][0] * submesh_choices[s[1]][1]
    #         print('')
    #         throughput = 1/iter_time
    #         cost = get_cluster_cost([gpu_used_1, gpu_used_2], gpu_price)
    #         print(f'used {gpu_used_1} {gpu_name[0]} + {gpu_used_2} {gpu_name[1]}')
    #         print(f'best_iter_time: {iter_time}, throughput: {throughput}, cost: {cost}')
    #         print('\n')





    """fix cost"""
    # num_gpu = 8
    # ratio = 8
    # while num_gpu <= 8:
    #     print(f"###### budget: {num_gpu} ######")
    #     mixed_cluster = get_mix_cluster(num_gpu, ratio)
    #     print(mixed_cluster)
    #     for cluster in mixed_cluster:
    #         if cluster[0] + cluster[1] > num_layers * 8:
    #             continue
            
    #         iter_time, solution, f, f_argmin = dp_hete(num_layers, cluster,
    #                         num_micro_batches, submesh_choices,
    #                         num_autosharding_configs, compute_cost_h,
    #                         max_n_succ_stages_h)

    #         if solution is None:
    #             print("no solution found")
    #         else:            
    #             gpu_used_1 = 0
    #             gpu_used_2 = 0
    #             print('best solution:')
    #             for s in solution:
    #                 print(s[0], submesh_choices[s[1]], autosharding_configs[s[1]][s[2]], 'a100' if s[3]==0 else 'rtx', end='\t')
    #                 if s[3]==0:
    #                     gpu_used_1 += submesh_choices[s[1]][0] * submesh_choices[s[1]][1]
    #                 else:
    #                     gpu_used_2 += submesh_choices[s[1]][0] * submesh_choices[s[1]][1]
    #             print('')
    #             throughput = 1/iter_time
    #             cost = get_cluster_cost(gpu_used_1, gpu_used_2, ratio)
    #             print(f'a100:{gpu_used_1} rtx: {gpu_used_2}')
    #             print(f'best_iter_time: {iter_time}, throughput: {throughput}, cost: {cost}, normalized: {throughput/cost}')
    #             print('')
    #     num_gpu += 8
