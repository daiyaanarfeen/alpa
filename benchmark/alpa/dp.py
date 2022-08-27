from alpa.timer import timers
import numpy as np
from alpa.pipeline_parallel.stage_construction import get_submesh_choices, get_all_submesh_autosharding_config_choices
# from alpa.device_mesh import VirtualPhysicalMesh
from typing import Any, List, Union, Sequence, Tuple, Optional
# from alpa.pipeline_parallel.stage_construction import dp, dp_impl

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

def dp(num_layers, num_devices, num_microbatches, submesh_choices,
       num_autosharding_configs, compute_cost, max_n_succ_stages):
    """Auto stage dynamic programming."""

    all_possible_stage_costs = np.sort(np.unique(compute_cost[0]))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."
    print(f"num of possible stage costs: {len(all_possible_stage_costs)} max: {np.max(all_possible_stage_costs)}")
    for max_stage_cost in all_possible_stage_costs:

        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = dp_impl(num_layers, num_devices, num_microbatches,
                                 submesh_choices, num_autosharding_configs,
                                 compute_cost, max_n_succ_stages,
                                 max_stage_cost)
        # print(f"max stage cost: {max_stage_cost} cost: {cost} solution: {solution}")
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost

    return best_cost, best_solution


def dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
            num_autosharding_configs, compute_cost, max_n_succ_stages,
            max_stage_cost):
    # f[s, start, d]
    f = np.full((num_layers, num_layers, num_devices),
                np.inf,
                dtype=np.float32)
    f_stage_max = np.full((num_layers, num_layers, num_devices),
                0.0,
                dtype=np.float32)
    f_argmin = np.full((num_layers, num_layers, num_devices, 3),
                       -1,
                       dtype=np.int32)

    for s in range(1, num_layers+1):
        for start in range(num_layers+1-s, 0, -1):
            for d in range(s, num_devices + 1):
                for end in range(start+s-1, start-1, -1):

                    for mesh_id, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        for config_id in range(num_autosharding_configs):

                            if s - 1 <= max_n_succ_stages[start-1, end-1, mesh_id, config_id]:
                                stage_cost = compute_cost[start-1, end-1, mesh_id, config_id]

                                if stage_cost > max_stage_cost:
                                    continue

                                if end + 1 > num_layers:
                                    new_cost = stage_cost
                                else:
                                    new_cost = stage_cost + f[s-1-1, end+1-1, d-n_submesh_devices-1]

                                if new_cost < f[s-1, start-1, d-1]:
                                    f[s-1, start-1, d-1] = new_cost
                                    if end + 1 > num_layers:
                                        f_stage_max[s-1, start-1, d-1] = stage_cost
                                    else:
                                        f_stage_max[s-1, start-1, d-1] = max(stage_cost, f_stage_max[s-1-1, end+1-1, d-n_submesh_devices-1])
                                    f_argmin[s-1, start-1, d-1] = (end, mesh_id, config_id)

    


def dp_hete(num_layers, num_devices, num_microbatches, submesh_choices,
       num_autosharding_configs, compute_cost, max_n_succ_stages):
    """Auto stage dynamic programming."""

    all_possible_stage_costs = np.sort(np.unique(compute_cost[0]))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."
    print(f"num of possible stage costs: {len(all_possible_stage_costs)} max: {np.max(all_possible_stage_costs)}")
    for max_stage_cost in all_possible_stage_costs:

        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = dp_impl_hete(num_layers, num_devices, num_microbatches,
                                 submesh_choices, num_autosharding_configs,
                                 compute_cost, max_n_succ_stages,
                                 max_stage_cost)
        # print(f"max stage cost: {max_stage_cost} cost: {cost} solution: {solution}")
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost

    return best_cost, best_solution


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
        if s > 3:
            break
        for i in range(num_layers - 1, -1, -1):
            for j in range(num_layers, i, -1):

                for num_gpu_1 in range(0, num_devices[0] + 1):
                    for num_gpu_2 in range(0, num_devices[1] + 1):
                        if num_gpu_1 + num_gpu_2 == 0:
                            continue
                        num_gpu = [num_gpu_1, num_gpu_2]
                        for gpu_id in range(len(num_devices)):
                            if num_gpu[gpu_id] == 0:
                                continue
                            for m, submesh in enumerate(submesh_choices):
                                n_submesh_devices = np.prod(np.array(submesh))
                                if n_submesh_devices <= num_gpu[gpu_id]:
                                    # TODO(zhuohan): This level of for loop is not
                                    #   necessary. It can be optimized by sorting
                                    #   the logical mesh shapes.
                                    # if max_stage_cost > 0.9:
                                    #     print(f'\tnum_stage: {s} start: {i} end: {j} gpu1: {num_gpu_1} gpu2: {num_gpu_2} used_gpu: {gpu_id+1} submesh: {submesh}')

                                    for n_config in range(num_autosharding_configs):
                                        if s - 1 <= max_n_succ_stages[gpu_id, i, j - 1, m, n_config]:
                                            stage_cost = compute_cost[gpu_id, i, j - 1, m, n_config]
                                            # if max_stage_cost > 0.9:
                                            #     print(f"\t\tstage cost: {stage_cost}")
                                            if gpu_id == 0:
                                                new_cost = f[s - 1, j, num_gpu_1-n_submesh_devices, num_gpu_2] + stage_cost
                                            else:
                                                new_cost = f[s - 1, j, num_gpu_1, num_gpu_2-n_submesh_devices] + stage_cost

                                            if (stage_cost <= max_stage_cost and
                                                    new_cost < f[s, i, num_gpu_1, num_gpu_2]):
                                                f[s, i, num_gpu_1, num_gpu_2] = new_cost
                                                if max_stage_cost > 0.9:
                                                    print(f"\t\tnew cost: {new_cost}")
                                                if gpu_id == 0:
                                                    f_stage_max[s, i, num_gpu_1, num_gpu_2] = max(
                                                        f_stage_max[s - 1, j,
                                                                    num_gpu_1 - n_submesh_devices, num_gpu_2],
                                                        stage_cost)
                                                else:
                                                    f_stage_max[s, i, num_gpu_1, num_gpu_2] = max(
                                                        f_stage_max[s - 1, j,
                                                                    num_gpu_1, num_gpu_2-n_submesh_devices],
                                                        stage_cost)
                                                    
                                                f_argmin[s, i, num_gpu_1, num_gpu_2] = (gpu_id, j, m, n_config)
                                        # else:
                                        #     print(f"\t\tinsufficient succ stages: {max_n_succ_stages[gpu_id, i, j - 1, m, n_config]}")


    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, 0, num_devices[0], num_devices[1]] < best_total_cost:
            best_s = s
            best_total_cost = f[s, 0, num_devices[0], num_devices[1]]
        
    if np.isinf(best_total_cost):
        return np.inf, None
    
    total_cost = f[best_s, 0, num_devices[0], num_devices[1]] + (
        num_microbatches - 1) * f_stage_max[best_s, 0, num_devices[0], num_devices[1]]

    current_s = best_s
    current_layer = 0
    current_devices = num_devices

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

    assert (current_s == 0 and current_layer == num_layers and
            current_devices[0] == 0 and current_devices[1] == 0)

    return total_cost, res


def solve(compute_cost, max_n_succ_stages):
    num_layers = compute_cost.shape[0]
    num_autosharding_configs = len(submesh_choices) + 1


if __name__ == '__main__':
    model_size = "2b"
    batch_size = 1536 if model_size=='2b' else 1520
    num_micro_batches = 24 if model_size=='2b' else 38

    profile = np.load(f'./compute-cost-wresnet-{model_size}-dgx.npy')
    compute_cost = profile[0, :, :, :, :]
    max_n_succ_stages = profile[1, :, :, :, :]  
    is_profiled = profile[2, :, :, :, :] 
    num_layers = compute_cost.shape[0]
    print(compute_cost.shape)

    num_hosts_profiled = 1
    num_devices_per_host = 8
    submesh_physical_shape_space: str = "power_of_two"
    submesh_logical_shape_space: str = "single_node_model_parallel"

    submesh_choices = get_submesh_choices(num_hosts_profiled, num_devices_per_host, submesh_physical_shape_space)
    print(f'submesh_choices: {submesh_choices}')

    virtual_mesh = VirtualPhysicalMesh(host_ids=list(range(num_hosts_profiled)),
                                       host_info=None,
                                       head_ip=None,
                                       num_devices_per_host=num_devices_per_host,
                                       parent=None)
    autosharding_configs = get_all_submesh_autosharding_config_choices(
        virtual_mesh, submesh_choices,
        submesh_logical_shape_space, batch_size)
    print('autosharding_configs:')
    for i in range(len(autosharding_configs)):
        print([(autosharding_configs[i][j][0].shape, autosharding_configs[i][j][1]) for j in range(len(autosharding_configs[i])) if autosharding_configs[i][j] is not None])
    num_autosharding_configs = len(autosharding_configs[0])



    # """homogeneous"""
    compute_cost = compute_cost[:, :, :len(submesh_choices), :num_autosharding_configs]
    max_n_succ_stages = max_n_succ_stages[:, :, :len(submesh_choices), :num_autosharding_configs]

    for i in range(num_layers):
        print(i, compute_cost[i, i, 3, :], max_n_succ_stages[i, i, 3, :])
    # print(compute_cost[3, 5, 3, :], max_n_succ_stages[3, 5, 3, :])
    # print(compute_cost[6, 7, 3, :], max_n_succ_stages[6, 7, 3, :])
    # print(compute_cost[8, 15, 3, :], max_n_succ_stages[8, 15, 3, :])

    compute_cost = np.full(compute_cost.shape, 1, dtype=np.float32)
    max_n_succ_stages = np.full(max_n_succ_stages.shape, np.inf, dtype=np.float32)

    num_devices = 64
    while num_devices <= 64:
        print(f"###### num_devices: {num_devices} ######")
        cost, solution = dp(num_layers, num_devices,
                        num_micro_batches, submesh_choices,
                        num_autosharding_configs, compute_cost,
                        max_n_succ_stages)
        assert solution is not None, "no solution in auto stage construction."
        print(f'best_cost: {cost}')
        print('best solution:')
        for s in solution:
            print(s[0], submesh_choices[s[1]], s[2])
        print('')
        num_devices *= 2


    exit()


    """heterogeneous"""
    # profile_dgx = np.load('./compute-cost-dgx.npy')
    # compute_cost_dgx = profile_dgx[0, :, :, :, :]
    # max_n_succ_stages_dgx = profile_dgx[1, :, :, :, :]  
    # is_profiled_dgx = profile_dgx[2, :, :, :, :] 

    # num_layers = 16
    # num_devices = [4,4]
    # num_micro_batches = 24
    # submesh_choices = ((1,1), (1,2), (1,4))
    # num_autosharding_configs = len(submesh_choices) + 1



    # compute_cost_h = np.zeros((2, compute_cost.shape[0], compute_cost.shape[1], len(submesh_choices), len(submesh_choices)+1))
    # compute_cost_h[0] = compute_cost[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    # compute_cost_h[1] = compute_cost[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    # max_n_succ_stages_h = np.zeros(compute_cost_h.shape)
    # max_n_succ_stages_h[0] = max_n_succ_stages[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    # max_n_succ_stages_h[1] = max_n_succ_stages[:, :, :len(submesh_choices), :len(submesh_choices)+1]


    # cost, solution = dp_h(num_layers, num_devices,
    #                 num_micro_batches, submesh_choices,
    #                 num_autosharding_configs, compute_cost_h,
    #                 max_n_succ_stages_h)


    print(f'best_cost: {cost}')
    print('best solution:')
    for s in solution:
        print(s[0], submesh_choices[s[1]], s[2])


    assert solution is not None, "no solution in auto stage construction."
    exit()



    # Parse solution
    forward_stage_layer_ids = [
        list(range(start_id, end_id))
        for (start_id, end_id), _, _ in solution
    ]
    submesh_shapes = [
        submesh_choices[submesh_id] for _, submesh_id, _ in solution
    ]
    # selected_autosharding_configs = [
    #     autosharding_configs[submesh_id][autosharding_config_id]
    #     for _, submesh_id, autosharding_config_id in solution
    # ]
    # logical_mesh_shapes = [
    #     mesh.shape for mesh, _ in selected_autosharding_configs
    # ]
    # autosharding_option_dicts = [
    #     option_dict for _, option_dict in selected_autosharding_configs
    # ]

    # Print and store the results
    print("Result forward_stage_layer_ids:", forward_stage_layer_ids)
    print("Result mesh_shapes:", submesh_shapes)
    # print("Result logical_mesh_shapes:", logical_mesh_shapes)
    # print("Result autosharding_option_dicts:", autosharding_option_dicts)



