from alpa.timer import timers
import numpy as np
from alpa.pipeline_parallel.stage_construction import dp, dp_impl


def dp_h(num_layers, num_devices, num_microbatches, submesh_choices,
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
        print(max_stage_cost)
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = dp_impl_h(num_layers, num_devices, num_microbatches,
                                 submesh_choices, num_autosharding_configs,
                                 compute_cost, max_n_succ_stages,
                                 max_stage_cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost


    return best_cost, best_solution


def dp_impl_h(num_layers, num_devices, num_microbatches, submesh_choices,
            num_autosharding_configs, compute_cost, max_n_succ_stages,
            max_stage_cost):
    """The core implementation of the DP algorithm."""
    # For f, layer ID start from 0
    # f[#pipeline stages,
    #   layer id that is currently being considered,
    #   number of devices used]
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
                                    for n_config in range(num_autosharding_configs):
                                        if s - 1 <= max_n_succ_stages[gpu_id, i, j - 1, m, n_config]:
                                            stage_cost = compute_cost[gpu_id, i, j - 1, m, n_config]
                                            
                                            if gpu_id == 0:
                                                new_cost = f[s - 1, j, num_gpu_1-n_submesh_devices, num_gpu_2] + stage_cost
                                            else:
                                                new_cost = f[s - 1, j, num_gpu_1, num_gpu_2-n_submesh_devices] + stage_cost

                                            if (stage_cost <= max_stage_cost and
                                                    new_cost < f[s, i, num_gpu_1, num_gpu_2]):
                                                f[s, i, num_gpu_1, num_gpu_2] = new_cost
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


    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, 0, num_devices[0], num_devices[1]] < best_total_cost:
            best_s = s
            best_total_cost = f[s, 0, num_devices[0], num_devices[1]]
        
    if np.isinf(best_total_cost):
        return np.inf, None
    
    total_cost = f[best_s, 0, num_devices, num_devices] + (
        num_microbatches - 1) * f_stage_max[best_s, 0, num_devices, num_devices]

    current_s = best_s
    current_layer = 0
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices[0] > 0 and current_devices[1]:
        used_gpu_id, next_start_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices[0], current_devices[1]])
        assert next_start_layer != -1 and current_devices[0] != -1 and current_devices[1] != -1
        res.append(((current_layer, next_start_layer), submesh_choice,
                    autosharding_choice))
        current_s -= 1
        current_layer = next_start_layer
        current_devices[used_gpu_id] -= np.prod(np.array(submesh_choices[submesh_choice]))

    assert (current_s == 0 and current_layer == num_layers and
            current_devices[0] == 0 and current_devices[1] == 0)

    return total_cost, res

if __name__ == '__main__':
    profile = np.load('./compute-cost-2022-08-23-05-16-16.npy')
    compute_cost = profile[0, :, :, :, :]
    max_n_succ_stages = profile[1, :, :, :, :]  
    print(compute_cost.shape)


    # """homogeneous"""
    # num_layers = 16
    # num_devices = 8
    # num_micro_batches = 24
    # submesh_choices = ((1,1), (1,2))
    # num_autosharding_configs = 3

    # compute_cost = compute_cost[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    # max_n_succ_stages = max_n_succ_stages[:, :, :len(submesh_choices), :len(submesh_choices)+1]

    # _, solution = dp(num_layers, num_devices,
    #                 num_micro_batches, submesh_choices,
    #                 num_autosharding_configs, compute_cost,
    #                 max_n_succ_stages)




    """heterogeneous"""
    num_layers = 16
    num_devices = [1,1]
    num_micro_batches = 24
    submesh_choices = ((1,1))
    num_autosharding_configs = len(submesh_choices) + 1

    compute_cost_h = np.zeros((2, compute_cost.shape[0], compute_cost.shape[1], len(submesh_choices), len(submesh_choices)+1))
    compute_cost_h[0] = compute_cost[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    compute_cost_h[1] = compute_cost[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    max_n_succ_stages_h = np.zeros(compute_cost_h.shape)
    max_n_succ_stages_h[0] = max_n_succ_stages[:, :, :len(submesh_choices), :len(submesh_choices)+1]
    max_n_succ_stages_h[1] = max_n_succ_stages[:, :, :len(submesh_choices), :len(submesh_choices)+1]


    _, solution = dp_h(num_layers, num_devices,
                    num_micro_batches, submesh_choices,
                    num_autosharding_configs, compute_cost_h,
                    max_n_succ_stages_h)





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



