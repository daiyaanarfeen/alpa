from flax import linen as nn
import jax.numpy as jnp
import jax
import logging

# from alpa import DeviceCluster, ProfilingResultDatabase, global_config
from alpa.device_mesh import DeviceCluster
# from alpa 
from alpa.util import run_cmd

import ray


def profile_hlo_instructions(operands_info):
    # return
    # for shape in operands_dict:
    #     logging.info(f'{shape.__repr__()} {shape[1][0].dimensions()}')
    
    run_cmd("mkdir -p tmp")
    # if args.efa:
    #     global_config.use_aws_efa = True

    # Initialize a useless jax GPU backend in the driver script.
    # This GPU backend takes 300MB GPU memory to store the CUDA context.
    # This simulates the environment of our benchmark scripts and
    # can make the profiling of available memory more accurate.
    # TODO(lmzheng): Modify jax so it does not allocate this useless CUDA context.
    jax.config.update('jax_platform_name', 'cpu')
    _ = jax.numpy.ones(1)

    # Connect to a ray cluster
    ray.init(address="auto")
    cluster = DeviceCluster()

    prof_database = cluster.profile_operands(cluster_key='default', 
                                             cache_filename='cost_model_cache',
                                             operands_info=operands_info)

    # prof_database = cluster.profile_all(
    #     args.cluster_key,
    #     args.max_comm_size_intra_node,
    #     args.max_comm_size_inter_node,
    #     max_fail_retry=args.max_fail_retry,
    #     cache_filename=args.cache_filename,
    #     dot_range=range(0, 8192, 128))
    # prof_database.save(args.filename)
    # print(f"Save profiling database to {args.filename}")



    exit()




    