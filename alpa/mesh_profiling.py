"""Profiling communication cost for device meshes."""
from collections import defaultdict
import math
import os
import pickle
import time

import numpy as np
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from optax import scale
from pytz import NonExistentTimeError
import ray

from alpa.util import (GB, print_used_time, XlaPassContext, to_str_round,
                       run_with_timeout)

import logging
logging.basicConfig(format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)

ops = xc.ops


class MeshProfilingResult:
    """Store the profiling result for a physical mesh."""

    def __init__(self):
        # Cost dictionary for communication primitives.
        # Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
        # The elements in the list is sorted according to the size (ascending).
        self.all_gather_cost_dict = defaultdict(list)
        self.all_reduce_cost_dict = defaultdict(list)
        self.all_to_all_cost_dict = defaultdict(list)
        self.reduce_scatter_cost_dict = defaultdict(list)
        self.available_memory_per_device = None

        # Cost dictionary for computation primitives.
        # Reuse the same data structure.
        # Dict[Tuple(None, dtype)] -> List[Tuple(flop_count, time)]
        self.dot_cost_dict = defaultdict(list)
        self.conv_cost_dict = []

        # Cost dictionary for specific operators
        # Dict[op_info] -> double
        self.op_cost_dict = []

    def update(self, new_mesh_result):
        raise NotImplementedError

    def make_monotonic(self):
        """Make the bandwidth monotonically increase along with the
        communication size."""
        for cost_dict in [
                self.all_gather_cost_dict, self.all_reduce_cost_dict,
                self.all_to_all_cost_dict, self.reduce_scatter_cost_dict,
                self.dot_cost_dict
        ]:
            new_cost_dict = {}

            for key, value in cost_dict.items():
                sizes = np.array([x[0] for x in value])
                times = np.array([x[1] for x in value])

                # make bandwidth monotonically increasing
                bandwidth = sizes / times
                for i in range(1, len(bandwidth)):
                    bandwidth[i] = max(bandwidth[i], bandwidth[i - 1])

                new_times = np.empty_like(times)
                for i in range(len(times)):
                    if sizes[i] == 0 or bandwidth[i] == 0:
                        new_times[i] = value[i][1]
                    else:
                        new_times[i] = sizes[i] / bandwidth[i]

                new_value = [
                    (value[i][0], new_times[i]) for i in range(len(value))
                ]
                new_cost_dict[key] = new_value

            cost_dict.update(new_cost_dict)

    def sort_cost_lists(self):
        """Sort the items in the list from smallest to largest. This is the
        format required by the HLO cost model in c++."""
        for cost_dict in [
                self.all_gather_cost_dict, self.all_reduce_cost_dict,
                self.all_to_all_cost_dict, self.reduce_scatter_cost_dict,
                self.dot_cost_dict
        ]:
            new_cost_dict = {}

            for key, value in cost_dict.items():
                sizes = [x[0] for x in value]
                indices = np.argsort(sizes)
                new_cost_dict[key] = [value[i] for i in indices]

            cost_dict.update(new_cost_dict)

    def estimate_all_gather(self, group, size, dtype):
        ret = (
            self._estimate_internal(group, size, dtype,
                                    self.all_gather_cost_dict) -
            self._estimate_internal(group, 0, dtype, self.all_gather_cost_dict))
        return ret

    def estimate_all_reduce(self, group, size, dtype):
        ret = (
            self._estimate_internal(group, size, dtype,
                                    self.all_reduce_cost_dict) -
            self._estimate_internal(group, 0, dtype, self.all_reduce_cost_dict))
        return ret

    @staticmethod
    def _estimate_internal(group, size, dtype, cost_dict):
        key = (group, dtype)
        cost_list = cost_dict[key]
        assert cost_list, f"Cannot find records for {(group, dtype)}"

        if size > cost_list[-1][0]:
            i = len(cost_list) - 2
        elif size < cost_list[0][0]:
            i = 0
        else:
            for i in range(len(cost_list) - 1):
                if cost_list[i][0] <= size <= cost_list[i + 1][0]:
                    break

        left_size = cost_list[i][0]
        left_cost = cost_list[i][1]
        right_size = cost_list[i + 1][0]
        right_cost = cost_list[i + 1][1]

        return (size - left_size) / (right_size - left_size) * (
            right_cost - left_cost) + left_cost

    def __str__(self):
        ret = "=== dot_cost_dict ===\n"
        for key, value in self.dot_cost_dict.items():
            sizes = np.array([x[0] for x in value])
            times = np.array([x[1] for x in value])
            tflops = sizes / times / 1e12
            ret += f"Key: {key}\nTFLOPS: {to_str_round(tflops, 2)}\n\n"

        ret += "=== all_reduce_cost_dict ===\n"
        for key, value in self.all_reduce_cost_dict.items():
            num_devices = len(key[0][0])
            sizes = np.array([x[0] for x in value])
            times = np.array([x[1] for x in value])
            comm_bytes = 2 * (num_devices -
                              1) / num_devices * sizes * to_np_dtype(
                                  key[1]).itemsize
            bandwidth = comm_bytes / times / GB
            ret += f"Key: {key}\nBandwidth: {to_str_round(bandwidth, 2)}\n\n"

        ret += "=== all_to_all_cost_dict ===\n"
        for key, value in self.all_to_all_cost_dict.items():
            num_devices = len(key[0][0])
            sizes = np.array([x[0] for x in value])
            times = np.array([x[1] for x in value])
            comm_bytes = ((num_devices - 1) / (num_devices**2) * sizes *
                          to_np_dtype(key[1]).itemsize)
            bandwidth = comm_bytes / times / GB
            ret += f"Key: {key}\nBandwidth: {to_str_round(bandwidth, 2)}\n\n"
        return ret


class ProfilingResultDatabase:
    """A database that stores profiling results for multiple device mesh
    shapes."""

    def __init__(self, data=None):
        self.data = data or {}

    def query(self, cluster_key, mesh_shape):
        key = (cluster_key, mesh_shape)
        return self.data[key]

    def update_one_mesh(self, cluster_key, mesh_shape, mesh_result):
        key = (cluster_key, mesh_shape)
        if key not in self.data:
            self.data[key] = mesh_result
        else:
            self.data[key].update(mesh_result)

    def update(self, new_database):
        for ((cluster_key, mesh_shape),
             mesh_result) in new_database.data.items():
            self.update_one_mesh(cluster_key, mesh_shape, mesh_result)

    def insert_dummy_mesh_result(self, cluster_key, mesh_shape):
        """Insert dummy results for a mesh."""
        key = (cluster_key, mesh_shape)
        assert key not in self.data

        # Copy data from mesh shape (1, 1)
        src_key = (cluster_key, (1, 1))
        assert src_key in self.data
        self.data[key] = self.data[src_key]

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.data, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            new_data = pickle.load(f)
        self.update(ProfilingResultDatabase(new_data))

    def __str__(self):
        ret = ""
        for (cluster_key, mesh_shape), value in self.data.items():
            ret += f"cluster_key: {cluster_key}, mesh_shape: {mesh_shape}\n"
            ret += str(value)
        return ret


def _op_parameter(builder, num, shape, dtype):
    shape = xc.Shape.array_shape(dtype, shape)
    name = ""
    replicated = []
    return ops.Parameter(builder, num,
                         shape.with_major_to_minor_layout_if_absent(), name,
                         replicated)


def _op_all_gather(operand, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    ret = ops.AllGather(operand, 0, len(replica_groups[0]),
                        replica_groups_protos, channel_id, None, True)
    return ret


def _op_all_reduce(operand, dtype, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    if reduce_op == "add":
        rc = xc.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(rc, 0, (), dtype)
        y = _op_parameter(rc, 1, (), dtype)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.AllReduce(operand, rc, replica_groups_protos, channel_id, None,
                        True)
    return ret


def _CreateBinaryAddComputation(dtype):
    """Computation (dtype, dtype) -> dtype that adds its two parameters."""
    c = _NewComputation("add_param0_by_param1")
    shape = xc.shape_from_pyval(np.array(0, dtype=dtype))
    shape = shape.with_major_to_minor_layout_if_absent()
    ops.Add(ops.Parameter(c, 0, shape), ops.Parameter(c, 1, shape))
    return c.build()

def _NewComputation(name):
      return xc.XlaBuilder(name)


def _op_reduce(operand, dtype, reduce_op):
    c = _NewComputation('a')
    ret = ops.Reduce(
        c,
        operands=[
            # ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype))
            operand
        ],
        init_values=[ops.Constant(c, np.float32(0))],
        computation=_CreateBinaryAddComputation(dtype),
        dimensions_to_reduce=[0])
    print("#######invoked")
    return ret

    # if reduce_op == "add":
    #     builder = xc.XlaBuilder("reduce_" + reduce_op)
    #     x = _op_parameter(builder, 0, (), dtype)
    #     y = _op_parameter(builder, 1, (), dtype)
    #     z = ops.Add(x, y)
    #     rc = builder.build(z)
    # else:
    #     raise NotImplementedError
    
    # # init_values = _op_parameter(builder, 0, (1,), dtype)
    # init_values = ops.ConstantLiteral(builder, np.array(0, dtype))
    # # print(init_values)
    # b = xc.XlaBuilder('out_builder')
    # ret = ops.Reduce(b, [operand], [init_values], rc, [0])
    # return ret


def _op_all_to_all(operand, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    ret = ops.AllToAll(operand, 0, 0, len(replica_groups[0]),
                       replica_groups_protos, channel_id, None, True)
    return ret


def _op_reduce_scatter(operand, dtype, reduce_op, replica_groups, channel_id):
    replica_groups_protos = xc.make_replica_groups(replica_groups)
    if reduce_op == "add":
        rc = xc.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(rc, 0, (), dtype)
        y = _op_parameter(rc, 1, (), dtype)
        z = ops.Add(x, y)
        rc = rc.build(z)
    else:
        raise NotImplementedError

    ret = ops.ReduceScatter(operand, rc, 0, len(replica_groups[0]),
                            replica_groups_protos, channel_id, None, True)
    return ret

def _op_reduce2(operand, dtype, reduce_op):
    if reduce_op == "add":
        builder = xc.XlaBuilder("reduce_" + reduce_op)
        x = _op_parameter(builder, 0, (), dtype)
        y = _op_parameter(builder, 1, (), dtype)
        z = ops.Add(x, y)
        rc = builder.build(z)
    else:
        raise NotImplementedError
    
    # init_values = _op_parameter(builder, 0, (1,), dtype)
    init_values = ops.ConstantLiteral(builder, np.array(0, dtype))
    # print(init_values)

    ret = ops.Reduce(builder, [operand], [init_values], rc, [0])
    return ret

    
def _compile_profiling_executable_while_loop(backend, shapes, op_func,
                                             num_devices):
    """
    Compile an xla executable for benchmarking operators.
    It is a while loop that calls the operator for multiple times.
    """

    in_tuple_shape = xc.Shape.tuple_shape(
        [xc.Shape.array_shape(np.dtype(np.int32), ())] +
        [xc.Shape.array_shape(dtype, shape) for shape, dtype in shapes])

    # print("tuple shapes:", in_tuple_shape.__repr__())
    # exit()

    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])

    # body
    body = xc.XlaBuilder("body")
    in_tuple = ops.Parameter(body, 0, in_tuple_shape)
    counter = ops.GetTupleElement(in_tuple, 0)
    counter = ops.Sub(counter, ops.Constant(body, np.int32(1)))

    operands = [
        ops.GetTupleElement(in_tuple, i + 1) for i in range(len(shapes))
    ]
    body.set_sharding(sharding)
    

    if op_func is None:
        def op_func(operands):
            # c = _NewComputation('a')
            ret = ops.Reduce(
                body,
                operands=[
                    # ops.Constant(c, np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype))
                    operands[0]
                ],
                init_values=[ops.Constant(body, np.float32(0))],
                computation=_CreateBinaryAddComputation(np.float32),
                dimensions_to_reduce=[0])
            operands[1] = ret

    op_func(operands)
    body.clear_sharding()
    ops.Tuple(body, [counter] + operands)
    body_computation = body.build()

    # condition
    cond = xc.XlaBuilder("condition")
    in_tuple = ops.Parameter(cond, 0, in_tuple_shape)
    counter = ops.GetTupleElement(in_tuple, 0)
    ops.Gt(counter, ops.Constant(cond, np.int32(0)))
    cond_computation = cond.Build()

    # while loop
    loop = xc.XlaBuilder("loop")
    counter = _op_parameter(loop, 0, (), np.dtype(np.int32))
    operands = [
        _op_parameter(loop, i + 1, shape, dtype)
        for i, (shape, dtype) in enumerate(shapes)
    ]
    while_init = ops.Tuple(loop, [counter] + operands)
    ops.While(cond_computation, body_computation, while_init)
    for i in range(len(shapes) + 1):
        loop.setup_alias((i,), i, ())
    loop_computation = loop.Build()

    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    shapes = [(1, np.int32)] + shapes
    return shapes, backend.compile(loop_computation, compile_options)


def _compile_profiling_executable_once(backend, shapes, op_func, num_devices):
    """
    Compile an xla executable for benchmarking operators.
    It runs the op only once.
    """

    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])

    body = xc.XlaBuilder("body")
    operands = [
        _op_parameter(body, i, shape, dtype)
        for i, (shape, dtype) in enumerate(shapes)
    ]
    body.set_sharding(sharding)
    op_func(operands)
    body.clear_sharding()
    ops.Tuple(body, operands)
    for i in range(len(shapes)):
        body.setup_alias((i,), i, ())
    body_computation = body.Build()

    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    return shapes, backend.compile(body_computation, compile_options)


def bound(value, minimum, maximum):
    return max(min(value, maximum), minimum)


def to_np_dtype(dtype_str: str):
    """Convert a string type to np dtype"""
    if dtype_str == "f32":
        return np.dtype("float32")
    elif dtype_str == "f16":
        return np.dtype("float16")
    else:
        return np.dtype(dtype_str)


def rank_0_print(host_id, msg):
    """Print message on rank 0."""
    if host_id == 0:
        print(msg, flush=True)


# A set containing all replica group patterns with nccl communicator created.
communicator_set = set()


def profile_operands(device_cluster, cluster_key, cache_filename, operands_info):
    # for shape in operands_info:
    #     logging.info(f'{shape.__repr__()} {shape[1][0].dimensions()}')

    """extract distinct op + operands shape"""
    op_dict = {}
    for op, operands_shape in operands_info:
        # logging.info(f'{op}, {operands_shape}')
        if op not in op_dict:
            op_dict[op] = set()
        op_dict[op].add(tuple(operands_shape))

    for op, shapes in op_dict.items():
        logging.info(f'{op}: {shapes}')

    """Profile the compute cost of operand."""
    physical_mesh = device_cluster.get_physical_mesh(host_ids=[0],
                                                     num_devices_per_host=1)

    num_gpu = 4
    op_infos_dict = dict()

    for op, shapes in op_dict.items():
        if op=='add' or op=='multiply' or op=='divide' or op=='sqrt' or op=='subtract' or op=='power' or op =='transpose':
            if op not in op_infos_dict:
                op_infos_dict[op] = set()
            for operands_shape in shapes:
                first_operand_shape = operands_shape[0]
                dtype_str = str(first_operand_shape.element_type())
                dim = first_operand_shape.dimensions()
                if len(dim) == 0: # scalar
                    shape_ = dim
                    op_infos_dict[op].add((shape_, dtype_str))
                else:
                    batch_dim = dim[0]
                    # for i in range(1, min(num_gpu, batch_dim)+1):
                    for i in range(1, 2):
                        if i != 1 and i % 2 != 0:
                            continue
                        dim = list(first_operand_shape.dimensions())
                        dim[0] = int(dim[0] / i)
                        shape_ = tuple(dim)
                        op_infos_dict[op].add((shape_, dtype_str))
        elif op =='dot':
            if op not in op_infos_dict:
                op_infos_dict[op] = set()
            for operands_shape in shapes:
                assert len(operands_shape) == 3, f'{op} has {len(operands_shape)} operands: {operands_shape}'
                lhs_shape, rhs_shape, out_shape = operands_shape
                dtype_str = str(lhs_shape.element_type())
#                op_infos_dict[op].add(((lhs_shape.dimensions(), rhs_shape.dimensions(), out_shape.dimensions()), dtype_str))
                op_infos_dict[op].add((lhs_shape.dimensions()[0], lhs_shape.dimensions()[-1], rhs_shape.dimensions()[-1], dtype_str))
        elif op=='reshape' :
            if op not in op_infos_dict:
                op_infos_dict[op] = set()
            for operands_shape in shapes:
                assert len(operands_shape) == 2, f'{op} has {len(operands_shape)} operands: {operands_shape}'
                original_shape, target_shape = operands_shape
                dtype_str = str(original_shape.element_type())
                op_infos_dict[op].add(((original_shape.dimensions(), target_shape.dimensions()), dtype_str))
        elif op=='reduce':
            if op not in op_infos_dict:
                op_infos_dict[op] = set()
            op_infos_dict[op].add(((2,1), 'f32'))
       
    op_infos = []
    for op, infos in op_infos_dict.items():
        if op != 'dot':
            continue
        op_infos += [(op, info) for info in infos]

    
    results = physical_mesh.profile_hlo_ops(op_infos, cache_filename)

    logging.info("results:")
    logging.info(f'{results}')

    runtime = 0
    for op, shapes in operands_info:
        if op != 'dot':
            continue
        operands_shape = tuple(shapes)
        logging.info(f"!!!!!! {op} : {shapes}")
        if op not in op_infos_dict:
            logging.info(f'{op} not profiled')
            continue
        
        if op=='add' or op=='multiply' or op=='divide' or op=='sqrt' or op=='subtract' or op=='power' or op =='transpose':
            first_operand_shape = operands_shape[0]
            dtype_str = str(first_operand_shape.element_type())
            dim = first_operand_shape.dimensions()
            if len(dim) == 0: # scalar
                shape_ = dim
                info = (shape_, dtype_str)
            else:
                # for i in range(1, min(num_gpu, batch_dim)+1):
                for i in range(1, 2):
                    if i != 1 and i % 2 != 0:
                        continue
                    dim = list(first_operand_shape.dimensions())
                    dim[0] = int(dim[0] / i)
                    shape_ = tuple(dim)
                    info = (shape_, dtype_str)
            res = results[(op, info)]
            runtime += res
        elif op =='dot':
            assert len(operands_shape) == 3, f'{op} has {len(operands_shape)} operands: {operands_shape}'
            lhs_shape, rhs_shape, out_shape = operands_shape
            dtype_str = str(lhs_shape.element_type())
            # op_infos_dict[op].add(((lhs_shape.dimensions(), rhs_shape.dimensions(), out_shape.dimensions()), dtype_str))
            
#            res = results[(op, ((lhs_shape.dimensions(), rhs_shape.dimensions(), out_shape.dimensions()), dtype_str))]
            res = results[(op, (lhs_shape.dimensions()[0], lhs_shape.dimensions()[-1], rhs_shape.dimensions()[-1], str(lhs_shape.element_type())))]
            runtime += res
        elif op=='reshape':
            assert len(operands_shape) == 2, f'{op} has {len(operands_shape)} operands: {operands_shape}'
            original_shape, target_shape = operands_shape
            dtype_str = str(original_shape.element_type())

            res = results[(op, ((original_shape.dimensions(), target_shape.dimensions()), dtype_str))]
            runtime += res
        elif op=='reduce':
            res = results[(op, ((2,1), 'f32'))]
            runtime += res

    logging.info(f'sum runtime: {runtime}') # runtime: 0.0011827544076368213
        
    
    exit()



          

    """Profile the compute cost of dot."""
    physical_mesh = device_cluster.get_physical_mesh(host_ids=[0],
                                                     num_devices_per_host=1)

    # Profile dot
    op_infos = []
    for dtype in ["f16", "f32"]:
        for n in dot_range:
            op_infos.append(("dot", (n, n, n, dtype)))
    results = physical_mesh.profile_hlo_ops(op_infos, cache_filename)

    exit()

    dot_cost_dict = defaultdict(list)
    for i in range(len(op_infos)):
        n, m, k, dtype = op_infos[i][1]
        flop_count = 2 * n * m * k
        dot_cost_dict[((), dtype)].append((flop_count, results[i]))
        print(f"Matmul: {(n, m, k, dtype)}, "
              f"TFLOPS: {flop_count / results[i]/ 1e12:.2f}")

    physical_mesh.shutdown()
    time.sleep(2)
    return dot_cost_dict


def profile_one_hlo_op(backend, local_devices, host_id, num_devices, op_info):
    """Profile one HLO operator."""
    dot_fp16_work = 100e12
    dot_fp32_work = 50e12
    comm_work = 1 << 32
    replica_groups = None

    if op_info[0] == "dot" and True:
        import ipdb; ipdb.set_trace()
        # CONV TEST
        dtype = to_np_dtype("float32")
        shapes = [((1, 3, 32, 32), dtype), ((128, 3, 3, 3), dtype), ((1, 128, 32, 32), dtype)]
        def op_func(operands):
            lhs, rhs, _ = operands
            strides = [1, 1]
            padding = [(1, 1), (1, 1)]
            lhs_dilation = [1, 1]
            rhs_dilation = [1, 1]
            dim_numbers = xc.make_convolution_dimension_numbers(('NCHW', 'OIHW', 'NCHW'), 2)
            out = ops.ConvGeneralDilated(lhs, rhs, strides, padding, lhs_dilation, rhs_dilation, dim_numbers, 1, 1)
            operands[-1] = out
        work = dot_fp32_work
        number = 4096
    elif op_info[0] == "dot":
        n, m, k, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        shapes = [((n, k), dtype), ((k, m), dtype), ((n, m), dtype)]

        if m != k:
            def op_func(operands):
                lhs, rhs, _ = operands
                dim_numbers = (((1,), (0,)), ((), ()))
                dim_numbers = xc.make_dot_dimension_numbers(dim_numbers)
                out = ops.DotGeneral(lhs, rhs, dim_numbers)
                operands[-1] = out
        else:
            def op_func(operands):
                dim_numbers = (((1,), (0,)), ((), ()))
                dim_numbers = xc.make_dot_dimension_numbers(dim_numbers)
                input = operands[0]
                for i in range(0):
                    out = ops.DotGeneral(input, operands[i+1], dim_numbers)
                    input = out
#                operands[-1] = out
            shapes = [((n, k), dtype)] + [((k, m), dtype) for i in range(0)] + [((n, m), dtype)]

        flop_ct = max(2 * n * m * k, 1)
        if dtype_str == "float16":
            work = dot_fp16_work
        elif dtype_str == "float32":
            work = dot_fp32_work
        else:
            raise ValueError(f"Invalid type: {dtype_str}")
        number = bound(int(work / flop_ct), 10, 1 << 12)
#        (lhs_shape, rhs_shape, out_shape), dtype_str = op_info[1]
#
#        if out_shape[0] == lhs_shape[0]:
#            lhs_contrac_dim = 1
#        else:
#            lhs_contrac_dim = 0
#        if out_shape[1] == rhs_shape[0]:
#            rhs_contrac_dim = 1
#        else:
#            rhs_contrac_dim = 0
#        dim_numbers = (((lhs_contrac_dim,), (rhs_contrac_dim,)), ((), ()))
#
#        dtype = to_np_dtype(dtype_str)
#        shapes = [(lhs_shape, dtype), (rhs_shape, dtype), (out_shape, dtype)]
#        def op_func(operands):
#            lhs, rhs, _ = operands
#            dim_numbers_ = xc.make_dot_dimension_numbers(dim_numbers)
#            out = ops.DotGeneral(lhs, rhs, dim_numbers_)
#            operands[-1] = out
#        number = 1 << 12
    elif op_info[0] == "add":
        shape, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        # print(f"find {op_info[0]}:", shape, str(dtype))
        shapes = [(shape, dtype), (shape, dtype), (shape, dtype)]

        def op_func(operands):
            lhs, rhs, _ = operands
            out = ops.Add(lhs, rhs)
            operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "subtract":
        shape, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        # print(f"find {op_info[0]}:", shape, str(dtype))
        shapes = [(shape, dtype), (shape, dtype), (shape, dtype)]

        def op_func(operands):
            lhs, rhs, _ = operands
            out = ops.Sub(lhs, rhs)
            operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "multiply":
        shape, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        # print(f"find {op_info[0]}:", shape, str(dtype))
        shapes = [(shape, dtype), (shape, dtype), (shape, dtype)]

        def op_func(operands):
            lhs, rhs, _ = operands
            out = ops.Mul(lhs, rhs)
            operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "divide":
        shape, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        # print(f"find {op_info[0]}:", shape, str(dtype))
        shapes = [(shape, dtype), (shape, dtype), (shape, dtype)]

        def op_func(operands):
            lhs, rhs, _ = operands
            out = ops.Div(lhs, rhs)
            operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "power":
        shape, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        # print(f"find {op_info[0]}:", shape, str(dtype))
        shapes = [(shape, dtype), (shape, dtype), (shape, dtype)]

        def op_func(operands):
            lhs, rhs, _ = operands
            out = ops.Pow(lhs, rhs)
            operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "sqrt":
        shape, dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        # print(f"find {op_info[0]}:", shape, str(dtype))
        shapes = [(shape, dtype), (shape, dtype)]

        def op_func(operands):
            lhs, _ = operands
            out = ops.Sqrt(lhs)
            operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "reshape":
        (original_shape, target_shape), dtype_str = op_info[1]
        dtype = to_np_dtype(dtype_str)
        shapes = [(original_shape, dtype), (target_shape, dtype)]
        def op_func(operands):
                operand, _= operands
                out = ops.Reshape(operand, list(target_shape))
                operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "transpose":
        original_shape, dtype_str = op_info[1]
        assert len(original_shape)==2, f'transpose has {len(original_shape)} dimensions, {original_shape}'
        dtype = to_np_dtype(dtype_str)
        shapes = [(original_shape, dtype), (original_shape[::-1], dtype)]
        def op_func(operands):
                operand, _= operands
                out = ops.Transpose(operand, (1,0))
                operands[-1] = out
        number = 1 << 12
    elif op_info[0] == "reduce":
        dtype = np.dtype(np.float32)
        shapes = [((2,), dtype), ((), dtype)]
        def op_func(operands):
            out = _op_reduce(operands[0], dtype, 'add')
            operands[-1] = out
        number = 1 << 5
    elif op_info[0] == "all-gather":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        size = size // len(replica_groups[0]) * len(replica_groups[0])
        shapes = [((size // len(replica_groups[0]),), dtype), ((size,), dtype)]

        def op_func(operands):
            if shapes[0][0][0] == 0:
                return
            channel_id = backend.create_channel_handle()
            out = _op_all_gather(operands[0], replica_groups, channel_id)
            operands[-1] = out

        number = bound(int(comm_work / max(size * dtype.itemsize, 1)), 10,
                       1 << 13)
    elif op_info[0] == "all-reduce":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        shapes = [((size,), dtype), ((size,), dtype)]

        def op_func(operands):
            channel_id = backend.create_channel_handle()
            out = _op_all_reduce(operands[0], dtype, "add", replica_groups,
                                 channel_id)
            operands[-1] = out

        number = bound(int(comm_work / max(size * dtype.itemsize, 1)), 10,
                       1 << 13)
    elif op_info[0] == "all-to-all":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        size = size // (len(replica_groups[0])**2) * (len(replica_groups[0])**2)
        shapes = [((size // len(replica_groups[0]),), dtype),
                  ((size // len(replica_groups[0]),), dtype)]

        def op_func(operands):
            if shapes[0][0][0] // len(replica_groups[0]) == 0:
                return
            channel_id = backend.create_channel_handle()
            out = _op_all_to_all(operands[0], replica_groups, channel_id)
            operands[-1] = out

        number = bound(int(comm_work / max(size * dtype.itemsize, 1)), 10,
                       1 << 13)
    elif op_info[0] == "reduce-scatter":
        replica_groups, dtype, size = op_info[1]
        dtype = to_np_dtype(dtype)
        size = size // len(replica_groups[0]) * len(replica_groups[0])
        shapes = [((size,), dtype), ((size // len(replica_groups[0]),), dtype)]

        def op_func(operands):
            if shapes[1][0][0] == 0:
                return
            channel_id = backend.create_channel_handle()
            out = _op_reduce_scatter(operands[0], dtype, "add", replica_groups,
                                     channel_id)
            operands[-1] = out

        number = bound(int(comm_work / max(size * dtype.itemsize, 1)), 10,
                       1 << 13)
    elif op_info[0] == "create-communicator":
        replica_groups, = op_info[1]
        dtype = to_np_dtype("f32")
        shapes = [((1024,), dtype), ((1024,), dtype)]

        def op_func(operands):
            channel_id = backend.create_channel_handle()
            out = _op_all_reduce(operands[0], dtype, "add", replica_groups,
                                 channel_id)
            operands[-1] = out
    elif op_info[0] == "barrier":
        replica_groups = (tuple(i for i in range(num_devices)),)
        dtype = to_np_dtype("f32")
        shapes = [((1,), dtype), ((1,), dtype)]

        def op_func(operands):
            channel_id = backend.create_channel_handle()
            out = _op_all_reduce(operands[0], dtype, "add", replica_groups,
                                 channel_id)
            operands[-1] = out
    else:
        raise NotImplementedError(f"Invalid op: {op_info[0]}")

    if op_info[0] in ["create-communicator", "barrier"]:
        rank_0_print(host_id, f"{op_info[0]}")

        # Compile
        all_shapes, compiled = _compile_profiling_executable_once(
            backend, shapes, op_func, num_devices)

        # Run
        device_inputs = []
        for shape, dtype in all_shapes:
            device_inputs.append([
                backend.buffer_from_pyval(np.ones(shape, dtype),
                                          local_devices[k])
                for k in range(len(local_devices))
            ])

        for d in local_devices:
            d.synchronize_all_activity()
        device_inputs = compiled.execute_sharded_on_local_devices(device_inputs)
        for d in local_devices:
            d.synchronize_all_activity()
        return 0
    else:
        # Create the nccl communicator
        # This step is a workaround for some nccl/xla deadlock
        if replica_groups and replica_groups not in communicator_set:
            tmp_op_info = ("create-communicator", (op_info[1][0],))
            profile_one_hlo_op(backend, local_devices, host_id, num_devices,
                               tmp_op_info)
            communicator_set.add(replica_groups)

        warmup = max(number // 10, 2)

        rank_0_print(
            host_id, f"Profiling {op_info}, number: {number}, "
            f"timestamp: {time.time():.0f}.")

        # Compile
        if op_info[0] == 'reduce': 
            all_shapes, compiled = _compile_profiling_executable_while_loop(
            backend, shapes, None, num_devices)
        else:
            all_shapes, compiled = _compile_profiling_executable_while_loop(
            backend, shapes, op_func, num_devices)

        # Warm up
        device_inputs = []
        for j, (shape, dtype) in enumerate(all_shapes):
            if j == 0:
                device_inputs.append([
                    backend.buffer_from_pyval(np.int32(warmup),
                                              local_devices[k])
                    for k in range(len(local_devices))
                ])
            else:
                np_array = np.ones(shape, dtype)
                device_inputs.append([
                    backend.buffer_from_pyval(np_array, local_devices[k])
                    for k in range(len(local_devices))
                ])

        for d in local_devices:
            d.synchronize_all_activity()
        device_inputs = compiled.execute_sharded_on_local_devices(device_inputs)
        for d in local_devices:
            d.synchronize_all_activity()

        # Run profiling
        device_inputs[0] = [
            backend.buffer_from_pyval(np.int32(number), local_devices[k])
            for k in range(len(local_devices))
        ]

        for d in local_devices:
            d.synchronize_all_activity()
        tic = time.time()
        compiled.execute_sharded_on_local_devices(device_inputs)
        for d in local_devices:
            d.synchronize_all_activity()
        toc = time.time()

        # Return
        mean_time = (toc - tic) / number
        rank_0_print(
            host_id, f"mean time: {mean_time}")
        return mean_time


def profile_hlo_ops(op_infos, backend, local_devices, host_id, num_devices,
                    cache_filename, single_timeout):
    """Profile a list of HLO operators on a worker."""
    results = []
    save_every = 15
    barrier_every = 5

    print(f'num of hosts: {host_id}, num_devices: {num_devices}')

    if os.path.exists(cache_filename):
        # rank_0_print(host_id,
        #              f"Load cached hlo op cost dict from {cache_filename}...")
        # with open(cache_filename, "rb") as cf:
        #     cache_dict = pickle.load(cf)
            cache_dict = {}
    else:
        cache_dict = {}

    

    old_cache_len = len(cache_dict)

    try:
        for i, op_info in enumerate(op_infos):
            if op_info in cache_dict:
                rank_0_print(host_id, f"Hit cache {op_info} ...")
                results.append(cache_dict[op_info])
                continue

            if i % barrier_every == 0:
                # Run barrier to reduce hanging/deadlock issues
                run_with_timeout(profile_one_hlo_op,
                                 (backend, local_devices, host_id, num_devices,
                                  ("barrier",)),
                                 timeout=single_timeout)

            # Profile one op
            mean_time = run_with_timeout(
                profile_one_hlo_op,
                (backend, local_devices, host_id, num_devices, op_info),
                timeout=single_timeout)
            cache_dict[op_info] = mean_time
            results.append(mean_time)

            if host_id == 0 and (i + 1) % save_every == 0:
                old_cache_len = len(cache_dict)
                rank_0_print(host_id, "Save cache...")
                with open(cache_filename, "wb") as cf:
                    pickle.dump(cache_dict, cf)
    except TimeoutError:
        print(f"Worker {host_id} timeout error", flush=True)
        return None
    except RuntimeError:
        print(f"Worker {host_id} runtime error", flush=True)
        return None

    if host_id == 0 and len(cache_dict) > old_cache_len:
        rank_0_print(host_id, "Save cache...")
        with open(cache_filename, "wb") as cf:
            pickle.dump(cache_dict, cf)


    return cache_dict
    return np.array(results)


def profile_dot(dot_range, device_cluster, cache_filename):
    """Profile the compute cost of dot."""
    physical_mesh = device_cluster.get_physical_mesh(host_ids=[0],
                                                     num_devices_per_host=1)

    # Profile dot
    op_infos = []
    for dtype in ["f16", "f32"]:
        for n in dot_range:
            op_infos.append(("dot", (n, n, n, dtype)))
    results = physical_mesh.profile_hlo_ops(op_infos, cache_filename)

    exit()

    dot_cost_dict = defaultdict(list)
    for i in range(len(op_infos)):
        n, m, k, dtype = op_infos[i][1]
        flop_count = 2 * n * m * k
        dot_cost_dict[((), dtype)].append((flop_count, results[i]))
        print(f"Matmul: {(n, m, k, dtype)}, "
              f"TFLOPS: {flop_count / results[i]/ 1e12:.2f}")

    physical_mesh.shutdown()
    time.sleep(2)
    return dot_cost_dict


def enumerate_all_collective_spec(num_hosts, num_devices_per_host,
                                  max_comm_size_intra_node,
                                  max_comm_size_inter_node):
    """Enumerate all possible collective groups."""
    # Enumerate all possible logical meshes
    logical_mesh_shapes = []
    num_devices = num_hosts * num_devices_per_host
    for i in range(1, num_devices + 1):
        if num_devices % i == 0:
            logical_mesh_shapes.append((num_devices // i, i))

    # Enumerate all replica groups
    all_specs = set()
    for logical_mesh_shape in logical_mesh_shapes:
        # dim 0
        replica_groups = []
        tmp_group = []
        for i in range(logical_mesh_shape[0]):
            tmp_group.append(
                tuple(i * logical_mesh_shape[1] + j
                      for j in range(logical_mesh_shape[1])))
        replica_groups.append(tuple(tmp_group))

        # dim 1
        tmp_group = []
        for j in range(logical_mesh_shape[1]):
            tmp_group.append(
                tuple(i * logical_mesh_shape[1] + j
                      for i in range(logical_mesh_shape[0])))
        replica_groups.append(tuple(tmp_group))

        for replica_group in replica_groups:
            for dtype in ["f32", "f16"]:
                # Debug filter
                #if replica_group != (tuple(range(32)),) or dtype != "f32":
                #    continue

                if (max(replica_group[0]) - min(replica_group[0]) <
                        num_devices_per_host):
                    max_comm_size = max_comm_size_intra_node
                else:
                    max_comm_size = max_comm_size_inter_node

                max_num_elem_log_2 = math.ceil(
                    math.log2(
                        (1 << max_comm_size) / to_np_dtype(dtype).itemsize))

                all_specs.add((tuple(replica_group), dtype, 0))
                for i in range(0, max_num_elem_log_2 + 1):
                    all_specs.add((tuple(replica_group), dtype, 1 << i))

    all_specs = list(all_specs)
    all_specs.sort(key=lambda k:
                   (k[0][0][0] - k[0][0][-1], to_np_dtype(k[1]).itemsize, k[2]))
    return list(all_specs)


def profile_all(device_cluster,
                cluster_key,
                max_comm_size_intra_node,
                max_comm_size_inter_node,
                max_fail_retry,
                cache_filename,
                dot_range=(0, 1024)):
    """Profile costs for all dot and communication primitives."""
    #  pylint: disable=import-outside-toplevel
    from alpa.pipeline_parallel.stage_construction import get_submesh_choices
    print_used_time(None)

    ##### Profile compute cost
    dot_cost_dict = profile_dot(dot_range, device_cluster, cache_filename)
    print_used_time("Profile dot")

    ##### Profile communication cost
    virtual_mesh = device_cluster.get_virtual_physical_mesh()
    submesh_choices = list(
        reversed(
            get_submesh_choices(virtual_mesh.num_hosts,
                                virtual_mesh.num_devices_per_host, "all")))

    # Load failed batch keys
    failed_batch_keys_filename = "tmp/failed_batch_keys.pkl"
    if os.path.exists(failed_batch_keys_filename):
        with open(failed_batch_keys_filename, "rb") as fbkf:
            failed_batch_keys = pickle.load(fbkf)
    else:
        failed_batch_keys = set()

    prof_database = ProfilingResultDatabase()
    for _, (num_hosts, num_devices_per_host) in enumerate(submesh_choices):
        print(f"Mesh shape: {(num_hosts, num_devices_per_host)}")

        # Slice a mesh
        tmp_mesh = virtual_mesh.slice_2d(tuple(range(num_hosts)),
                                         (tuple(range(num_devices_per_host)),) *
                                         num_hosts)
        all_specs = enumerate_all_collective_spec(num_hosts,
                                                  num_devices_per_host,
                                                  max_comm_size_intra_node,
                                                  max_comm_size_inter_node)

        op_infos = []
        for op_type in [
                "all-reduce", "all-gather", "all-to-all", "reduce-scatter"
        ]:
            for spec in all_specs:
                op_infos.append((op_type, spec))

        physical_mesh = tmp_mesh.get_physical_mesh()
        available_memory_per_device = physical_mesh.get_available_memory()

        def get_op_info_key(op_info):  # return (op_type, replica_group)
            return (op_info[0], op_info[1][0])

        # Profile operators in batch to resolve some deadlock issues
        results = []
        s = 0
        fail_ct = 0
        while s < len(op_infos):
            # Decide batch size
            batch_key = get_op_info_key(op_infos[s])
            batch_size = 1
            while (s + batch_size < len(op_infos) and
                   get_op_info_key(op_infos[s + batch_size]) == batch_key):
                batch_size += 1

            print(f"Batch size: {batch_size}, key: {batch_key}")

            # Profile a batch
            if batch_key in failed_batch_keys:
                # This batch is skipped due to too many errors
                batch_result = [np.inf] * batch_size
            else:
                try:
                    batch_result = physical_mesh.profile_hlo_ops(
                        op_infos[s:s + batch_size],
                        cache_filename,
                        single_timeout=bound(fail_ct * 100, 100, 400),
                        batch_timeout=batch_size * 100)
                except ray.exceptions.RayError:
                    batch_result = None

            if batch_result is not None:
                results.extend(batch_result)
                s += batch_size
                fail_ct = 0
            else:
                op_infos[s:s + batch_size] = reversed(op_infos[s:s +
                                                               batch_size])
                fail_ct += 1

                if fail_ct > max_fail_retry:
                    # Skip this batch if there are too many errors
                    print(f"Failed key: {batch_key}")
                    failed_batch_keys.add(batch_key)
                    with open(failed_batch_keys_filename, "wb") as fbkf:
                        pickle.dump(failed_batch_keys, fbkf)

                print(f"Reboot physical mesh. fail_ct: {fail_ct}")
                physical_mesh.shutdown(forced=True)
                physical_mesh = None
                while physical_mesh is None:
                    try:
                        time.sleep(10)
                        tmp_mesh.launched_physical_mesh = None
                        physical_mesh = tmp_mesh.get_physical_mesh()
                    except ray.exceptions.RayError:
                        ray.shutdown()
                        ray.init(address="auto")
                        physical_mesh = None

        # Parse results
        all_gather_cost_dict = defaultdict(list)
        all_reduce_cost_dict = defaultdict(list)
        all_to_all_cost_dict = defaultdict(list)
        reduce_scatter_cost_dict = defaultdict(list)
        for i in range(len(op_infos)):
            op_type, (replica_groups, dtype, size) = op_infos[i]
            array_size = size * to_np_dtype(dtype).itemsize
            num_devices = len(replica_groups[0])

            if op_type == "all-gather":
                communication_size = array_size * (num_devices -
                                                   1) / num_devices
                all_gather_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            elif op_type == "all-reduce":
                communication_size = 2 * array_size * (num_devices -
                                                       1) / num_devices
                all_reduce_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            elif op_type == "all-to-all":
                communication_size = array_size * (
                    num_devices - 1) / num_devices / num_devices
                all_to_all_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            elif op_type == "reduce-scatter":
                communication_size = array_size * (num_devices -
                                                   1) / num_devices
                reduce_scatter_cost_dict[(replica_groups, dtype)].append(
                    (size, results[i]))
            else:
                raise ValueError(f"Invalid op: {op_type}")

            bandwidth = communication_size / results[i]
            print(f"Op: {op_infos[i]}, Bandwidth: {bandwidth / GB:.2f} GB/s")

        physical_mesh.shutdown()

        mesh_result = MeshProfilingResult()
        mesh_result.dot_cost_dict = dot_cost_dict
        mesh_result.all_gather_cost_dict = all_gather_cost_dict
        mesh_result.all_reduce_cost_dict = all_reduce_cost_dict
        mesh_result.all_to_all_cost_dict = all_to_all_cost_dict
        mesh_result.reduce_scatter_cost_dict = reduce_scatter_cost_dict
        mesh_result.available_memory_per_device = available_memory_per_device
        mesh_result.sort_cost_lists()
        mesh_result.make_monotonic()
        prof_database.update_one_mesh(cluster_key,
                                      (num_hosts, num_devices_per_host),
                                      mesh_result)

    print_used_time("Profile communication")
    return prof_database


def estimate_hlo_module_cost(hlo_module,
                             profiling_results,
                             num_micro_batches=1,
                             grad_sync_channel_ids=""):
    """Estimate the cost of an HLO module with the HLO instruction level cost
    model."""
    with XlaPassContext({
            "gpu_cost_model::profiling_results": profiling_results,
            "gpu_cost_model::num_micro_batches": num_micro_batches,
            "gpu_cost_model::grad_sync_channel_ids": grad_sync_channel_ids,
            "gpu_cost_model::verbose": 0,
    }):
        return xe.estimate_hlo_module_cost(hlo_module)
