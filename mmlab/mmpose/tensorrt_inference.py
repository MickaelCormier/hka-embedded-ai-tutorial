import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import ctypes

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def load_engine(a_engine_path):
    ctypes.CDLL('/mmlab/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    trt_runtime = trt.Runtime(TRT_LOGGER)

    with open(a_engine_path, 'rb') as engine_file:
        engine_data = engine_file.read()

    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    return engine

def trt_dtype_to_str(a_trt_dtype):
    if a_trt_dtype == trt.DataType.FLOAT:
        return 'FP32'
    elif a_trt_dtype == trt.DataType.HALF:
        return 'FP16'
    elif a_trt_dtype == trt.DataType.INT32:
        return 'INT32'
    elif a_trt_dtype == trt.DataType.INT8:
        return 'INT8'
    elif a_trt_dtype == trt.DataType.BOOL:
        return 'BOOL'
    else:
        return 'Unknown'
    
def trt_dtype_to_np_dtype(a_trt_dtype):
    if a_trt_dtype == trt.DataType.FLOAT:
        return np.float32
    elif a_trt_dtype == trt.DataType.HALF:
        return np.float16
    elif a_trt_dtype == trt.DataType.INT32:
        return np.int32
    elif a_trt_dtype == trt.DataType.INT8:
        return np.int8
    elif a_trt_dtype == trt.DataType.BOOL:
        return np.bool_
    else:
        return 'Unknown'

def allocate_buffers(a_engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in a_engine:
        shape = a_engine.get_tensor_shape(binding)
        size = trt.volume(shape) * 1
        dtype = a_engine.get_binding_dtype(binding)
        host_mem = cuda.pagelocked_empty(size, trt_dtype_to_np_dtype(dtype))
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if a_engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            print(f'Input shape: {shape}, Type: {trt_dtype_to_str(dtype)}')
            inputs.append(HostDeviceMem(host_mem, device_mem))
        elif a_engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            print(f'Output shape: {shape}, Type: {trt_dtype_to_str(dtype)}')
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def run_infer(a_engine, a_input_data, a_num_runs=1):
    print('Init TRT & allocate buffers')
    inputs, outputs, bindings, stream = allocate_buffers(a_engine)
    context = a_engine.create_execution_context()

    # Run inference
    execution_times = []
    print('Start inference.')

    for _ in range(a_num_runs):
        start_time = time.time()

        np.copyto(inputs[0].host, a_input_data.ravel())

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

        stream.synchronize()
        result = [out.host for out in outputs]
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        execution_times.append(execution_time)

    # Remove outliers
    Q1 = np.percentile(execution_times, 25)
    Q3 = np.percentile(execution_times, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    no_outliers = [time for time in execution_times if time >= lower_bound and time <= upper_bound]

    avg_time = np.mean(execution_times)
    max_time = np.max(execution_times)
    min_time = np.min(execution_times)
    std_dev = np.std(execution_times)

    print(f'With outliers: avg time: {avg_time} ms, max time: {max_time} ms, min time: {min_time} ms, std dev: {std_dev} ms')

    avg_time = np.mean(no_outliers)
    max_time = np.max(no_outliers)
    min_time = np.min(no_outliers)
    std_dev = np.std(no_outliers)

    print(f'Without outliers: avg time: {avg_time} ms, max time: {max_time} ms, min time: {min_time} ms, std dev: {std_dev} ms')

    return result

if __name__ == '__main__':
    engine_path = 'export/end2end.engine'
    engine = load_engine(engine_path)
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    result = run_infer(engine, input_data, 50)
    # print(result)
