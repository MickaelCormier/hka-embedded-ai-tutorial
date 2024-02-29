import onnxruntime as ort
import numpy as np
import time

def run_infer(a_model_path, a_input_data, a_num_runs=1, a_use_cuda=False):
    # Init ONNX Session
    providers = ["CUDAExecutionProvider"] if a_use_cuda else ["CPUExecutionProvider"]
    print(f'Provider: {providers}')

    print('Init ONNX Session.')
    session_options = ort.SessionOptions()
    session = ort.InferenceSession(
        a_model_path,
        session_options,
        providers=providers
    )

    # Get model info
    model_inputs = session.get_inputs()
    input_names = [input.name for input in model_inputs]
    print(f'Input names: {input_names}')

    input_shapes = [input.shape for input in model_inputs]
    print(f'Input shapes: {input_shapes}')

    model_outputs = session.get_outputs()
    output_names = [output.name for output in model_outputs]
    print(f'Output names: {output_names}')

    output_shapes = [output.shape for output in model_outputs]
    print(f'Output shapes: {output_shapes}')

    # Run inference
    execution_times = []

    print('Start inference.')

    for _ in range(a_num_runs):
        start_time = time.time()
        result = session.run(output_names, {input_names[0]: a_input_data})
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
    model_path = 'yolov8_onnx_mo/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.onnx'
    input_data = np.random.rand(1, 3, 640, 640).astype(np.float32)

    print('CPU:')
    result = run_infer(model_path, input_data, 50, False)
    # print(result)
    print('GPU:')
    result = run_infer(model_path, input_data, 50, True)
    # print(result)
