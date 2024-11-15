import os
import time
import argparse
import subprocess as sp
import numpy as np
from pathlib import Path
import csv
import shlex
from itertools import product
import pandas as pd
import sys
import graph_features
import uuid
import zipfile

def proc(x):
    return float(x.split(' ')[0])

def dump_hlo_get_gpu_runtime_features(model_name, command, hlo_path, smi_interval):  
    smi_command = [
        'nvidia-smi',
        '--query-gpu=name,utilization.memory,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits',
        '-i', '0'
    ]
    # Run the command and decode the output
    result = sp.check_output(smi_command).decode('utf-8').strip()

    # Split the result into individual values
    gpu_name, memory_utilization, gpu_utilization, gpu_temperature = result.split(', ')

    # Convert to appropriate types
    gpu_name = gpu_name.strip()
    max_memory_usage = float(memory_utilization)
    max_gpu_usage = float(gpu_utilization)
    max_gpu_temperature = float(gpu_temperature)
    avg_memory_usage = 0
    avg_gpu_usage = 0
    avg_gpu_temp = 0
    count = 0

    xla_dump_to_val = f'{hlo_path}/{model_name}'
    env = os.environ.copy()
    env['XLA_FLAGS'] = f'--xla_dump_hlo_as_proto --xla_dump_to={xla_dump_to_val}'
    timeout = 600  
    start_time = time.time()
    #print(shlex.split(command))
    #print(gpu_name, cuda_version, driver_version, used_memory, total_memory, max_memory_usage, max_gpu_usage, max_gpu_temperature)
    model_train_process = sp.Popen(shlex.split(command), env=env, stdout=sp.DEVNULL)

    while model_train_process.poll() is None and (time.time() - start_time) < timeout:
        command = [
            'nvidia-smi', 
            '--query-gpu=utilization.memory,utilization.gpu,temperature.gpu', 
            '--format=csv,noheader,nounits',
            '-i', '0'
        ]
        result = sp.check_output(command).decode('utf-8').strip()
        
        # Parse the result for memory utilization, GPU utilization, and temperature
        memory_utilization, gpu_util, gpu_temp = map(float, result.split(', '))

        # Track max values for each metric
        max_memory_usage = max(max_memory_usage, memory_utilization)
        max_gpu_usage = max(max_gpu_usage, gpu_util)
        max_gpu_temperature = max(max_gpu_temperature, gpu_temp)

        count += 1
        avg_gpu_usage += gpu_util
        avg_memory_usage += memory_utilization
        avg_gpu_temp += gpu_temp

        time.sleep(smi_interval)

    process_success = 0
    if (time.time() - start_time) >= timeout:
        print("Timelimit exceeded.")
    if model_train_process.poll() == 0:
        print(f"HLO dump successful for {model_name}.")
        process_success = 1
    else:
        print(f"HLO dump failed for {model_name}.")
       
    return process_success, gpu_name, [round(max_memory_usage, 2), round(avg_memory_usage/count, 2), max_gpu_usage, round(avg_gpu_usage/count, 2), max_gpu_temperature, round(avg_gpu_temp/count, 2)]

def get_largest_pb_file(hlo_model_path):
    largest_pb_file = None
    pb_size = 0

    for file_path in Path(hlo_model_path).rglob('*.pb'):
        file_size = file_path.stat().st_size  # Get the size of the file in bytes   
        if file_size > pb_size:
            pb_size = file_size
            largest_pb_file = file_path
    #print(largest_pb_file)
    return largest_pb_file

def extract_hlo_to_npz(largest_pb_file, model_npz_file):
    node_opcode, node_feat, edge_index, node_config_ids, node_splits = graph_features.extract_graph_features(str(largest_pb_file))
    npz_dict = {
    'node_opcode': node_opcode,
    'node_feat': node_feat,
    'edge_index': edge_index,
    'node_config_ids': node_config_ids,
    'node_splits': node_splits
    }
    np.savez(model_npz_file, **npz_dict)
    print("Successfully extracted pb dump and converted to npz.")

def hlo_compress_cleanup(model_hlo_path):
    hlo_zip = f"{model_hlo_path}.zip"
    with zipfile.ZipFile(hlo_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(model_hlo_path):
            file_path = os.path.join(model_hlo_path, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=file)
                os.remove(file_path)
    os.rmdir(model_hlo_path)

def extract_gpu_static_features(gpu_name, gpu_static_csv):
    with open(gpu_static_csv, mode='r') as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            if row['gpu_name'] in gpu_name:
                return [
                    row['gpu_make'], gpu_name, row['gpu_arch'], row['gpu_cc'], row['gpu_core_count'],
                    row['gpu_sm_count'], row['gpu_memory_size'],
                    row['gpu_memory_type'], row['gpu_memory_bw'],
                    row['gpu_tensor_core_count']
                ]
    
    print(f'GPU static features not found for {gpu_name}.')
    return None

def combine_gpu_features_csv(combined_csv, model_run_stat_csv, npz_path, gpu_static_features, gpu_runtime_features, gpu_info_csv):
    headers = ['npz_path', 'gpu_make', 'gpu_name', 'gpu_arch', 'gpu_cc', 'gpu_core_count', 'gpu_sm_count', 'gpu_memory_size', 'gpu_memory_type', 'gpu_memory_bw', 
               'gpu_tensor_core_count', 'max_memory_util', 'avg_memory_util', 'max_gpu_util', 'avg_gpu_util', 'max_gpu_temp', 'avg_gpu_temp']
    
    gpu_features = [npz_path] + gpu_static_features + gpu_runtime_features

    csv_file = os.path.isfile(gpu_info_csv)
    with open(gpu_info_csv, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        if not csv_file:
            csv_writer.writerow(headers)
        
        csv_writer.writerow(gpu_features)

    # Check if model_run_stat_csv exists
    if os.path.exists(model_run_stat_csv):
        model_run_stat_df = pd.read_csv(model_run_stat_csv).tail(1)
        gpu_info_df = pd.read_csv(gpu_info_csv).tail(1)
        combined_df = pd.concat([model_run_stat_df, gpu_info_df], axis=1)
        file_exists = os.path.isfile(combined_csv)
        combined_df.to_csv(combined_csv, mode='a', index=False, header=not file_exists)
        print(f'GPU static and runtime features, model features combined in {combined_csv}.')

    else:
        print(f"{model_run_stat_csv} does not exist")

def main(model_name, command, hlo_path, npz_path, gpu_static_csv, gpu_info_csv, smi_interval, combined_csv, model_stats_csv):
    if os.path.exists(combined_csv): #Comment this if condition for multiple-runs
        df = pd.read_csv(combined_csv)
        if df['npz_path'].astype(str).str.contains(model_name[:-5]).any():
            print(f"Skipping: {model_name}. Data exists.")
            return

    hlo_success, gpu_name, gpu_runtime_features = dump_hlo_get_gpu_runtime_features(model_name, command, hlo_path, smi_interval)

    if hlo_success:
        hlo_model_path = os.path.join(hlo_path, model_name)
        model_npz_file = os.path.join(npz_path, model_name + '.npz')
        largest_pb_file =  get_largest_pb_file(hlo_model_path)
        extract_hlo_to_npz(largest_pb_file, model_npz_file)
        gpu_static_features = extract_gpu_static_features(gpu_name, gpu_static_csv)
        if gpu_static_features:
            combine_gpu_features_csv(combined_csv, model_stats_csv, model_npz_file, gpu_static_features, gpu_runtime_features, gpu_info_csv)
        hlo_compress_cleanup(hlo_model_path)
    else:
        print(f"Skipping: {model_name}. Cannot generate HLO.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_src_file', '-s', required=True, type=str)
    parser.add_argument('--model_hlo_path', '-hlo', required=True, type=str)
    parser.add_argument('--model_npz_path', '-n', required=True, type=str)
    parser.add_argument('--gpu_static_csv', required=True, type=str)
    parser.add_argument('--gpu_info_csv', required=True, type=str)
    parser.add_argument('--model_stats', required=True, type=str)
    parser.add_argument('--combined_csv', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--model_json_path', required=True, type=str)
    parser.add_argument('--smi_interval', default=0.5)
    args = parser.parse_args()

    models = ["VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3", "Xception", 
    "MobileNet", "MobileNetV2", "DenseNet121", "DenseNet169", "DenseNet201", 
    "NASNetMobile", "NASNetLarge", "EfficientNetB0", "EfficientNetB1", "EfficientNetB7"]
    optimizers = ["adam", "sgd"]
    samples = [1, 10, 100, 1000]
    epochs = [1, 2, 5, 10]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    learning_rates = [0.01, 0.001, 0.0001]

    for optimizer, sample, epoch, batch, lr in product(optimizers, samples, epochs, batch_sizes, learning_rates):
        if batch <= sample:
            lr_str = str(lr).replace('.', '')
            unique_number = int(str(uuid.uuid4().int)[:4])
            model_name = f"{args.model}_opt{optimizer}_s{sample}_e{epoch}_b{batch}_lr{lr_str}_{unique_number}"
            command = f"python {args.model_src_file} -f {args.model_json_path}/{args.model}_architecture.json -o {optimizer.lower()} -n {sample} -e {epoch} -b {batch} -l {lr} -c {args.model_stats}"
            print(f"\nRunning {model_name}")
            main(model_name, command, args.model_hlo_path, args.model_npz_path, args.gpu_static_csv, args.gpu_info_csv, args.smi_interval, args.combined_csv, args.model_stats)
