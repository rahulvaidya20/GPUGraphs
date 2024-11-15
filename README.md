# GPUGraphs

GPU Resource prediction dataset for DNN workloads.

## Requirements
1. Tensorflow-gpu: Tensorflow with GPU support
2. Nvidia GPU, CUDA drivers
3. HLO to NPZ module from https://github.com/google-research-datasets/tpu_graphs#graph-feature-extraction
   
## Description
Generates data using two python script files - pipeline.py and train.py

pipeline.py: This is the main pipeline script that orchestrates data generation by running train.py across various model and parameter combinations. 
Key Functions:
1. dump_hlo_get_gpu_runtime_features: Runs nvidia-smi at intervals (0.5 seconds) during model training to capture GPU metrics like utilization, memory usage, and temperature. Simultaneously, it sets environment variables for HLO dumping.
2. get_largest_pb_file: Identifies the largest .pb file (HLO dump) generated, representing the most resource-intensive part of the model.
extract_hlo_to_npz: Converts the largest HLO .pb file to an NPZ format for efficient storage of graph features.
combine_gpu_features_csv: Combines model and GPU features into a single CSV record, enabling comprehensive performance data storage.
Main Execution Flow: Iterates over combinations of hyperparameters (e.g., optimizer, batch size, epochs) and invokes train.py for each, coordinating the collection of GPU and HLO data and saving them to structured CSV files.

train.py: Trains models and records model-specific metrics (e.g., batch time, epoch time), using a Keras Callback to capture timing data during training. Please modify train.py to support more models.
Key Functions:
1. StepTimeLogger (Callback): Tracks batch and epoch timings for logging in real-time during training.
2. load_model_from_json: Loads and compiles a Keras model architecture from a JSON file, allowing parameterization with various optimizers and learning rates.
3. train_model: Executes the training process with generated random data, simulating actual training to collect timing data.
4. write_to_csv: Logs the collected training metrics (e.g., total time, batch time, epoch time) to CSV.

train.py requires model summary json file as an argument.

## Features
model_name,samples,input_dim_w,input_dim_h,input_dim_c,output_dim,optimizer,epochs,batch,learn_rate,tf_version,cuda_version,batch_time,epoch_time,fit_time,gpu_make,gpu_name,gpu_arch,gpu_cc,gpu_core_count,gpu_sm_count,gpu_memory_size,gpu_memory_type,gpu_memory_bw,gpu_tensor_core_count,max_memory_util,avg_memory_util,max_gpu_util,avg_gpu_util,max_gpu_temp,avg_gpu_temp

Generates HLO dump for all models and stores in numpy (npz) format.

## Usage
```
    python -u pipeline.py \
    -s train.py \
    -hlo ${MODEL_HLO_PATH} \
    -n ${MODEL_NPZ_PATH} \
    --gpu_static_csv ${GPU_STATIC_FEATURES_PATH} \
    --gpu_info_csv ${GPU_STATS_PATH}/gpu_runtime_features.csv \
    --model_stats ${MODEL_STATS_PATH}/model_run_stats.csv \
    --model ${MODEL_NAME} \
    --model_json_path ${MODEL_JSON_PATH} \
    --combined_csv ${COMBINED_CSV_PATH}/combined_stats.csv
```
By default, pipeline.py script run through all these combinations below:
```
    models = ["VGG16", "VGG19", "ResNet50", "ResNet101", "ResNet152", "InceptionV3", "Xception", 
    "MobileNet", "MobileNetV2", "DenseNet121", "DenseNet169", "DenseNet201", 
    "NASNetMobile", "NASNetLarge", "EfficientNetB0", "EfficientNetB1", "EfficientNetB7"]
    optimizers = ["adam", "sgd"]
    samples = [1, 10, 100, 1000]
    epochs = [1, 2, 5, 10]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    learning_rates = [0.01, 0.001, 0.0001]
```
Please generate model summaries in json format from here: https://keras.io/api/applications/
Review generate_json.py for sample

## Required Arguments

```
--model_src_file', '-s' - Path to Model train file. See train.py for reference
--model_hlo_path, -hlo  - Path to store HLO dumps for a given model.
--model_npz_path, '-n'  - Path to store converted npz files for a given model.
--gpu_static_csv        - Path to GPU hardware features stored in CSV format. See gpu_static_features.csv for sample
--gpu_info_csv          - Path to GPU runtime features as csv
--model_stats           - Path to store model runtime stats as csv
--combined_csv          - Path to combine all features into one csv
--model                 - Model to run
--model_json_path       - Path to generated model jsons
--smi_interval          - nvidia-smi interval. Default every 0.5 seconds
  ```
