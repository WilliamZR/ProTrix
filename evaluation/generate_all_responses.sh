cd ..
model_path=$1
device=$2

list=$(ls data/evaluation_data)

## evaluate and compute score for each benchmark
for benchmark in ${list};
do
## benchmark are in format as {benchmark_name}_dev.json
## now extract the benchmark_name
benchmark_name=$(echo $benchmark | tr "_" "\n" | head -n 1)
CUDA_VISIBLE_DEVICES=$device python  evaluation/generate_with_sql.py \
    --model_path $model_path \
    --benchmark $benchmark_name &&
python evaluation/compute_score.py\
    --model_path $model_path \
    --benchmark $benchmark_name 
done
