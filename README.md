# ProTrix
Code and data for ProTrix: Building Models for Planning and Reasoning over Tables with Sentence Context

![Plan-then-Reason](framework.png)

## Introduction
We propose a *Plan-then-Reason* framework to answer user queries on tabular tasks with sentence context. The framework first plans the reasoning pathways by ingesting the query and the context, and assigns each step to textual and program-based reasoning to arrive at the final answer. We construct TrixInstruct, an instruction-tuning set to build models with generalizability and interpretability over tables with sentence context. We develop ProTrix, capable of planning and reasoning on tabular tasks with sentence context. ProTrix can generalize to unseen tasks and achieves comparable performance with GPT-3.5-turbo.

## Models
Coming Soon!

We will release our models [ProTrix](TODO) and [ProTrix-Coder](TODO) on huggingface.

## Environments
```
conda create --name protrix --file requirements.txt
```
We use [vllm](https://github.com/vllm-project/vllm) to speed up the inference. 
```
conda activate protrix
```
```
├── data
│   ├── evaluation_data
│   ├── outputs
│   └── TrixInstruct.json
├── evaluation
│   ├── compute_score.py
│   ├── evaluate_with_llm.py
│   ├── generate_all_responses.sh
│   ├── sql_tool.py
│   └── evaluate_with_sql.py
├── README.md
└── requirement.txt
```

## Data Format of TrixInstruct
```
{
    'id': instance id from the original dataset,
    'instruction': instruction for generation the responses,
    'output': response obtain from GPT-4,
    'answer': gold answer from the original dataset
}
```

## Inference
You can run the following command to generate result for a specific benchmark. We splits data on different GPUs to speed up the process with multiprocessing. The result will be saved at data/outputs/{model_name}
```
CUDA_VISIBLE_DEVICE={} python evaluation/evaluate_with_sql.py --model_path {your_path_to_protrix} --benchmark {benchmark}
```
Or you can run the following command to generate for all the benchmarks in data/evaluation_data
```
cd evaluation
sh generate_all_responses.sh {protrix_path} {device}
```
This script will generate an approximation of the evaluation result. 
## Evaluation
Since our model is not trained to follow the rule or grammar of each dataset, we provide two methods to compute the final score.

**Heuristic Matching Method**

We try to match the answer in the concluding sentence. This method is not perfect but there is only a tiny proporation of mismatch based on our human evaluation. We use this method for developing our model. We use this metric to report the final result of fact verification tasks in our paper.
```
python evaluation/compute_score.py --benchmark {benchmark} --model_path {model_path}
```
**LLM Method**

We employ Llama-2-70B-chat to access the correctness of the final answer. We use this method to report the final result of question answering tasks in our paper.
```
python evaluate_with_llm.py --benchmark {benchmark} --model_path {model_path} --llm_path {llm_path}
```
```
python compute_score.py --benchmark {benchmark} --model_path {model_path} --eval_with_llm
```
## Citation 
```
To be updated
```

## License
Check out the [license for our models](https://github.com/facebookresearch/llama/blob/main/LICENSE). Our curated dataset is under the MIT license.

## Contact
If you have any questions or want to discuss future research directions, feel free to email ziruiwu@pku.edu.cn