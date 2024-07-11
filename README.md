
# Powershell set proxy
```shell
$env:HTTP_PROXY="http://xxx.com:xxx"
$env:HTTPS_PROXY="http://xxx.com:xxx"

conda create -n pytorch_20 python=3.8

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install torchvision
git clone -b v4.34.1 --depth 1 https://github.com/huggingface/transformers.git
pip install -e ./
conda install datasets
pip install datasets evaluate accelerate transformers==4.34.1 scipy scikit-learn --proxy=xxx.com:xxx
cd .\transformers\examples\pytorch\text-classification\

$env:HTTP_PROXY="http://xxx.com:xxx"
$env:HTTPS_PROXY="http://xxx.com:xxx"
$env:HF_DATASETS_OFFLINE=1
$env:HF_EVALUATE_OFFLINE=1
$env:TRANSFORMERS_OFFLINE=1
python run_glue.py --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english --task_name sst2 --do_eval   --max_seq_length 384 --output_dir ./tmp --per_device_eval_batch_size 24 --dataloader_drop_last

python run_glue.py --model_name_or_path "deepset/roberta-base-squad2" --task_name sst2 --do_eval   --max_seq_length 384 --output_dir ./tmp --per_device_eval_batch_size 24 --dataloader_drop_last
```

# Core bingding in Power Shell
```shell
cmd.exe /c "start /affinity 3FFF python -i  torchvision_models.py"

cmd.exe /c "start /affinity 3FFF python -i run_glue.py --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english --task_name sst2 --do_eval   --max_seq_length 384 --output_dir ./tmp --per_device_eval_batch_size 8 --dataloader_drop_last"

cmd.exe /c "start /affinity 3FFF python -i run_glue.py --model_name_or_path deepset/roberta-base-squad2 --task_name sst2 --do_eval   --max_seq_length 384 --output_dir ./tmp --per_device_eval_batch_size 8 --dataloader_drop_last"
```