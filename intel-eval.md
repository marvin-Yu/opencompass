#### API Models with CPU-only

```bash
conda create -n opencompass python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass && git checkout -b 31afe870267de4e216bf56936ae07e027886c5d7 intel-eval
pip install -e .
# also please install requirements packages via `pip install -r requirements/api.txt` for API models if needed.
pip install -r requirements/api.txt
```

#### 使用 OpenCompass 提供的更加完整的数据集 (~500M)，可以使用下述命令进行下载和解压：

```bash
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;
```


python run.py --models cmri_ailab_qianwen14_kunlun --datasets sft_1