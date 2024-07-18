Marigold setup steps:

First install miniforge:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Then activate miniforge:
```bash
source /home/$USER/miniforge3/bin/activate
```

Create enviroment and install dependencies into it
```bash
mamba env create -n marigold --file environment.yaml
conda activate marigold
```

Lastly, install required packages for training
```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```
