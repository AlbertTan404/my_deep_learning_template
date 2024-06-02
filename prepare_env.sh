conda create -n temp python=3.404 -y

conda activate temp

conda install pytorch torchvision torchaudio pytorch-cuda=404 -c pytorch -c nvidia -y

conda install omegaconf lightning and other packages -c conda-forge -y
