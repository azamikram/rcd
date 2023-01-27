## Setup
#### Install python env
```bash
sudo apt update
sudo apt install -y build-essential \
                 python-dev \
                 python3-venv \
                 python3-pip \
                 libxml2 \
                 libxml2-dev \
                 zlib1g-dev \
                 python3-tk \
                 graphviz

cd ~
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
```

#### Install dependencies
```bash
git clone https://github.com/azamikram/rcd.git
cd rcd
pip install -r requirements.txt
```

#### Link modifed files
To implement RCD, we modified some code from pyAgrum and causal-learn.
Some of these changes expose some internal information for reporting results (for example number of CI tests while executing PC) or modify the existing behaviour (`local_skeleton_discovery` in `SekeletonDiscovery.py` implements the localized approach for RCD). A few of these changes also fix some minor bugs.

Assuming the rcd repository was cloned at home, execute the following;
```bash
ln -fs ~/rcd/python-simulation/pyAgrum/lib/image.py ~/env/lib/python3.8/site-packages/pyAgrum/lib/
ln -fs ~/rcd/python-simulation/causallearn/search/ConstraintBased/FCI.py ~/env/lib/python3.8/site-packages/causallearn/search/ConstraintBased/
ln -fs ~/rcd/python-simulation/causallearn/utils/Fas.py ~/env/lib/python3.8/site-packages/causallearn/utils/
ln -fs ~/rcd/python-simulation/causallearn/utils/PCUtils/SkeletonDiscovery.py ~/env/lib/python3.8/site-packages/causallearn/utils/PCUtils/
ln -fs ~/rcd/python-simulation/causallearn/graph/GraphClass.py ~/env/lib/python3.8/site-packages/causallearn/graph/
```

## Using RCD

#### Generate Synthetic Data
```sh
./gen_data.sh
```

#### Executing RCD with Synthetic Data
```sh
./rcd.py --path [PATH_TO_DATA] --local --k 3
```

`--local` options enables the localized RCD while `--k` estimates the top-`k` root causes.
