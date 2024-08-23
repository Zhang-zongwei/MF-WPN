# Multisource spatiotemporal fusion deep neural network for wind speed prediction

*2024.08*

## Abstract

Accurate short-term wind speed prediction is crucial for maintaining the safe, stable, and efficient operation of wind power systems. We propose a multisource meteorological data fusion wind prediction network (MF-WPN) to study fine-grid vector wind speed prediction, taking Northeast China as an example. The MF-WPN models the deterministic evolution of winds via a new spatiotemporal encoder–decoder and captures the uncertain evolution of winds from the precursor evolution of geopotential and temperature via multisource spatiotemporal fusion modules. Finally, training constraints are applied to the wind magnitude, structure, and direction via composite loss. Experiments show that the mean square error of the wind speed predicted by the MF-WPN is 0.26 m/s at the first hour, 0.52 m/s at the third hour, and 1.01 m/s at the 12th hour; the accuracy of the wind direction prediction in eight directions reaches 98% at the first hour and 93% at the third hour. Transfer experiments demonstrate the outstanding generalized performance of the MF-WPN, which can be quickly applied to offsite prediction. Efficiency experiments show that the MF-WPN takes only 18 ms to predict vector wind speeds on a 24-hour fine grid over the future northeastern region. With its demonstrated accuracy and efficiency, the MF-WPN can be an effective tool for predicting vector wind speeds in large regional wind centers and can help in ultrashort- and short-term deployment planning for wind power.

![Graphabstract](data/graphabstract.jpg)


## Installation

```
conda create -n wpn python=3.8
conda activate wpn
git clone https://github.com/Zhang-zongwei/MF-WPN.git

pip install -r requirements.txt
```

## Overview

- `data/:` contains a test set for the northeast region of the manuscript, which can be downloaded via the link .
- `openstl/models/mfwpn.py:` contains the network architecture of this MF-WPN.
- `openstl/modules/:` contains partial modules of the MF-WPN.
- `utils/:` contains data processing files and loss calculations..
- `chkfile/:` contains weights for predicting 24-hour wind speeds in the Northeast region, which can be downloaded via the link.
- `result/:` contains predicted wind speed results and evaluation methods.
- `config.py：`  training configs for the MF-WPN.
- `main.py:` Train the MF-WPN.
- `test.py` Test the MF-WPN.

## Data preparation
The data used in this study and its processing have been described in detail in the manuscript. To facilitate the testing, we have prepared the [MF-WPN weights](https://drive.google.com/file/d/1YrJP1sCWUcsHcYdNL_sWFbkuS4WfaeJf/view?usp=sharing) and the [test dataset](https://drive.google.com/drive/folders/1qQMV8xBRDI5Vg9pxigLAJNEOtNC4O87x?usp=sharing).

## Train
After the data is ready, use the following commands to start training the model:
```
python main.py
```

## Test
We provide the test model weights and test dataset, which can be tested using the following commands after downloading:
```
python test.py
```

Note that the predictions are obtained as npy files containing u, v variables, which need to be converted to wind speed results using: 
```
python uv_to_wind.py
```
After that, we can obtain the wind speed prediction evaluation result by:
```
python evaluate.py
```
## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL). We sincerely appreciate for their contributions.
