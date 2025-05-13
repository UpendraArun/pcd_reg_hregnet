## Table of Contents
[1. Recommended Environment](#recommended-environment)

[2. Dataset Preparation](#dataset-preparation)

[3. Docker Env Setup](#Docker-Env-Setup)

[4. Train and Test](#train-and-test)

[5. Models](#models)

[6. Add your custom model and dataset](#Add-your-custom-model-and-dataset)

[7. Registration results](#Registration-results)

[8. Useful Commands](#useful-commands)

[9. Acknowledgments](#acknowledgments)

## Recommended Environment

### System Requirements
- Ubuntu 22.04
- Python >= 3.11
- CUDA 12.4
- Rest all of the environment requirements are as mentioned in the `Dockerfile`
 
## Dataset preparation
Store all the datasets in one folder `data`

Dataset used in this thesis:
- **[MANTruckScenes](https://brandportal.man/d/QSf8mPdU5Hgj/downloads)**

Other Datasets that can be used 
- **[A2D2](https://www.a2d2.audi/a2d2/en/download.html)**
- **[NuScenes](https://www.nuscenes.org/nuscenes#download)**
- **[KITTI](https://semantic-kitti.org/dataset.html#download)**

#### MANTruckScenes dataset folder structure
```
data
├── truckscenes
│   ├── samples
│   │   ├──LIDAR_LEFT
│   │   ├──LIDAR_RIGHT
│   ├── sweeps
│   ├── v1.0-trainval
├── truckscenes_test
│   ├── samples
│   │   ├──LIDAR_LEFT
│   │   ├──LIDAR_RIGHT
│   ├── sweeps
│   ├── v1.0-test
├── truckscenes_mini
│   ├── samples
│   │   ├──LIDAR_LEFT
│   │   ├──LIDAR_RIGHT
│   ├── sweeps
│   ├── v1.0-mini
```


## Docker Env Setup 

```
cd ~/pcd_reg_hregnet
```

```
docker build -t pcd_calib:latest -f Dockerfile .
```

```
cd ..
```

The following docker command creates a container named `pcd_reg` with follwing volumes attached.
- pcd_reg_hregnet
- data (the folder with all the datasets)
Replace `your_dataset_path` with your local dataset folder path

```
docker run --privileged --name pcd_reg --gpus device='all,"capabilities=compute,utility,graphics"' -it --network host --shm-size 64GB -v $(pwd)/pcd_reg_hregnet:/workspace -v /your_dataset_path/:/workspace/data pcd_calib:latest /bin/bash
```


Once inside the contatiner, install `point_utils` package
```
cd /workspace/Models/HRegNet/PointUtils
```

```
pip install .
```

Login to wandb from terminal to setup logging - 
```
wandb login
``` 


## Train and Test

To train a model, run the following command from `/workspace`:
```bash
sh scripts/train_man_registration.sh
```

To test the model on a test dataset, use:
```bash
sh scripts/test_man.sh
```

Before running, ensure the relevant hyperparameters and directory paths are updated in the respective `.sh` files.

## Models

- HRegNet 	- Original HRegNet model
- Model V1 	- HRegNet with MI added before coarse registration layer
- Model V2 	- HRegNet with MI added after coarse registration layer (aka **Adaption 1 / A1**)
- Model V3 	- Model V2 with regression head 
- Model V4 	- Model V3 with Overlap loss + MI Loss
- Model V5 	- Unused custom model with  self and cross attention between point clouds
- Model V6 	- Backbone changed to Point Transformer V3(Ptv3) (aka **Adaption 2 / A2**, Encoder only Ptv3 used with no pooling)
- Model V6a - Backbone changed to Ptv3 (Encoder Decoder setup with Serialized Pooling) 

- Caution:
        Ptv3 uses flash attention which works only on GPUs newer than Ampere architecture.
        If working with GPU older than Ampere, set `enable_flash=False` inside `__init__()` of `Class PointTransformerV3` in `models/model_v6/ptv3_mod.py`

## Registration results

Initial decalibration of ±20°, ±0.50m on the source point cloud

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">Rotation [deg]</th>
      <th colspan="2">Translation [m]</th>
    </tr>
    <tr>
      <th>MAE</th>
      <th>STD</th>
      <th>MAE</th>
      <th>STD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HRegNet</td>
      <td>0.133</td>
      <td>0.053</td>
      <td>0.058</td>
      <td>0.144</td>
    </tr>
    <tr>
      <td>A1</td>
      <td>0.142</td>
      <td>0.046</td>
      <td>0.054</td>
      <td>0.212</td>
    </tr>
    <tr>
      <td>A2</td>
      <td>0.144</td>
      <td>0.042</td>
      <td>0.049</td>
      <td>0.139</td>
    </tr>
    <tr>
      <td>A2 + Point-to-Point ICP</td>
      <td>0.100</td>
      <td>0.583</td>
      <td>0.286</td>
      <td>0.120</td>
    </tr>
    <tr>
      <td>A2 + Point-to-Plane ICP</td>
      <td>0.078</td>
      <td>0.109</td>
      <td>0.191</td>
      <td>0.057</td>
    </tr>
    <tr>
      <td>Point-to-Point ICP</td>
      <td>0.394</td>
      <td>1.301</td>
      <td>0.386</td>
      <td>0.698</td>
    </tr>
    <tr>
      <td>Point-to-Plane ICP</td>
      <td>0.302</td>
      <td>1.401</td>
      <td>0.354</td>
      <td>0.787</td>
    </tr>
  </tbody>
</table>


<div align="center">
	<img src="assets/S1.png" width="33%" />
	<p align="center"><em>Sample Scene</em></p>
</div>

<div style="display: flex; justify-content: center;">
	<img src="assets/S1_ground_truth.png" width="33%" />
	<img src="/assets/S1_initial_decalib.png" width="33%" />
	<img src="/assets/S1_A2_output.png" width="33%" />
</div>


## Add Custom Model, Loss Function, and Dataset

This framework supports easy integration of custom models, loss functions, and datasets. Follow these steps:

1. **Add Model**:  
	- Create a folder for your model under `models/` (e.g., `models/custom_model/`).
	- Implement your model in this folder.

2. **Add Loss Function**:  
	- Implement your custom loss function in the `losses/` directory.

3. **Add Dataset**:  
	- Place your dataset in the `data/` folder.  
	- Create a dataloader in `dataset/` and update `dataset/config.json` with the dataset configuration.  
	- Import your dataloader in `data_loader.py`.

4. **Update Training Script**:  
	- Copy `train_reg_vx.py` and modify it to include your custom model, loss function, and dataloader.  
	- Adjust the training logic as needed.

## Useful Commands:
- **Check GPU**:  
	`nvidia-smi`, `watch nvidia-smi`, `nvcc --version`

- **Test Internet Connection inside container**:  
	`ping google.com`

- **Restart Docker**:  
	`sudo systemctl restart docker`

- **Verify CUDA**:  
	```python
	import torch
	torch.cuda.is_available()
	```

- **Disk Usage**:  
	`df -h`, `du -sh /home`


### Using `tmux` for Multiple Training Sessions

When running a container, it typically operates with a single terminal in the background (not the VS Code terminal), even after closing the VS Code GUI. To run multiple training sessions simultaneously, `tmux` can be very helpful.

#### Common `tmux` Commands:
- **Create a new session**:  
	```bash
	tmux new -s <session_name>
	```
- **Detach from a session without terminating it**:  
	Press `Ctrl + B`, then press `D`.
- **List all active sessions**:  
	```bash
	tmux list-sessions
	```
- **Reattach to a detached session**:  
	```bash
	tmux attach -t <session_name>
	```
- **Terminate a session completely**:  
	```bash
	tmux kill-session -t <session_name>
	```


## Acknowledgments

- **HRegNet**: [HRegNet](https://github.com/ispc-lab/HRegNet)

- **PointTransformerV3**: [PTv3](https://github.com/Pointcept/PointTransformerV3)

- **Mutual Information Loss**: [MMI_PCN](https://github.com/wendydidi/MMI/tree/main/MI_PCN)

- **Chamfer Distance**: [otaheri/chamfer_distance](https://github.com/otaheri/chamfer_distance)

- **Overlap Loss**: [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)

- **Metrics and Visualize**: [CalibViT](https://github.com/extrinsic-calibration/CalibViT)




