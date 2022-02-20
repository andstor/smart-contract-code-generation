# High Performance Computing

> Idun HPC @ NTNU


## Getting started

### Connect
```shell
ssh username@idun-login1.hpc.ntnu.no
ssh username@idun-login2.hpc.ntnu.no
```

If outside of campus network, use VPN.

### GPT-NeoX
```shell
./setup/gpt-neox.sh
```

## Configuration

### GPUs

* P100 (40 available GPUS) with 16GB
* V100 (38 available GPUS) with 16GB and 32GB
* A100 (64 available GPUS) with 40GB and 80GB

### GPU Memory
|Memory|GPU Name|
|:---:|:---:|
|16G|P100|
|16G|V100|
|32G|V10032|
|40G|A100m40|
|80G|A100m80|

### Script
```shell
#SBATCH --gres=gpu:A100m40:1
```

## Commands

### Monitor logs in real time
```shell
tail -f -n +1 /path/file.log
```

### Controlling Slurm jobs
#### **Submit Slurm job**
Create Slurm job file and make it executable.
```shell
chmod u+x job.slurm
```
Dispatch job.slurm to the cluster by:
```shell
sbatch job.slurm
```

#### **Get all jobs**
```shell
squeue
```

#### **Get all jobs for user `< only pending | only running >` in `<partition>`**
```shell
squeue -u username <-t PENDING|-t RUNNING> <-p partition>
```

#### **Show detailed info on `<jobid>`**
```shell 
scontrol show jobid -dd <jobid>
```

#### **Cancel specific `<jobid>`**
```shell
scancel < jobid >
```

#### **Cancel all `<pending>` jobs for `<username>`**
```shell
scancel <-t PENDING> -u <username>
```


