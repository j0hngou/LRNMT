- Copy the whole folder:
```scp -r /path/to/here/ lcur____@lisa.surfsara.nl:/path/to/there/```

- Load the modules to have conda:
```
module load 2021
module load Anaconda3/2021.05
```

- Open a gpu node with a terminal: (1 hour to install stuff, blocking call to cluster thus once you get assigned a node you won't get kicked out)
```srun -p gpu_shared_course -n 1 --mem=32000M --ntasks-per-node 1 --gpus 1 --cpus-per-task 2 -t 1:00:00 --pty /bin/bash```

> User should change to lcur____@"random_string"

- Install the environment:
```
cd NMTDistillation
conda env create -f env.yml
```

- Exit the node and reenter to restart the shell (conda shenanigans)
```
exit
srun -p gpu_shared_course -n 1 --mem=32000M --ntasks-per-node 1 --gpus 1 --cpus-per-task 2 -t 1:00:00 --pty /bin/bash
conda activate dl4nlp
```


- Activate the environment:
```conda activate dl4nlp```

- Run the code:
```python main.py```

- If you wanna set up a train job: use [train.job](lisa/train.job). ```sbatch lisa/train.job```. If there's a waitlist for the titanrtx, change ```#SBATCH --partition=gpu_titanrtx_shared_course``` to ```#SBATCH --partition=gpu_shared_course```.