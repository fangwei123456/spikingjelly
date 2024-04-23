
# Fully Spiking Actor Network With Intralayer Connections for Reinforcement Learning

## Dependency

We suggest to use anaconda install all packages.

Install `torch>=1.5.0` by referring to:

https://pytorch.org/get-started/previous-versions/

Install `tensorboard`:

```shell
pip install tensorboard
```

Install `SpikingJelly`:

```shell
pip install spikingjelly
```

## Prepare MuJoCo Engine

Install `gym==0.18.0`:

```shell
pip install gym==0.18.0
```

Install `MuJoCo==2.1.0` by referring to:

https://github.com/openai/mujoco-py#install-mujoco

## Training

Running the following commands, we can train the SNN:

```python
python hybrid_td3_cuda_norm.py --env HalfCheetah-v3 --encoder_pop_dim 10 --decoder_pop_dim 10 --encoder_var 0.15 --start_model_idx 0 --num_model 10 --epochs 100 --device_id 0 --root_dir [YOUR_DIR] --encode pop-det --decode last-mem
```

We can also try different hyper-parameters to get higher performance.

## Inference

After training, the model weights will be saved in the './params/xxx' directory. Then we can load weights and run the inference using the same hyper-parameters.

```python
python test_hybrid_td3_cpu.py --env HalfCheetah-v3 --encoder_pop_dim 10 --decoder_pop_dim 10 --encoder_var 0.15 --num_model 10 --root_dir [YOUR_DIR] --encode pop-det --decode last-mem
```
