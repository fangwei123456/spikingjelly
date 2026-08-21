
# Deep Reinforcement Learning with Spiking Q-learning (Based on PTAN)

## Dependency

```shell
pip install gym==0.18.0
pip install gym[atari]
pip install atari-py==0.2.5
```

## Training

```bash
python train.py --cuda --game breakout --T 8 --dec_type max-mem --seed 123
```
