python prepare_data.py

python train.py --name ema_32_awp_5e-6_025 --yaml hparams/ema_32_awp_5e-6_025.yml --no_wandb
python swa.py --folder runs/ema_32_awp_5e-6_025/ --last 30
python predict.py --ckpt runs/swa/ema_32_awp_5e-6_025.ckpt --test

python train.py --name ema_32_awp_5e-6 --yaml hparams/ema_32_awp_5e-6.yml --no_wandb
python swa.py --folder runs/ema_32_awp_5e-6/ --last 30
python predict.py --ckpt runs/swa/ema_32_awp_5e-6.ckpt --test

python train.py --name ema_32_awp_8e-6 --yaml hparams/ema_32_awp_8e-6.yml --no_wandb
python swa.py --folder runs/ema_32_awp_8e-6/ --last 30
python predict.py --ckpt runs/swa/ema_32_awp_8e-6.ckpt --test

python train.py --name xlarge_ema_32_5e-6_2 --yaml hparams/xlarge_ema_32_5e-6_2.yml --no_wandb
python swa.py --folder runs/xlarge_ema_32_5e-6_2/ --last 20
python predict.py --ckpt runs/swa/xlarge_ema_32_5e-6_2.ckpt --test --xlarge

python train.py --name xlarge --yaml hparams/xlarge.yml --no_wandb
python swa.py --folder runs/xlarge/ --last 20
python predict.py --ckpt runs/swa/xlarge.ckpt --test --xlarge

python ensemble.py
python ensemble.py --test