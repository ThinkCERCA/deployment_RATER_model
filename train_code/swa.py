import glob
import torch
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder', default=None, action='store', required=True
    )

    parser.add_argument(
        '--last', default=None, type=int, required=False
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

os.makedirs('runs/swa', exist_ok=True)

def average_checkpoints(input_ckpts, output_ckpt):
    assert len(input_ckpts) >= 1
    data = torch.load(input_ckpts[0], map_location='cpu')['state_dict']
    swa_n = 1
    for ckpt in input_ckpts[1:]:
        new_data = torch.load(ckpt, map_location='cpu')['state_dict']
        swa_n += 1
        for k, v in new_data.items():
            if v.dtype != torch.float32:
                print(k)
            else:
                data[k] += (new_data[k] - data[k]) / swa_n

    torch.save(dict(state_dict=data), output_ckpt)

ckpts = sorted(glob.glob(opt.folder + 'weights/*.ckpt'))[-opt.last:]
print(ckpts)

save_ckpt = opt.folder.replace('runs/', 'runs/swa/')
if save_ckpt[-1] == '/':
    save_ckpt = save_ckpt[:-1]
save_ckpt += '.ckpt'

average_checkpoints(ckpts, save_ckpt)