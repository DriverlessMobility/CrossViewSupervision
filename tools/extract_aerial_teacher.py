#!/usr/bin/env python3
"""Extract aerial encoder/downsampler weights from a fusion checkpoint."""

import argparse
import os

import torch

MODULE_PREFIXES = ("aid.encoder", "aid.down")


def _map_key(key: str) -> str:
    key = key.replace('aid_encoder', 'aid.encoder')
    key = key.replace('aid_downsampler', 'aid.down')
    if 'aid.encoder.' in key and '.net.' not in key:
        key = key.replace('aid.encoder.', 'aid.encoder.net.')
    return key


def extract_subset(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    subset = {}
    for name, value in state.items():
        mapped = _map_key(name)
        if any(mapped.startswith(prefix + '.') for prefix in MODULE_PREFIXES):
            subset[mapped] = value
    if not subset:
        raise RuntimeError('No parameters matching AID encoder/downsampler were found.')
    return {'state_dict': subset}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract aerial teacher weights from a fusion checkpoint.')
    parser.add_argument(
        '--work-dir', required=True, help='Path to the fusion training work dir.')
    parser.add_argument(
        '--checkpoint', default='latest.pth',
        help='Checkpoint filename inside the work dir (default: latest.pth).')
    parser.add_argument(
        '--output', required=True,
        help='Destination path for the extracted teacher checkpoint.')
    args = parser.parse_args()

    ckpt_path = args.work_dir
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    teacher_state = extract_subset(ckpt_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(teacher_state, args.output)
    print(f'Saved aerial teacher checkpoint to {args.output}')


if __name__ == '__main__':
    main()
