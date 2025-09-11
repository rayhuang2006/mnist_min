#!/usr/bin/env python3
"""
Export TFLite from an ONNX model.

Two-step pipeline:
  1) Convert ONNX -> TensorFlow SavedModel (via onnx2tf)
  2) Convert SavedModel -> TFLite (via tf.lite.TFLiteConverter)

Requirements (CPU ok):
  pip install onnx onnx2tf tensorflow

Usage:
  python export_tflite.py --onnx model.onnx --out model.tflite
Options:
  --quant int8 : (optional) post-training dynamic range quantization (int8 weights)
"""
import argparse, os, shutil, subprocess, sys

def run(cmd):
    print('>>', ' '.join(cmd))
    r = subprocess.run(cmd, check=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', type=str, default='model.onnx')
    ap.add_argument('--out', type=str, default='model.tflite')
    ap.add_argument('--tmp', type=str, default='_onnx2tf_out')
    ap.add_argument('--quant', type=str, choices=['none','int8'], default='none')
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f'{args.onnx} not found. Please export ONNX first.')

    # 1) ONNX -> TF (SavedModel) via onnx2tf CLI
    if os.path.exists(args.tmp):
        shutil.rmtree(args.tmp)
    run(['onnx2tf', '-i', args.onnx, '-o', args.tmp])

    # 2) SavedModel -> TFLite
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(args.tmp)
    if args.quant == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print(f'Saved TFLite to {args.out}')

if __name__ == '__main__':
    main()
