#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jax/Orbax Gemma Checkpoint to Raw Bytes Converter

This script converts a Jax/Orbax Gemma model checkpoint into a directory structure
containing raw byte representations of the model's parameter arrays, as well as their shape.

Usage:
  python convert_checkpoint.py <source_dir> [--target_dir <target_dir>]

Arguments:
  source_dir: Path to the directory containing the Jax/Orbax checkpoint.
  target_dir: (Optional) Path to the directory where the raw bytes will be saved.
              Defaults to '<source_dir>/raw/' if not provided.

It requires the following libraries installed, probably in a virtual environment (venv) or equivalent::

```
pip install jax "git+https://github.com/google-deepmind/gemma.git"
```
"""

import argparse
import os
import jax
from gemma import params as params_lib

def read_parameter(path):
    """
    Read model checkpoint parameters from path to directory.

    :param path: Path to directory holding the checkpoint to read the parameters from.
    :return: PyTree of jaxlib.xla_extension.ArrayImpl
    """
    path = os.path.expanduser(path)
    return params_lib.load_and_format_params(path)


def write_params(params, base_dir):
    """Write parameters to structured to directory: each file correspond to one array written as raw bytes."""
    base_dir = os.path.expanduser(base_dir)
    for path, array in flatten_params(params):
        base_file_path = os.path.join(base_dir, *path)

        # Create necessary directories
        os.makedirs(os.path.dirname(base_file_path), exist_ok=True)

        # Save array.
        with open(base_file_path+".raw", 'wb') as f:
            f.write(array.tobytes())

        # Save shape.
        with open(base_file_path+".shape", 'w') as f:
            f.write(serialize_shape(array))


def path_to_str_tuple(path):
    """Converts a PyTree path (tuple of jax.tree_util.DictKey) to a tuple of strings."""
    return [e.key for e in path]


def flatten_params(params):
    """Convert PyTree of arrays to a list of pairs of (path, array), where path is itself a tuple of strings."""
    list = []
    def append_to_list(path, value):
        list.append((path_to_str_tuple(path), value))

    jax.tree_util.tree_map_with_path(append_to_list,  params)
    return list


def serialize_shape(array):
    """Return an encoding of the given array's shape (including dtype)."""
    return ",".join([f"{str(array.dtype)}"]+[str(i) for i in array.shape])


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma Jax/Orbax checkpoint to raw bytes.")
    parser.add_argument("source_dir", help="Path to the source directory containing the checkpoint.")
    parser.add_argument("--target_dir", help="Path to the target directory where the raw bytes will be saved. Defaults to source_dir + 'raw/' if not provided.")

    args = parser.parse_args()

    source_dir = os.path.abspath(os.path.expanduser(args.source_dir))
    target_dir = os.path.abspath(os.path.expanduser(args.target_dir)) if args.target_dir else os.path.join(source_dir, "raw")

    print("Conversion from Jax/Orbax Gemma checkpoint to raw arrays/shapes:")
    print(f"\tSource directory: {source_dir}")
    print(f"\tTarget directory: {target_dir}")

    # We don't want to use GPU memory for this.
    jax.config.update('jax_platform_name', 'cpu')

    params = read_parameter(source_dir)
    write_params(params, target_dir)


if __name__ == "__main__":
    main()