#!/usr/bin/env python3
"""Compare WaveSim final fields, using the first run as the reference."""

import argparse
import math
from pathlib import Path
import struct
import sys


TOLERANCE = 0.01
FLOAT32 = struct.Struct("<f")
CHUNK_BYTES = 1024 * 1024


def read_shape(directory):
    path = directory / "meta.txt"
    fields = {}
    with path.open(encoding="utf-8") as metadata:
        for line_number, line in enumerate(metadata, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2 or parts[0] in fields:
                raise ValueError(f"{path}:{line_number}: expected a unique key and value")
            key, value = parts
            fields[key] = value

    try:
        shape = tuple(int(fields[axis]) for axis in ("nx", "ny", "nz"))
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: nx, ny, and nz must be integers") from error
    if min(shape) < 1:
        raise ValueError(f"{path}: nx, ny, and nz must be positive")
    return shape


def read_chunk(stream, size, expected_bytes):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"{stream.name}: expected {expected_bytes} bytes")
    return data


def compare_fields(reference, candidate, expected_bytes):
    reference_sq = diff_sq = max_abs_diff = 0.0
    with reference.open("rb") as ref_file, candidate.open("rb") as test_file:
        for offset in range(0, expected_bytes, CHUNK_BYTES):
            size = min(CHUNK_BYTES, expected_bytes - offset)
            ref_values = FLOAT32.iter_unpack(read_chunk(ref_file, size, expected_bytes))
            test_values = FLOAT32.iter_unpack(read_chunk(test_file, size, expected_bytes))
            for (ref,), (test,) in zip(ref_values, test_values):
                if not math.isfinite(ref):
                    raise ValueError(f"{reference}: contains NaN or infinity")
                if not math.isfinite(test):
                    raise ValueError(f"{candidate}: contains NaN or infinity")
                diff = ref - test
                reference_sq += ref * ref
                diff_sq += diff * diff
                max_abs_diff = max(max_abs_diff, abs(diff))

        for stream in (ref_file, test_file):
            if stream.read(1):
                raise ValueError(f"{stream.name}: expected {expected_bytes} bytes")

    if reference_sq == 0.0:
        raise ValueError(f"{reference}: reference L2 norm is zero")
    return math.sqrt(diff_sq) / math.sqrt(reference_sq), max_abs_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    args = parser.parse_args()

    try:
        ref_shape = read_shape(args.reference_dir)
        test_shape = read_shape(args.candidate_dir)
        if ref_shape != test_shape:
            raise ValueError(f"grid shape mismatch: {ref_shape} vs {test_shape}")
        nx, ny, nz = ref_shape
        relative_l2, max_abs_diff = compare_fields(
            args.reference_dir / "final.bin",
            args.candidate_dir / "final.bin",
            nx * ny * nz * FLOAT32.size,
        )
    except (OSError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1

    passed = relative_l2 < TOLERANCE
    status = "PASS" if passed else "FAIL"
    print(f"{status} relative_l2={relative_l2:.12g} max_abs_diff={max_abs_diff:.12g}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
