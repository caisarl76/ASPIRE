"""Test-side decoder for the SONIC packed ZMQ wire format (mirrors the C++ header parser)."""

from __future__ import annotations

import json

import numpy as np

HEADER_SIZE = 1280
_DTYPES = {"u8": np.uint8, "i32": "<i4", "i64": "<i8", "f32": "<f4", "f64": "<f8"}


def decode_packed(message: bytes, topic: bytes) -> dict[str, np.ndarray]:
    assert message.startswith(topic), "topic prefix mismatch"
    body = message[len(topic) :]
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00"))
    offset = HEADER_SIZE
    fields: dict[str, np.ndarray] = {}
    for field in header["fields"]:
        dtype = np.dtype(_DTYPES[field["dtype"]])
        count = int(np.prod(field["shape"]))
        end = offset + count * dtype.itemsize
        fields[field["name"]] = np.frombuffer(body[offset:end], dtype=dtype).reshape(field["shape"])
        offset = end
    return fields
