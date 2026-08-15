"""Length-prefixed pickle framing over a plain TCP socket, shared by
distill_teacher_server.py and distill_vlm_onpolicy_remote.py. Stdlib only --
both ends are the same user's two machines on a Thunderbolt bridge, not an
untrusted network, so pickle is fine here.
"""
from __future__ import annotations

import pickle
import struct

_LEN = struct.Struct(">Q")


def send_msg(sock, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_LEN.pack(len(payload)))
    sock.sendall(payload)


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (length,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    return pickle.loads(_recv_exact(sock, length))
