"""Trusted Linux entrypoint: verify irreversible identity reduction before work."""

import ctypes
import json
import os
import sys


def prctl(option, value=0):
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.prctl
    call.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    call.restype = ctypes.c_int
    result = call(option, value, 0, 0, 0)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def identity():
    return dict(
        uids=list(os.getresuid()),
        gids=list(os.getresgid()),
        groups=os.getgroups(),
        no_new_privs=prctl(39),  # PR_GET_NO_NEW_PRIVS
    )


def verify(phase):
    observed = identity()
    if observed != dict(uids=[1000] * 3, gids=[1000] * 3, groups=[], no_new_privs=1):
        raise RuntimeError(f"worker identity check failed: {observed}")
    print(json.dumps(dict(kind="identity", phase=phase, **observed)), flush=True)


def run(phase, command):
    if sys.platform != "linux":
        raise RuntimeError("the privilege launcher requires Linux")
    if phase not in ("drop", "verify") or not command:
        raise ValueError("expected drop/verify followed by a worker argv")
    if phase == "drop":
        prctl(38, 1)  # PR_SET_NO_NEW_PRIVS; inherited across fork and exec.
        os.setgroups([])
        os.setresgid(1000, 1000, 1000)
        os.setresuid(1000, 1000, 1000)
        verify("before_exec")
        # A fresh interpreter must confirm the identity inherited through exec.
        argv = [sys.executable, "-I", "-B", os.path.abspath(__file__), "verify", *command]
        os.execv(sys.executable, argv)
    else:
        verify("after_exec")
        os.execvp(command[0], command)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
