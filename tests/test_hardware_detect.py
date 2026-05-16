import builtins


def test_detect_hardware_does_not_crash_without_torch(monkeypatch):
    # Ensure the CUDA env hint doesn't force GPU detection.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch not installed")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import subprocess

    _orig_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd and "nvidia-smi" in (cmd[0] if isinstance(cmd, (list, tuple)) else str(cmd)):
            class _Result:
                returncode = 1

            return _Result()
        return _orig_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    from utils.hardware_detect import detect_hardware

    profile = detect_hardware()
    assert profile["has_gpu"] is False
    assert profile["profile_name"] in {"standard", "lightweight", "powerful"}

