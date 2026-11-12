"""Generate a submission in a separate Python process with pinned library versions.

Our three submissions were computed on different computers with different versions of
scikit-learn, numpy and pandas, and the versions change a few percent of the predictions.
To regenerate each file as it was uploaded, its recipe is run with the versions it was made with:
they are installed with ``pip install --target`` into a separate folder, which is put first on
PYTHONPATH for that process only. The notebook's own environment is not touched.

If the pinned versions cannot be installed (for example without internet), the recipe is run with
the libraries of the current environment instead and a warning is printed.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(cmd, **kwargs):
    """Run a command and print its output line by line, so that it shows up in a notebook cell."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", **kwargs)
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def install_requirements(requirements, env_dir):
    reqs = [line.strip() for line in Path(requirements).read_text().splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("python==")]
    print(f"Installing {' '.join(reqs)}", flush=True)
    return _run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                 "--target", str(env_dir), *reqs]) == 0


def run_pinned(name, requirements, out_dir, env_dir, data_dir=None):
    """Run ``python -m pokefds.submissions <name>`` with the library versions in `requirements`."""
    env_dir, out_dir = Path(env_dir), Path(out_dir)
    if not env_dir.exists() and not install_requirements(REPO / requirements, env_dir):
        print("WARNING: could not install the pinned versions, using the current environment. "
              "The file may differ from the uploaded one on a few percent of the battles.", flush=True)

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    paths = [str(REPO)] + ([str(env_dir)] if env_dir.exists() else [])
    env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in [env.get("PYTHONPATH")] if p])
    versions = ('import sys, numpy, pandas, sklearn; print("Python", sys.version.split()[0], '
                '"| numpy", numpy.__version__, "| pandas", pandas.__version__, '
                '"| scikit-learn", sklearn.__version__)')
    _run([sys.executable, "-c", versions], env=env)
    cmd = [sys.executable, "-u", "-m", "pokefds.submissions", name, "--out", str(out_dir)]
    if data_dir:
        cmd += ["--data-dir", str(Path(data_dir).resolve())]
    if _run(cmd, env=env, cwd=str(REPO)) != 0:
        raise RuntimeError(f"submission {name} failed")
