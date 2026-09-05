import os, subprocess, glob, json, shutil
STAGE = "/kaggle/working/gpt_oss_ds"
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE + "/offline_wheels", exist_ok=True)
subprocess.run(["pip", "download", "kernels==0.14.0", "huggingface_hub", "-d", STAGE + "/offline_wheels"], capture_output=True, text=True)
assert glob.glob(STAGE + "/offline_wheels/kernels-*.whl"), "FALHOU wheels"
# triton: o kernel MXFP4 (triton_kernels) importa `triton`; a imagem Kaggle pode nao
# trazer. Baixa a versao que o torch instalado exige (ex.: 'triton==3.5.0').
import importlib.metadata as _md
_spec = next((r.split(";")[0].strip() for r in (_md.requires("torch") or [])
              if r.strip().startswith("triton")), "triton")
print("triton spec:", _spec)
subprocess.run(["pip", "download", _spec, "--no-deps", "-d", STAGE + "/offline_wheels"], capture_output=True, text=True)
assert glob.glob(STAGE + "/offline_wheels/triton-*.whl"), "FALHOU wheel triton"
from huggingface_hub import snapshot_download
snapshot_download("kernels-community/triton_kernels", local_dir=STAGE + "/triton_kernels_repo")
PKG = STAGE + "/triton_kernels_repo/build/torch-universal/triton_kernels/__init__.py"
assert os.path.isfile(PKG), "FALHOU: pacote triton_kernels ausente"
print("OK kernel pkg:", os.path.dirname(PKG))
user = os.environ.get("KAGGLE_USERNAME") or "luigiishii"
json.dump({"title": "gpt-oss-offline-kernels", "id": user + "/gpt-oss-offline-kernels", "licenses": [{"name": "CC0-1.0"}]}, open(STAGE + "/dataset-metadata.json", "w"))
r = subprocess.run(["kaggle", "datasets", "version", "-p", STAGE, "-m", "kernel pkg v9 + triton wheel", "--dir-mode", "zip"], capture_output=True, text=True)
print(r.stdout[-300:])
print(r.stderr[-300:])
