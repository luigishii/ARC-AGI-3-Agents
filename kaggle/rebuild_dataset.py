import os, subprocess, glob, json, shutil
STAGE = "/kaggle/working/gpt_oss_ds"
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE + "/offline_wheels", exist_ok=True)

# 1) baixa kernels 0.14.0 (EXISTE e funciona) + deps
r = subprocess.run(["pip", "download", "kernels==0.14.0", "huggingface_hub",
                    "-d", STAGE + "/offline_wheels"], capture_output=True, text=True)
print(r.stderr[-300:])
wheels = glob.glob(STAGE + "/offline_wheels/kernels-*.whl")
assert wheels, "FALHOU: nenhum wheel baixado -- NAO suba"
print("OK wheel:", [os.path.basename(x) for x in wheels],
      "| total wheels:", len(os.listdir(STAGE + "/offline_wheels")))

# 2) repo de kernels como ARQUIVOS REAIS (local_dir, sem symlinks -> sobrevive ao zip).
#    O LOCAL_KERNELS aponta pra ca; get_local_kernel le build/<variante>/metadata.json.
from huggingface_hub import snapshot_download
snapshot_download("kernels-community/triton_kernels",
                  local_dir=STAGE + "/triton_kernels_repo")
assert glob.glob(STAGE + "/triton_kernels_repo/build/*/metadata.json"), \
    "FALHOU: build/<variante>/metadata.json ausente no repo de kernels"
print("OK kernel repo:", glob.glob(STAGE + "/triton_kernels_repo/build/*"))

# 3) metadata (mesmo id -> nova versao do MESMO dataset)
user = os.environ.get("KAGGLE_USERNAME") or "luigiishii"
json.dump({"title": "gpt-oss-offline-kernels", "id": user + "/gpt-oss-offline-kernels",
           "licenses": [{"name": "CC0-1.0"}]}, open(STAGE + "/dataset-metadata.json", "w"))

# 4) publica nova versao
r = subprocess.run(["kaggle", "datasets", "version", "-p", STAGE, "-m", "kernels 0.14.0",
                    "--dir-mode", "zip"], capture_output=True, text=True)
print(r.stdout[-300:])
print(r.stderr[-300:])
