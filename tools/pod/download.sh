# Models + data onto the persistent /workspace disk (run once).
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
set -x
hf download guidelabs/steerling-8b-instruct
hf download openai/clip-vit-large-patch14-336
hf download liuhaotian/LLaVA-Pretrain blip_laion_cc_sbu_558k.json images.zip --repo-type dataset --local-dir /workspace/data/LLaVA-Pretrain
cd /workspace/data && wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -oq annotations_trainval2017.zip -d coco
echo DL_DONE
