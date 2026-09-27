# One AVoCaDO training-set video shard (~21.4GB, ~4K clips with audio) -> extracted mp4s; the tarball is deleted after extraction.
source /workspace/env.sh
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
P=${1:-000}; D=/workspace/data/omni/av/avocado
hf download AVoCaDO-Captioner/training_set videos.part-$P.tar.gz --repo-type dataset --local-dir $D > /dev/null 2>&1 && echo "downloaded part $P"
cd $D && tar xzf videos.part-$P.tar.gz && rm videos.part-$P.tar.gz && echo "extracted: $(ls videos | wc -l) mp4 files, $(du -sh videos | cut -f1)"
echo SHARD_DONE
