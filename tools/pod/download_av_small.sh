# VALOR-32K (mteb test subset, parquet with video+audio) and AVE videos; CPU/network only.
source /workspace/env.sh
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
O=/workspace/data/omni/av
hf download mteb/VALOR-32K --repo-type dataset --local-dir $O/valor32k > /dev/null 2>&1 && echo "ok valor32k $(du -sh $O/valor32k | cut -f1)" || echo "FAIL valor32k"
hf download UnFaZeD07/AVE-Dataset --repo-type dataset --local-dir $O/ave > /dev/null 2>&1 && echo "ok ave download" || echo "FAIL ave"
cd $O/ave && unzip -q videos.zip && rm videos.zip && echo "ave: $(find . -name '*.mp4' | wc -l) mp4, $(du -sh . | cut -f1)"
echo AVSMALL_DONE
