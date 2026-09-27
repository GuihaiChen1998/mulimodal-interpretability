# FSD50K from Zenodo (record 4060432; CC-BY 4.0; split zips). HF mirror downloads were rate-limited.
O=/workspace/data/omni/audio/fsd50k_zenodo; mkdir -p $O; cd $O
B=https://zenodo.org/records/4060432/files
for f in FSD50K.ground_truth.zip FSD50K.metadata.zip FSD50K.doc.zip \
         FSD50K.eval_audio.z01 FSD50K.eval_audio.zip \
         FSD50K.dev_audio.z01 FSD50K.dev_audio.z02 FSD50K.dev_audio.z03 FSD50K.dev_audio.z04 FSD50K.dev_audio.z05 FSD50K.dev_audio.zip; do
  [ -s $f ] && continue
  wget -q --tries=5 -O $f "$B/$f?download=1" && echo "ok $f $(du -h $f | cut -f1)" || { echo "FAIL $f"; rm -f $f; }
done
for f in FSD50K.ground_truth.zip FSD50K.metadata.zip FSD50K.doc.zip; do unzip -oq $f; done
echo "audio zips kept split; merge with: zip -s 0 FSD50K.dev_audio.zip --out dev.zip && unzip dev.zip"
echo FSD_DONE
