# Omni-modal concept data, tier 1: label vocabularies + caption corpora (small), plus FSD50K audio and
# MSR-VTT videos (medium). Each item is independent; failures are logged and skipped.
# Usage on the pod: bash tools/pod/download_omni.sh > /workspace/logs/download_omni.log 2>&1
source /workspace/env.sh  # provides the `hf` CLI (venv on the container disk)
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
O=/workspace/data/omni
mkdir -p $O/audio $O/video
get() {  # get URL DEST
  echo "== $1"; mkdir -p "$(dirname "$2")"
  wget -q --tries=3 -O "$2" "$1" && echo "ok $(du -h "$2" | cut -f1) $2" || { echo "FAIL $1"; rm -f "$2"; }
}

# ---- audio: vocabularies ----
get https://raw.githubusercontent.com/audioset/ontology/master/ontology.json $O/audio/audioset/ontology.json
get http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv $O/audio/audioset/class_labels_indices.csv
# ---- audio: caption corpora (text only) ----
for s in train val test; do
  get https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/$s.csv $O/audio/audiocaps/$s.csv
done
for f in clotho_captions_development.csv clotho_captions_validation.csv clotho_captions_evaluation.csv; do
  get "https://zenodo.org/records/4783391/files/$f?download=1" $O/audio/clotho/$f
done
# ---- audio: clips ----
get https://github.com/karoldvl/ESC-50/archive/master.zip $O/audio/esc50/ESC-50-master.zip && \
  (cd $O/audio/esc50 && unzip -oq ESC-50-master.zip && rm ESC-50-master.zip && echo "ok esc50 unzipped")
echo "== FSD50K (HF Fhrozen/FSD50k, ~34.5GB, CC-BY-4.0)"
hf download Fhrozen/FSD50k --repo-type dataset --local-dir $O/audio/fsd50k > /dev/null 2>&1 && echo "ok fsd50k" || echo "FAIL fsd50k"

# ---- video: vocabularies ----
echo "== VGGSound csv (HF Loie/VGGSound)"
hf download Loie/VGGSound vggsound.csv README.md --repo-type dataset --local-dir $O/video/vggsound > /dev/null 2>&1 && echo "ok vggsound csv" || echo "FAIL vggsound csv"
get https://storage.googleapis.com/deepmind-media/Datasets/kinetics700_2020.tar.gz $O/video/kinetics700/kinetics700_2020.tar.gz && \
  (cd $O/video/kinetics700 && tar xzf kinetics700_2020.tar.gz && echo "ok kinetics700 annotations")
# ---- video: caption corpora ----
get https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json $O/video/vatex/vatex_training_v1.0.json
get https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json $O/video/vatex/vatex_validation_v1.0.json
echo "== MSR-VTT (HF friedrichor/MSR-VTT, ~2.3GB incl. videos)"
hf download friedrichor/MSR-VTT --repo-type dataset --local-dir $O/video/msrvtt > /dev/null 2>&1 && echo "ok msrvtt" || echo "FAIL msrvtt"

du -sh $O/audio/* $O/video/* 2>/dev/null
echo OMNI_DONE
