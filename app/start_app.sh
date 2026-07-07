#!/usr/bin/env bash
# Required: dataset_face_strat_merged, dataset_lesion_merged
cd "$(dirname "$0")/.."          # repo 루트로 이동 (outputs/ 등 상대경로 기준)
python3 app/eval_precompute.py
streamlit run app/app.py
