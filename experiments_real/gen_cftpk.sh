#!/bin/bash
# Fair-tab cold reruns with SAVE_TOPK — protocol symmetry with spatial ctpk members.
PY=/home/weizyuv/.conda/envs/timesfm311/bin/python
for RS in Center:Ce Negev:Ne Northwest:No; do
  R=${RS%%:*}; S=${RS##*:}
  for SEED in 111 222 333 444 555; do
    F=/home/weizyuv/slurm_real/cw_${S}_cftpk_s${SEED}.sbatch
    { echo '#!/bin/bash'
      echo '#SBATCH --chdir=/home/weizyuv'
      echo '#SBATCH --partition=main'
      echo '#SBATCH --gpus=rtx_4090:1'
      echo '#SBATCH --cpus-per-task=8'
      echo '#SBATCH --mem=64G'
      echo '#SBATCH --time=0-06:00:00'
      echo "#SBATCH --job-name=cw${S}cftpk${SEED}"
      echo "#SBATCH --output=/home/weizyuv/slurm_real/cw_${S}_cftpk_s${SEED}-%j.out"
      echo "#SBATCH --error=/home/weizyuv/slurm_real/cw_${S}_cftpk_s${SEED}-%j.out"
      echo 'module load anaconda'
      echo 'export PYTHONPATH=/home/weizyuv/artsrc'
      echo "export COLD=1 REGION=$R RUN_TAG=${S}_cftpk STATIONVEC=1 CACHE=1 DETERMINISTIC=1 WARMUP=2 K=1 SEED=$SEED EPOCHS=120 PATIENCE=30 SAVE_TOPK=5 CKPT_ROOT=/home/weizyuv/expreal/${R}_cold"
      echo "export DATASET_DIR=/home/weizyuv/stationvec_MIN_$R"
      echo "$PY -u -m pipeline.train"
    } > "$F"
    sbatch "$F"
  done
done
