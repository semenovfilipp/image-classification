#!/usr/bin/env bash
# Все эксперименты по очереди; аргументы пробрасываются в scripts/train.py.
# Завершённые пропускаются, прерванные продолжаются с последней эпохи.
set -u

EXPERIMENTS=(
  resnet18_feature_extract
  resnet18_finetune
  efficientnet_b0_feature_extract
  efficientnet_b0_finetune
  alexnet_feature_extract
  alexnet_finetune
  resnet18_scratch
  efficientnet_b0_scratch
  alexnet_scratch
  vgg16_bn_feature_extract
  vgg16_bn_finetune
  vgg16_bn_scratch
)

failed=()
for exp in "${EXPERIMENTS[@]}"; do
  echo "================ ${exp} ================"
  if ! python scripts/train.py --config "configs/experiments/${exp}.yaml" "$@"; then
    echo "!!! ${exp} упал"
    failed+=("${exp}")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "Упавшие эксперименты: ${failed[*]}"
  exit 1
fi
echo "Все эксперименты завершены"
