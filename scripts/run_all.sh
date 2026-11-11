#!/usr/bin/env bash
# Запускает все эксперименты по очереди. Завершённые пропускаются, прерванные
# продолжаются с последней эпохи, поэтому после обрыва Colab просто запустите снова.
#
#   bash scripts/run_all.sh --output-root /content/drive/MyDrive/pets/outputs
#
# Все аргументы пробрасываются в scripts/train.py.
set -u

# порядок: сначала быстрые и самые информативные, тяжёлые VGG в конце
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
