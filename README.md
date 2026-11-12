# Классификация пород кошек и собак (Oxford-IIIT Pet) на PyTorch

## Задача

Многоклассовая классификация изображений: **37 пород** кошек и собак из
[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

Цель — написать на чистом PyTorch даталоадеры, цикл обучения и подсчёт метрик, а затем
сравнить архитектуры из `torchvision` в трёх режимах:

| режим | что происходит |
|---|---|
| `scratch` | случайная инициализация, учится вся сеть |
| `feature_extract` | веса ImageNet, backbone заморожен, учится только новый последний слой |
| `finetune` | веса ImageNet, учится вся сеть (backbone с LR в 10 раз ниже, чем у головы) |

Архитектуры: **AlexNet, VGG16-BN, ResNet18, EfficientNet-B0**.

## Данные и EDA

Подробно: [`notebooks/eda.ipynb`](notebooks/eda.ipynb).

* 7349 картинок, 37 классов: 12 пород кошек (2371 картинка) и 25 пород собак (4978); классы сбалансированы — в train от 79 до 85 картинок на класс;
* официальный `test` (3669) оставлен тестом, `trainval` (3680) поделён стратифицированно на train/val 85/15;
* данных на класс мало, поэтому transfer learning должен сильно выигрывать у обучения с нуля;
* размеры картинок разные: медиана 500×375, большинство не больше 500px по большей стороне,
  соотношение сторон (ш/в) в основном от 0.75 до 1.46, есть и вертикальные, и горизонтальные кадры; животное обычно в центре;
* задача fine-grained: часть пород очень похожа (питбуль и стаффорд, британская и русская голубая).

Предобработка (`scripts/prepare_data.py`): картинки один раз уменьшаются до 256px по короткой стороне,
чтобы на 2 CPU в Colab декодирование JPEG не тормозило GPU.
Аугментации train: `RandomResizedCrop(224, scale=(0.5, 1))`, `HorizontalFlip`, `ColorJitter(0.2)`;
val/test: `Resize(256)` + `CenterCrop(224)`; нормализация по статистикам ImageNet.

## Обучение

* оптимизатор AdamW, weight decay 0.05, label smoothing 0.1, gradient clipping 1.0;
* LR: линейный warmup и косинусное затухание (шаг — батч);
* mixed precision (AMP) на GPU, batch size 64, seed 42;
* early stopping по val accuracy; в тест идёт лучший по val чекпоинт;
* scratch: до 40 эпох, feature_extract: 10, finetune: 12 (точные значения в `configs/experiments/`);
* метрики: accuracy и macro-F1 (`src/utils/metrics.py`), логи в TensorBoard (опционально W&B).

## Результаты

Все 12 запусков на Google Colab, Tesla T4, около 1 ч 52 мин чистого обучения.
Метрики посчитаны на официальном test (3669 картинок) по лучшему на val чекпоинту.

**Test accuracy** (в скобках macro-F1):

| модель | параметры | scratch | feature_extract | finetune | сек/эпоха |
|:--|--:|--:|--:|--:|--:|
| AlexNet | 57.2M | 0.377 (0.373) | 0.754 (0.753) | 0.781 (0.779) | 21–24 |
| VGG16-BN | 134.4M | 0.345 (0.330) | 0.910 (0.910) | **0.922** (0.921) | 27–42 |
| ResNet18 | 11.2M | **0.573** (0.569) | 0.895 (0.894) | 0.902 (0.900) | 19–25 |
| EfficientNet-B0 | 4.1M | 0.534 (0.527) | **0.933** (0.932) | 0.914 (0.913) | 20–24 |

Полная таблица (обучаемые параметры, число эпох, лучшая эпоха, val accuracy, время):
[`results/results.md`](results/results.md).

![кривые обучения](results/curves.png)
![test accuracy](results/test_accuracy.png)

### Выводы

1. **Transfer learning решает задачу, обучение с нуля — нет.** Pretrained-модели дают 0.75–0.93 test accuracy,
   с нуля 0.35–0.57; разница от +33 п.п. (ResNet18) до +57 п.п. (VGG16-BN). На ~85 картинок на класс
   сеть не может выучить признаки с нуля, а ImageNet уже содержит множество пород собак и кошек,
   поэтому его признаки почти напрямую подходят к задаче.
2. **Лучший результат — EfficientNet-B0 в режиме feature extraction: 0.933**, при этом у неё всего 4M параметров
   и обучалось лишь 50k (один линейный слой), обучение заняло 3.5 минуты. VGG16-BN finetune почти догоняет (0.922),
   но в 33 раза больше по параметрам и вдвое медленнее за эпоху.
3. **Feature extraction и finetune почти равны** (разница ≤ 2.6 п.п.). Finetune немного лучше у AlexNet, VGG и ResNet18,
   а у EfficientNet-B0 даже хуже: train accuracy доходит до ~0.99, val выходит на плато с 7-й эпохи, то есть
   при дообучении всей сети на маленьком датасете модель начинает переобучаться. Для такого объёма данных
   заморозка backbone — разумный выбор по умолчанию; для finetune стоит попробовать меньший LR backbone и сильнее аугментации.
4. **С нуля архитектура важнее размера.** ResNet18 (0.573) и EfficientNet-B0 (0.534) с residual-связями и BatchNorm
   учатся заметно быстрее, чем AlexNet (0.377) и VGG16-BN (0.345). У ResNet18 большой разрыв train/val (0.92 / 0.69):
   это переобучение. У VGG и AlexNet train ≈ val ≈ 0.4: это недообучение, за 40 эпох они не сошлись.
   Кривые scratch-моделей к 40-й эпохе ещё растут, так что с большим числом эпох результат был бы выше,
   но до pretrained-моделей всё равно далеко.
5. **AlexNet — худшая pretrained-модель** (0.75–0.78): её ImageNet-признаки заметно слабее более поздних архитектур.
6. **Val accuracy выше test на 1–7 п.п.** У нас по val выбирается лучшая эпоха, а val маленький (~550 картинок),
   поэтому оценка на val оптимистична. Итоговые выводы делаем по test.
7. **Ошибки лучшей модели — визуально похожие породы**
   ([`results/confusion_matrix_best.png`](results/confusion_matrix_best.png)): American Pit Bull Terrier ↔
   Staffordshire Bull Terrier (29 ошибок в обе стороны), Ragdoll → Birman (14), Egyptian Mau → Bengal (13),
   Siamese ↔ Birman, British Shorthair ↔ Russian Blue. Эти пары трудно различить и человеку.
8. **Скорость.** Время эпохи у AlexNet, ResNet18 и EfficientNet-B0 почти одинаковое (~20–25 с): упираемся
   в загрузку и аугментацию картинок на 2 CPU Colab, а не в GPU. Реально в вычисления упирается только VGG16 (~42 с).

## Структура репозитория

```
├── configs/
│   ├── base.yaml                 # общие параметры (данные, обучение, логирование)
│   └── experiments/              # 12 конфигов <model>_<mode>.yaml, наследуют base.yaml
├── notebooks/
│   ├── eda.ipynb                 # разведочный анализ
│   └── colab_train.ipynb         # запуск всех экспериментов в Google Colab
├── scripts/
│   ├── prepare_data.py           # скачивание, ресайз, разбиение train/val/test
│   ├── train.py                  # обучение одной модели по конфигу + тест
│   ├── run_all.sh                # все эксперименты подряд
│   ├── evaluate.py               # отчёт по классам, матрица ошибок
│   └── make_report.py            # сводная таблица и графики в results/
├── src/
│   ├── data/dataset.py           # Dataset, трансформации, DataLoader'ы
│   ├── models/factory.py         # модели torchvision, замена головы, заморозка, группы LR
│   ├── training/trainer.py       # train/val loop, AMP, early stopping, чекпоинты, resume
│   └── utils/                    # конфиги, seed, метрики, логгеры
├── results/                      # итоговые таблицы и графики (коммитятся, весов нет)
└── requirements.txt
```

## Как запустить

### Google Colab (рекомендуется, бесплатного T4 достаточно)

Откройте [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) в Colab, выберите среду T4 GPU,
запускайте ячейки по порядку. Данные, чекпоинты и логи сохраняются на Google Drive
(`MyDrive/oxford_pets`), поэтому после отключения Colab достаточно запустить ячейки снова: завершённые
эксперименты пропустятся, прерванный продолжится с последней эпохи.

Место на Drive: веса лучших моделей занимают около 2.5 ГБ (больше всего VGG16, ~0.5 ГБ на эксперимент).

### Локально (нужен GPU с CUDA)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# torch с CUDA: https://pytorch.org/get-started/locally/
pip install -r requirements.txt

python scripts/prepare_data.py                     # ~800 МБ, около 5 минут
python scripts/train.py --config configs/experiments/resnet18_finetune.yaml
bash scripts/run_all.sh                            # все 12 экспериментов
python scripts/make_report.py                      # таблица и графики в results/
python scripts/evaluate.py --config configs/experiments/resnet18_finetune.yaml
tensorboard --logdir outputs
```

Полезные опции `train.py`:

```bash
--output-root PATH       # куда писать чекпоинты/логи (по умолчанию outputs/)
-o train.epochs=5        # переопределить любой параметр конфига (можно несколько раз)
--limit-batches 3        # быстрый прогон для отладки
--force                  # перезапустить завершённый эксперимент с нуля
```

Новый эксперимент — новый YAML в `configs/experiments/` с `base: configs/base.yaml`
и переопределёнными полями; имя файла становится именем эксперимента.
