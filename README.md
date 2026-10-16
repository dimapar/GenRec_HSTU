# HSTU Robustness on MovieLens 1M

Репозиторий содержит два этапа исследования устойчивости последовательной рекомендательной модели HSTU на MovieLens 1M. На первом этапе изучается влияние удаления и замены одного фильма в истории. На втором проверяется влияние маскирования просмотренных фильмов и временных меток, корректность удаления и чувствительность последних десяти позиций истории.

Работа основана на репозитории [GenRec_HSTU](https://github.com/tribz334/GenRec_HSTU), который воспроизводит модель из статьи [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152). В этой версии добавлены совместимые с Windows резервные операции PyTorch, просмотр предсказаний для одного пользователя, два этапа robustness-эксперимента, статистический анализ, графики и интерфейс вызова модели через готовые embeddings.

## Что реализовано

- обучение HSTU на MovieLens 1M и вычисление baseline-метрик;
- просмотр истории, target и Top-10 рекомендаций для одного test-пользователя;
- первый этап: удаление или случайная замена фильма в позициях `first`, `25%`, `50%`, `75%`, `last`;
- второй этап: сравнение динамической и фиксированной масок кандидатов, отдельное изменение ID фильма и времени, исследование последних десяти позиций;
- проверка удаления: согласованность длины, padding, временных меток, маски внимания и выбора последнего реального состояния;
- расчёт Top-10 overlap, Jaccard index, Jensen-Shannon divergence, изменения Top-1 и ранга target;
- усреднение повторов внутри пользователя, 95%-е доверительные интервалы по пользователям и парные сравнения;
- сохранение отдельных сравнений и сводных таблиц в CSV;
- графики SVG/PDF/PNG с числовыми подписями и HTML-подборки; для второго этапа сохранены 38 графиков;
- редактируемые записки в Markdown и их PDF-версии;
- вызов модели через готовые embeddings с сохранением графа дифференцирования; Якобиан пока не вычисляется.

## Установка

Требуется Python 3.10.

Эксперимент проверен в Windows с Python 3.10.3, PyTorch 2.14.0+cu132, CUDA 13.2 и NVIDIA GeForce RTX 5060. На обучение до уровня baseline ушло около 1 часа 20 минут.
### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```


Для работы на GPU должна быть установлена CUDA-версия PyTorch, совместимая с драйвером видеокарты.

Для запуска тестов установите дополнительные зависимости:

```powershell
python -m pip install -e ".[test]"
```

Для подготовки PNG-графиков второго этапа нужен Poppler: команда `pdftoppm` должна быть доступна через `PATH`. Скриптам графиков также можно передать `--pdftoppm PATH` с полным путём к исполняемому файлу. Пакеты `reportlab` и `markdown-it-py` для сборки PDF входят в зависимости проекта.

### Особенность Windows

В исходном репозитории используется `fbgemm-gpu`. Он предназначен прежде всего для Linux и в Windows не требуется. Если библиотека недоступна, код автоматически использует реализации операций на обычном PyTorch. Сообщение `Failed to import fbgemm_gpu. Falling back to Pytorch implementation` выводится в логе с уровнем `ERROR`, но в данном случае не останавливает выполнение: оно сообщает о переходе на резервную реализацию. В конфигурации HSTU также отключен `torch.compile`, что обеспечивает более предсказуемый запуск в Windows.

## Запуск первого этапа

Все команды выполняются из корня репозитория. Подготовка данных и обучение нужны перед первым запуском. Для второго этапа используется уже обученный checkpoint `epoch_097.ckpt`; повторно обучать модель не требуется.

### 1. Подготовить MovieLens 1M

```powershell
python src\generative_recommenders_pl\scripts\prepare_data.py data=ml-1m
```

Данные загружаются и преобразуются в формат последовательностей, используемый моделью.

### 2. Обучить HSTU

На GPU:

```powershell
python src\generative_recommenders_pl\scripts\train.py experiment=ml-1m-hstu trainer=gpu
```

На CPU:

```powershell
python src\generative_recommenders_pl\scripts\train.py experiment=ml-1m-hstu trainer=cpu
```

Логи и checkpoint сохраняются в каталоге:

```text
logs/train/runs/RUN_ID/
```

Далее нужно указать путь к выбранному checkpoint. В PowerShell удобно сохранить его в переменную:

```powershell
$CKPT = "logs\train\runs\RUN_ID\checkpoints\CHECKPOINT.ckpt"
```

`RUN_ID` и `CHECKPOINT.ckpt` следует заменить значениями из конкретного запуска.

### 3. Посмотреть предсказание для одного пользователя

```powershell
python src\generative_recommenders_pl\scripts\inspect_user.py "ckpt_path=$CKPT"
```

Скрипт выводит упорядоченную историю, удержанный target и Top-10 рекомендаций. По умолчанию используется первый test-пользователь, то есть `user_index=0`.

Чтобы выбрать пользователя по его порядковому индексу в test-наборе, добавьте параметр `+user_index`. Индексация начинается с нуля:

```powershell
python src\generative_recommenders_pl\scripts\inspect_user.py "ckpt_path=$CKPT" +user_index=10
```

Чтобы выбрать пользователя по его исходному идентификатору `user_id` из MovieLens, используйте:

```powershell
python src\generative_recommenders_pl\scripts\inspect_user.py "ckpt_path=$CKPT" +user_id=42
```

Следует передавать только один из этих параметров. Если указаны оба, `user_id` имеет приоритет над `user_index`.

### 4. Запустить robustness-эксперимент

```powershell
python src\generative_recommenders_pl\scripts\run_robustness.py "ckpt_path=$CKPT"
```

Большая команда с параметрами не нужна: основной протокол уже задан значениями по умолчанию.

| Параметр командной строки | По умолчанию | Назначение |
| --- | --- | --- |
| `ckpt_path=PATH` | обязательный | Путь к checkpoint модели |
| `+num_users=N` | `500` | Количество test-пользователей в эксперименте |
| `+user_offset=N` | `0` | Сколько первых test-пользователей пропустить |
| `+replacement_repeats=N` | `5` | Число случайных замен для каждой позиции |
| `+base_seed=N` | `42` | Базовый seed для воспроизводимых случайных замен |
| `+top_k=K` | `10` | Размер сравниваемого списка рекомендаций |
| `+device=DEVICE` | `cuda`, если доступна, иначе `cpu` | Устройство вычислений: `cuda` или `cpu` |
| `+progress_every=N` | `25` | Выводить прогресс после каждых `N` пользователей; `0` отключает промежуточный вывод |
| `output_file=PATH` | `outputs/robustness/stage1/ml1m_hstu_robustness.csv` | Путь к итоговому CSV или Parquet-файлу |

При необходимости отдельные параметры можно переопределить через Hydra. Например, быстрый пробный запуск на 50 пользователях:

```powershell
python src\generative_recommenders_pl\scripts\run_robustness.py "ckpt_path=$CKPT" +num_users=50
```

Параметры можно комбинировать. Например, запуск на 100 пользователях с двумя случайными заменами, `Top-20` и другим seed:

```powershell
python src\generative_recommenders_pl\scripts\run_robustness.py "ckpt_path=$CKPT" +num_users=100 +replacement_repeats=2 +top_k=20 +base_seed=123
```

Префикс `+` требуется для параметров, которых нет в исходной конфигурации Hydra. У `ckpt_path` и `output_file` этот префикс не используется.

### 5. Построить графики

```powershell
python src\generative_recommenders_pl\scripts\plot_robustness.py
```

Скрипт автоматически читает стандартный CSV и сохраняет графики, сводные таблицы и HTML-отчет в `outputs/robustness/stage1/plots/`.

## Результаты первого этапа

Для обученного checkpoint получены baseline-значения.

$$
\mathrm{HR@10}=0.2975
$$

$$
\mathrm{NDCG@10}=0.1692
$$

Возмущения ранних и средних позиций слабо меняют Top-10, тогда как изменение последнего элемента истории оказывает сильное влияние. При случайной замене последнего элемента средний Top-10 overlap снижается до `0.249`, а первая рекомендация меняется в `89.2%` случаев.

Подробное определение метрик и интерпретация результатов приведены в технической записке.

## Файлы первого этапа

| Артефакт | Путь |
| --- | --- |
| Checkpoint модели | `logs/train/runs/RUN_ID/checkpoints/CHECKPOINT.ckpt` |
| Метрики обучения и baseline | `logs/train/runs/RUN_ID/csv/version_0/metrics.csv` |
| Результаты для каждого пользователя | `outputs/robustness/stage1/ml1m_hstu_robustness.csv` |
| Сводка по позиции возмущения | `outputs/robustness/stage1/plots/robustness_summary_by_position.csv` |
| Сводка по длине истории | `outputs/robustness/stage1/plots/robustness_summary_by_history_length.csv` |
| Графики и таблицы | `outputs/robustness/stage1/plots/robustness_plots.html` |
| Техническая записка | `outputs/robustness/stage1/robustness_note.md` |
| Техническая записка в PDF | `outputs/robustness/stage1/robustness_note.pdf` |


## Формат результатов первого этапа

Основной структурированный результат эксперимента находится в `outputs/robustness/stage1/ml1m_hstu_robustness.csv`. Одна строка соответствует одному сравнению исходной и возмущенной истории для конкретного пользователя, позиции и повторения.

Для `delete` записывается одна строка на каждую позицию, поэтому получается $500 \cdot 5 = 2500$ строк. Для `replace_random` используются пять случайных замен: $500 \cdot 5 \cdot 5 = 12500$ строк. Всего файл содержит 15 000 строк и 27 столбцов.

Префикс `original_` обозначает результат на исходной истории, а `perturbed_` - результат после удаления или замены одного элемента.

| Столбец | Тип | Содержание |
| --- | --- | --- |
| `user_index` | целое | Индекс пользователя внутри test-набора, начиная с нуля |
| `user_id` | целое | Исходный идентификатор пользователя в MovieLens |
| `history_length` | целое | Число реальных элементов истории без padding |
| `target_id` | целое | ID удержанного правильного следующего фильма |
| `target_rating` | целое | Оценка, которую пользователь поставил target-фильму |
| `perturbation_type` | строка | Тип возмущения: `delete` или `replace_random` |
| `position_label` | строка | Относительная позиция: `first`, `p25`, `p50`, `p75` или `last` |
| `position_index` | целое | Точный индекс измененного элемента в реальной истории, начиная с нуля |
| `position_fraction` | вещественное | Относительное положение элемента в истории от `0.0` до `1.0` |
| `original_item_id` | целое | ID фильма, находившегося в выбранной позиции до возмущения |
| `perturbed_item_id` | целое или пусто | ID случайного фильма после замены; для `delete` поле пустое |
| `repetition` | целое | Номер повторения случайной замены, начиная с нуля; для `delete` равен `0` |
| `seed` | целое или пусто | Seed конкретной случайной замены; для `delete` поле пустое |
| `top_k` | целое | Размер сравниваемого списка рекомендаций |
| `topk_overlap` | вещественное | Доля общих фильмов в исходном и возмущенном Top-K |
| `jaccard` | вещественное | Размер пересечения Top-K, деленный на размер их объединения |
| `js_divergence` | вещественное | Jensen-Shannon divergence между полными распределениями вероятностей; в CSV хранится без умножения на 1000 |
| `original_top1_id` | целое | Первая рекомендация для исходной истории |
| `perturbed_top1_id` | целое | Первая рекомендация после возмущения |
| `top1_changed` | логическое | `True`, если первая рекомендация изменилась |
| `original_target_rank` | целое или пусто | Позиция target среди кандидатов до возмущения, начиная с единицы |
| `perturbed_target_rank` | целое или пусто | Позиция target среди кандидатов после возмущения, начиная с единицы |
| `target_rank_delta` | целое или пусто | `perturbed_target_rank - original_target_rank`; положительное значение означает ухудшение позиции target |
| `original_target_in_topk` | логическое | Входил ли target в исходный Top-K |
| `perturbed_target_in_topk` | логическое | Входит ли target в Top-K после возмущения |
| `original_topk_ids` | JSON-список | ID фильмов в исходном Top-K в порядке убывания score |
| `perturbed_topk_ids` | JSON-список | ID фильмов в Top-K после возмущения в порядке убывания score |

Ранги target могут быть пустыми, если фильм отсутствует среди допустимых кандидатов или замаскирован как уже просмотренный. В этом случае `target_rank_delta` также остается пустым. Поля со списками хранятся внутри CSV как JSON-массивы, например `[364, 48, 1688, ...]`.

Файл можно загрузить в pandas:

```python
import pandas as pd

results = pd.read_csv("outputs/robustness/stage1/ml1m_hstu_robustness.csv")
```

## Основные файлы эксперимента

```text
src/generative_recommenders_pl/scripts/
├── inspect_user.py
├── run_robustness.py
├── plot_robustness.py
├── controlled_robustness.py
├── run_controlled_robustness.py
├── audit_robustness.py
├── analyze_controlled_robustness.py
├── plot_controlled_robustness.py
├── plot_masking_robustness.py
├── plot_timestamp_robustness.py
├── plot_tail_sensitivity.py
├── refresh_stage2_plots.py
└── write_controlled_note.py

outputs/robustness/
├── stage1/                       # исходный эксперимент
│   ├── ml1m_hstu_robustness.csv
│   ├── robustness_note.md
│   ├── robustness_note.pdf
│   └── plots/
└── stage2/                       # контроль маски, времени и inference
    ├── controlled_results.csv
    ├── mask_only_results.csv
    ├── reference_reproduction.csv
    ├── manifest.json
    ├── inference_audit.json
    ├── analysis_validation.json
    ├── summary_legacy_positions.csv
    ├── summary_tail_positions.csv
    ├── controlled_note.md        # источник основной записки
    ├── controlled_note_draft.pdf # актуальная PDF-версия
    ├── timestamps_section.md
    ├── tail_sensitivity_section.md
    └── plots/
        ├── stage2_plots.html     # все 38 графиков
        ├── controlled_plots.html
        ├── masking/
        ├── timestamps/
        └── tail_sensitivity/
```

## Второй этап: контролируемая чувствительность

Второй этап проверяет, сохраняется ли сильная чувствительность к последнему фильму после контроля маски кандидатов и временных меток. Checkpoint `epoch_097.ckpt`, первые 500 test-пользователей, пять случайных замен для каждой позиции и базовый seed `42` сохранены. Архитектура, обученные параметры и логика модели не меняются. CSV первого этапа используется для проверки воспроизводимости.

Входная история задаётся идентификаторами фильмов, соответствующими временными метками и фактической длиной. Неиспользуемые позиции заполняются нулями. При подготовке входа время target помещается сразу после реальной истории; после удаления эта служебная временная метка переносится по новой длине. Перед предсказанием выбирается состояние последнего реального элемента.

Основная записка содержит три эксперимента:

1. **Маскирование просмотренных фильмов.** Сравниваются `dynamic` (маска каждой фактической истории) и `fixed_original` (маска исходной истории для обоих предсказаний). Маски кандидатов применяются к одним и тем же вычисленным оценкам модели. Дополнительно сохранена диагностика изменения только маски при исходных оценках.
2. **Влияние временных меток.** При фиксированной маске сравниваются замена ID с сохранением времени, удаление фильма вместе с его оценкой и временем, изменение только времени просмотра. Для времени используются шесть сдвигов: ±1 час, ±1 день и ±7 дней. Они ограничены соседними событиями, для последнего элемента верхняя граница - время target. Ограниченные и нулевые сдвиги включены в основные средние.
3. **Последние десять позиций.** При фиксированной маске сравниваются те же три изменения. `distance_from_end=0` означает последний фильм, `1` - предпоследний; на графиках позиции обозначены числами от 10 до 1. Относительные позиции первого этапа и последние десять не дублируются, если совпадают.

Повторы усредняются сначала внутри пользователя: пять случайных замен или шесть временных сдвигов. Для удаления получается одно значение на пользователя и позицию. Затем вычисляется среднее по 500 пользователям. Доверительные интервалы получены через 5000 bootstrap-выборок пользователей с возвращением; для сравнения последней и предпоследней позиций сначала вычисляется разность внутри пользователя. Границы интервалов отдельных сдвигов не усредняются.

Интервалы точечные и условные на фиксированной модели и выбранных возмущениях. Первые 500 пользователей не являются случайной выборкой из всей совокупности пользователей. На графиках показаны средние без отрезков доверительных интервалов; в разделе последних десяти позиций численные границы приведены в таблицах под графиками. В ячейке таблицы среднее выделено жирным, интервал записан следующей строкой.

Получено 164 928 основных сравнений и 41 232 сравнения только маски. Каталог кандидатов сохранён: 3883 фильма из `movies.csv`, включая фильмы без оценок в MovieLens 1M. JS в CSV хранится без масштабирования, в графиках и записке используется множитель 1000. Top-1 changed на графиках и в записке указан в процентах. Парные разности режимов в `paired_mask_differences_*.csv` имеют направление `dynamic - fixed_original`.

### Воспроизведение расчёта модели

Команды выполняются из корня репозитория в окружении проекта с подготовленными данными и исходным checkpoint. Сохранённые CSV уже содержат результаты полного запуска; для пересборки графиков и записки достаточно следующего подраздела.

```powershell
$CKPT = "logs/train/runs/baseline/checkpoints/epoch_097.ckpt"
$env:PYTHONPATH = "$PWD/src"

# Независимо воспроизвести старые 15 000 сравнений.
python src/generative_recommenders_pl/scripts/run_robustness.py "ckpt_path=$CKPT" output_file=outputs/robustness/stage2/reference_reproduction.csv hydra.run.dir=outputs/robustness/stage2/reproduction_hydra

# Контролируемый эксперимент; перед расчётом выполняется аудит inference.
python src/generative_recommenders_pl/scripts/run_controlled_robustness.py "ckpt_path=$CKPT" hydra.run.dir=outputs/robustness/stage2/controlled_hydra
```

Основной запуск занимает около 11 минут на указанном выше GPU. Скрипт не перезаписывает существующий `controlled_results.csv`: для нового запуска задайте другой каталог через `+stage2_output_dir=PATH`; явная перезапись включается через `+overwrite=true`. Для пробного расчёта можно использовать `+num_users=3 +stage2_output_dir=tmp/controlled_smoke`. Основная записка описывает полный зафиксированный протокол.

В `manifest.json` сохранены хеши checkpoint, подготовленных данных, каталога, входов каждого пользователя и исходного кода расчёта, фактическая конфигурация и окружение. `inference_audit.json` содержит результаты проверки удаления и интерфейса embeddings. `analysis_validation.json` проверяет seeds, замены, рекомендации старого протокола, совпадение масок при изменении только времени и отсутствие изменений рекомендаций при неизменных временных интервалах.

### Анализ сохранённых CSV и построение графиков

Повторный вызов модели не требуется. Порядок команд важен: специализированные графики используют сводки общего анализа.

```powershell
$env:PYTHONPATH = "$PWD/src"

# Общие средние, 95%-е интервалы, парные разности и проверка Stage 1.
python src/generative_recommenders_pl/scripts/analyze_controlled_robustness.py

# Общая подборка и сравнение динамической и фиксированной масок.
python src/generative_recommenders_pl/scripts/plot_controlled_robustness.py
python src/generative_recommenders_pl/scripts/plot_masking_robustness.py

# Три изменения истории: время усреднено по шести сдвигам внутри пользователя.
python src/generative_recommenders_pl/scripts/plot_timestamp_robustness.py

# Последние десять позиций: три графика и таблицы доверительных интервалов.
python src/generative_recommenders_pl/scripts/plot_tail_sensitivity.py

# Собрать PDF из отредактированного Markdown.
python src/generative_recommenders_pl/scripts/write_controlled_note.py
```

Для результатов в другом каталоге передайте `--input-dir PATH` анализатору и скриптам графиков. Основной Markdown создаётся и редактируется вручную: сборщик PDF ожидает его в выбранном каталоге либо принимает явный путь через `--input`.

Для обновления оформления уже сохранённого полного набора из 38 графиков используйте:

```powershell
python src/generative_recommenders_pl/scripts/refresh_stage2_plots.py
```

Скрипт читает готовые сводки, обновляет SVG/PDF/PNG и HTML-подборки, сохраняя имена файлов. Markdown, CSV, JSON и PDF самих записок не изменяются. После обновления изображений PDF основной записки нужно пересобрать отдельно командой `write_controlled_note.py`.

Если нужны только отдельные обновления по готовым сводкам, доступны `plot_timestamp_robustness.py --plots-only`, `plot_tail_sensitivity.py --render-only` и `plot_tail_sensitivity.py --tables-only`. Последний вариант обновляет только HTML-таблицы, сохраняя графики и статистику.

### Результаты последней позиции

| Возмущение | Маска | Top-10 overlap | JS × 1000 | Top-1 changed, % |
| --- | --- | ---: | ---: | ---: |
| Удаление фильма | Динамическая | 0.5112 | 0.3609 | 73.60 |
| Удаление фильма | Фиксированная | 0.5222 | 0.2285 | 70.60 |
| Замена фильма | Динамическая | 0.24928 | 1.6675 | 89.20 |
| Замена фильма | Фиксированная | 0.25056 | 1.4633 | 89.08 |
| Изменение только времени, среднее по шести сдвигам | Фиксированная | 0.87743 | 0.0531 | 18.50 |

При фиксированной маске замена предпоследнего фильма даёт overlap `0.7314`, последнего - `0.25056`. Парная разность «последний минус предпоследний» равна `-0.48084`, её 95%-й доверительный интервал - примерно `[-0.49776; -0.46396]`. Скачок чувствительности на последней позиции сохраняется после контроля маски.

Изменение только времени в выбранном наборе сдвигов даёт меньший средний эффект. Эти результаты относятся к сохранённой модели, группе пользователей и протоколу, включая ограниченные и нулевые временные сдвиги.

### Файлы второго этапа

Все пути в таблице указаны относительно `outputs/robustness/stage2/`.

| Артефакт | Путь                                                                         |
| --- |------------------------------------------------------------------------------|
| Основные сравнения | `controlled_results.csv`                                                     |
| Диагностика только маски | `mask_only_results.csv`                                                      |
| Воспроизведение первого этапа | `reference_reproduction.csv`                                                 |
| Протокол и проверки | `manifest.json`, `inference_audit.json`, `analysis_validation.json`          |
| Средние и интервалы для пяти позиций | `summary_legacy_positions.csv`                                               |
| Средние и интервалы для последних десяти | `summary_tail_positions.csv`                                                 |
| Парные разности режимов | `paired_mask_differences_legacy.csv`, `paired_mask_differences_tail.csv`     |
| Сравнение последней и соседних позиций | `tail_contrasts.csv`                                                         |
| Покрытие временных сдвигов | `timestamp_diagnostics_legacy.csv`, `timestamp_diagnostics_tail.csv`         |
| Условные результаты при изменившихся временных интервалах | `timestamp_changed_buckets_legacy.csv`, `timestamp_changed_buckets_tail.csv` |
| Все 38 графиков | `plots/stage2_plots.html`                                                    |
| Общая подборка эксперимента | `plots/controlled_plots.html`                                                |
| Динамическая и фиксированная маски | `plots/masking/masking_plots.html`                                           |
| Сравнение трёх изменений с усреднением времени | `plots/timestamps/averaged_impact_plots.html`                                |
| Средние для сравнения трёх изменений | `plots/timestamps/averaged_impact_summary.csv`                               |
| Последние десять позиций: графики и интервалы | `plots/tail_sensitivity/tail_sensitivity_plots.html`                         |
| Средние и интервалы для трёх изменений последних позиций | `plots/tail_sensitivity/tail_impact_summary.csv`                             |
| Парное сравнение последней и предпоследней позиций | `plots/tail_sensitivity/last_minus_penultimate.csv`                          |
| Техническая записка | `controlled_note.md`                                                         |
| PDF-версия технической записки | `controlled_note.pdf`                                                        |

### Вызов модели через готовые embeddings

Интерфейс `models/utils/inference.py::logits_from_embeddings` принимает входные векторы **до** позиционного препроцессора, фактические длины, timestamps и фиксированные векторы кандидатов. Для обычного позиционного препроцессора ID фильмов не нужны. Интерфейс возвращает неотфильтрованные оценки по каталогу и сохраняет граф автоматического дифференцирования. Для вспомогательных препроцессоров с признаками ID такой вызов явно не поддерживается.

```python
from generative_recommenders_pl.models.utils.inference import logits_from_embeddings

model.eval()
# E: [B, 211, 50], lengths: [B], timestamps: [B, 211].
# timestamps содержит время target в позиции lengths[b].
# candidate_embeddings_t: [50, 3883], нормализация как в исходном скрипте.
# Когда понадобятся производные, E должен требовать градиент, а вызов
# должен выполняться вне torch.no_grad() и torch.inference_mode().
logits = logits_from_embeddings(
    model, E, lengths, timestamps, candidate_embeddings_t
)
```

В проверках оценки через embeddings совпали с обычным вызовом по ID точно. Якобиан и другие производные на этом этапе не вычислялись.

### Проверки

```powershell
python -m pytest -q
```

В текущем окружении прошли все 34 теста. Проверяются независимая реконструкция после удаления, обновлённые длины и временная служебная позиция, padding, пустая история, прежние seeds и заменяющие фильмы, ограничение времени, временные интервалы, влияние одной маски, равенство двух интерфейсов и расчёт интервалов по пользователям вместо отдельных строк замен.
