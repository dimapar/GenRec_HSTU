# HSTU Robustness on MovieLens 1M

Репозиторий содержит эксперимент по устойчивости последовательной рекомендательной модели HSTU к локальным изменениям пользовательской истории. Исследуется, насколько меняются рекомендации, если удалить или случайно заменить ровно один просмотренный фильм в разных позициях истории.

Работа основана на репозитории [GenRec_HSTU](https://github.com/tribz334/GenRec_HSTU), который воспроизводит модель из статьи [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152). В этой версии добавлены совместимые с Windows резервные операции PyTorch, просмотр предсказаний для одного пользователя, robustness-эксперимент, метрики и построение графиков.

## Что реализовано

- обучение HSTU на MovieLens 1M и вычисление baseline-метрик;
- просмотр истории, target и Top-10 рекомендаций для одного test-пользователя;
- удаление или случайная замена одного элемента истории в позициях `first`, `25%`, `50%`, `75%`, `last`;
- расчет Top-10 overlap, Jaccard index, Jensen-Shannon divergence, изменения Top-1 и ранга target;
- сохранение результатов на уровне отдельных пользователей в CSV;
- построение трех обязательных графиков, сводных таблиц и HTML-отчета;
- техническая записка с описанием архитектуры, предобработки, возмущений и результатов.

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

### Особенность Windows

В исходном репозитории используется `fbgemm-gpu`. Он предназначен прежде всего для Linux и в Windows не требуется. Если библиотека недоступна, код автоматически использует реализации операций на обычном PyTorch. Сообщение `Failed to import fbgemm_gpu. Falling back to Pytorch implementation` выводится в логе с уровнем `ERROR`, но в данном случае не останавливает выполнение: оно сообщает о переходе на резервную реализацию. В конфигурации HSTU также отключен `torch.compile`, что обеспечивает более предсказуемый запуск в Windows.

## Запуск эксперимента

Все команды выполняются из корня репозитория.

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
| `output_file=PATH` | `outputs/robustness/ml1m_hstu_robustness.csv` | Путь к итоговому CSV или Parquet-файлу |

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

Скрипт автоматически читает стандартный CSV и сохраняет графики, сводные таблицы и HTML-отчет в `outputs/robustness/plots/`.

## Результаты

Для обученного checkpoint получены baseline-значения.

$$
\mathrm{HR@10}=0.2975
$$

$$
\mathrm{NDCG@10}=0.1692
$$

Возмущения ранних и средних позиций слабо меняют Top-10, тогда как изменение последнего элемента истории оказывает сильное влияние. При случайной замене последнего элемента средний Top-10 overlap снижается до `0.249`, а первая рекомендация меняется в `89.2%` случаев.

Подробное определение метрик и интерпретация результатов приведены в технической записке.

## Структура результатов

| Артефакт | Путь |
| --- | --- |
| Checkpoint модели | `logs/train/runs/RUN_ID/checkpoints/CHECKPOINT.ckpt` |
| Метрики обучения и baseline | `logs/train/runs/RUN_ID/csv/version_0/metrics.csv` |
| Результаты для каждого пользователя | `outputs/robustness/ml1m_hstu_robustness.csv` |
| Сводка по позиции возмущения | `outputs/robustness/plots/robustness_summary_by_position.csv` |
| Сводка по длине истории | `outputs/robustness/plots/robustness_summary_by_history_length.csv` |
| Графики и таблицы | `outputs/robustness/plots/robustness_plots.html` |
| Техническая записка | `outputs/robustness/robustness_note.md` |
| Техническая записка в PDF | `outputs/robustness/robustness_note.pdf` |


## Формат файла с robustness-результатами

Основной структурированный результат эксперимента находится в `outputs/robustness/ml1m_hstu_robustness.csv`. Одна строка соответствует одному сравнению исходной и возмущенной истории для конкретного пользователя, позиции и повторения.

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

results = pd.read_csv("outputs/robustness/ml1m_hstu_robustness.csv")
```

## Основные файлы эксперимента

```text
src/generative_recommenders_pl/scripts/
├── inspect_user.py
├── run_robustness.py
└── plot_robustness.py

outputs/robustness/
├── ml1m_hstu_robustness.csv
├── robustness_note.md
├── robustness_note.pdf
└── plots/
```
