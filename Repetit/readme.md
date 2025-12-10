# 🎯 Fraud Detection & Lead Scoring для EdTech-платформы

> ML-система выявления недобросовестных исполнителей и интеллектуальной приоритизации заявок

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-GPU-yellow.svg)](https://catboost.ai)
[![BERTopic](https://img.shields.io/badge/BERTopic-NLP-green.svg)](https://maartengr.github.io/BERTopic)

## 📋 Описание проекта

EdTech-платформа по подбору репетиторов передаёт контакты учеников исполнителям и получает комиссию с каждого проведённого занятия. Однако **~20% исполнителей** уклоняются от оплаты различными способами:
- Занижение реальной стоимости занятий
- Неоплата проведённых уроков
- Манипуляции с расписанием

**Задача:** Построить ML-модель для раннего выявления недобросовестных исполнителей и оптимизации распределения заявок.

## 🏗️ Архитектура решения

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   6 таблиц      │────▶│  Feature Engine  │────▶│    CatBoost     │
│   833K записей  │     │   92 признака    │     │   AUC-ROC: 0.81 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  Текстовые поля │────▶│    BERTopic      │
│  purpose/info   │     │  ~200 кластеров  │
└─────────────────┘     └──────────────────┘
```

## 📊 Данные

| Таблица | Описание |
|---------|----------|
| `orders` | Заявки на исполнителей |
| `teachers_info` | Профили исполнителей |
| `suitable_teachers` | Подходящие исполнители для заявки |
| `lessons` | История занятий |
| `lesson_course` | Курсы учеников |
| `teacher_prices` | Цены исполнителей |

**Итоговый датасет:** 833K записей после объединения таблиц

**Целевой признак:** `target = 1` если `status_id ∈ {5, 6, 13, 15}` (успешная оплата заявки)

**Баланс классов:** 200K положительных / 633K отрицательных (~24% / 76%)

## 🔧 Feature Engineering

### Категории признаков (92 всего):

**📅 Временные** (top по важности)
- `day`, `month`, `weekday`, `hour`, `year`
- `time_on_platform_in_years`

**👤 Профиль исполнителя**
- Возраст, опыт преподавания
- Рейтинги: `rating_for_users`, `rating_for_admin`, `star_rating`
- Флаги подтверждения: email, телефон

**💰 Ценовые**
- `lesson_price`, `minimal_price`, `lesson_cost`
- Агрегаты: min/max/mean по исполнителям

**📝 Семантические (BERTopic)**
- `topics_purpose` — тематический кластер заявки
- `probabilities_purpose` — уверенность в кластере
- `purpose_phrase_count` — количество тем в заявке

### Top-10 признаков по важности:

| # | Признак | Importance |
|---|---------|------------|
| 1 | `day` | 3.12 |
| 2 | `month` | 3.08 |
| 3 | `pupil_category_new_id` | 3.03 |
| 4 | `rating_for_admin_max` | 2.63 |
| 5 | `rating_for_admin_mean` | 2.57 |
| 6 | `subject_id` | 2.48 |
| 7 | `probabilities_purpose` | 2.26 |
| 8 | `display_days_min` | 2.21 |
| 9 | `rating_mean` | 2.16 |
| 10 | `lessons_per_week` | 2.14 |

## 🤖 NLP Pipeline (BERTopic)

```python
# Модель для эмбеддингов
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Снижение размерности
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# Кластеризация
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', 
                        cluster_selection_method='eom', prediction_data=True)

# BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=200
)
```

**Характеристики:**
- ⏱️ Время обучения: 90 минут (GPU)
- 💾 Размер модели: 11 GB
- 🎯 Выделено ~200 семантических кластеров

**Примеры кластеров:**
- `единый гос. экзамен`, `математика`, `подготовка`
- `английский`, `разговорный`, `бизнес`
- `python`, `программирование`, `алгоритмы`

## 📈 Результаты

| Метрика | Значение |
|---------|----------|
| **AUC-ROC** | **0.81** |

### Параметры CatBoost (GridSearch)

```python
model = CatBoostClassifier(
    learning_rate=0.145,
    depth=12,
    l2_leaf_reg=2,
    task_type="GPU",
    devices='0:1',
    random_seed=42
)
```

## 🚀 Запуск

### Требования

```bash
pip install pandas numpy catboost bertopic sentence-transformers umap-learn hdbscan matplotlib
```

### Структура проекта

```
repetit-fraud-detection/
├── data/                      # .feather файлы (не включены)
├── repetit_analysis.ipynb     # Основной notebook
└── README.md
```

### Использование

```python
# Jupyter Notebook
jupyter notebook repetit_analysis.ipynb
```

## 💡 Ключевые инсайты

1. **Временные паттерны критичны** — день, месяц и час подачи заявки в топ-10 по важности
2. **Административный рейтинг > пользовательского** — внутренние метрики платформы более предиктивны
3. **Семантика заявок работает** — BERTopic-признаки (`probabilities_purpose`) в топ-10
4. **Дубликаты заявок информативны** — `order_number` показывает историю перезаявок

## 🛠️ Технологический стек

- **ML:** CatBoost, Scikit-learn
- **NLP:** BERTopic, Sentence-Transformers, UMAP, HDBSCAN
- **Data:** Pandas, NumPy, Feather
- **Visualization:** Matplotlib
- **Hardware:** NVIDIA GPU (CUDA)

## 📝 Признаки мошенничества (из бизнес-анализа)

Выявленные паттерны недобросовестного поведения:
- Несоответствие цены в заявке и реальной
- 1 занятие на ученика (отношение к общему количеству)
- Разная стоимость с разными учениками
- Занятия в расписании без оплат
- Оплаты рядом по времени (самооплата)
- Редкие заходы в приложение
- Статус "договорились", но оплат нет >7 дней

## 📄 Лицензия

MIT
