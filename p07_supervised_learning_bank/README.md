# Задача
На основе данных из банка определить клиента, который может уйти

# Описание проекта
Банк стал замечать, что его клиенты постепенно уходят. Этот процесс происходит каждый месяц и, хоть и незначительно, но ощутимо влияет на общую картину. Маркетологи в банке провели анализ и пришли к выводу, что сохранение существующих клиентов обходится дешевле, чем привлечение новых.
Теперь требуется разработать модель прогнозирования, которая поможет определить, уйдет ли клиент из банка в ближайшее время или нет. Для этого предоставлены исторические данные о поведении клиентов и об их расторжении договоров с банком.

# Выводы
После проведения анализа, выяснилось, что лучшей моделью в данном проекте является модель случайного леса. Она показала наивысшую F1-меру среди всех моделей, которыми мы проводили эксперименты. С другой стороны, логистическая регрессия показала наихудшие результаты и не достигла требуемого значения F1=0.59.

Наша модель позволяет предсказывать уход клиента из банка в 63% случаев, что на 13% превышает результаты константной модели. Перед применением балансировки классов, модель случайного леса достигла F1-меры равной 0.6146926536731634. Затем мы провели сравнение трех различных методов балансировки классов: взвешивание выборки, увеличение классов и уменьшение классов. Наилучший результат по F1-мере показал метод взвешивания классов. На валидационной выборке достигнута F1-мера 0.6578313253012048. Однако, на тестовой выборке F1-мера составила 0.6109785202863962, существенно уступая результату на валидационной выборке. Тем не менее, удалось преодолеть требуемый барьер в 0.59. После дообучения модели на валидационной выборке, метрика F1 увеличилась примерно на 0.02.

Метрика AUC-ROC также немного улучшилась после проведения балансировки классов и составила 0.86. В сравнении с константной моделью, чья метрика равна 0.5, наша модель показывает гораздо более высокие результаты, что свидетельствует о ее хорошей предсказательной способности.

# Стек технологий
Pandas, Python, Matplotlib, Scikit-learn

# Направление
Машинное обучение: классификация, подбор гиперпараметров, выбор модели МО

# Статус проекта
Завершен

