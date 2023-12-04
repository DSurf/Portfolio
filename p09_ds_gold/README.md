# Задача
Спрогнозировать концентрацию золота при проведении процесса очистки золота.

# Описание проекта
Наш проект в области Data Science направлен на создание модели машинного обучения для промышленной компании, специализирующейся на разработке решений для эффективной работы промышленных предприятий. Нашей целью является разработка модели, которая сможет предсказать коэффициент восстановления золота из золотосодержащей руды на основе данных, связанных с параметрами добычи и очистки.

Эта модель будет иметь важное значение для оптимизации производства, так как она позволит компании избегать запуска предприятий с невыгодными характеристиками. Ранее эта задача требовала значительных усилий и являлась сложной для оценки и прогнозирования. Однако с помощью нашей модели машинного обучения, основанной на анализе исторических данных и параметров добычи и очистки, компания сможет получить точные прогнозы по коэффициенту восстановления золота.

Наш подход позволит промышленным предприятиям принимать более обоснованные и информированные решения по оптимизации производства, снижая риски и повышая эффективность. Мы стремимся не только предоставить модель, но и реализовать ее в рабочую среду, чтобы промышленные компании могли использовать ее в своей повседневной деятельности.

Наш проект имеет потенциал принести значительное экономическое преимущество промышленным предприятиям, а также внести вклад в развитие области промышленного производства и применения аналитики данных.

# Выводы
В процессе разработки проекта  предпринимались попытки улучшить финальную метрику.  
Положительно повлияло на метрику следующее:

* Удаление артефактов из тренеровочной выборки.
* Правильный подбор гиперпараметров.
* Выбор наилучшей модели при крос-валидации.

Попытки, которые отрицательно повлияли на метрику следующее:

* Была попытка избавиться от выбросов. При обучении модели на данных без выбросов, итоговая sMAPE теста стала хуже.     
* Избавление от некоторых признаков черновой очистки для набора признаков для прогнозирования финального коэффициент восстановления золота.
* Удаление некоторых признаков, распределения которых отличались наибольшим образом между тренеровочной и тестовой выборками. 
* Попытались избавиться от мультиколлинеарности, но это немного ухудшило sMAPE, поэтому, оставляем все признаки для обучения.


Для обучения модели использовали 2 разных набора признаков

Выбрали 3 модели для обучения - RandomForestRegressor(), Ridge(), GradientBoostingRegressor()

При обучении моделей использована кросс-валидация и поиск гиперпараметров   

Лучший результат показала модель RandomForestRegressor() при использовании GridSearchCV

Её итоговое sMAPE для RandomForestRegressor на тренеровочной выборке составило -4.927190570429951, а на тестовой выборке 8.858568421147506  

Была обучена и проверена константная модель DummyRegressor. Итоговое sMAPE для DummyRegressor составило 9.83177803432217    

Таким образом, наши обученные модели предсказывают коэффициент восстановления золота из золотосодержащей руды точнее, чем константная модель.

# Стек технологий
Pandas, Python, Matplotlib, NumPy, Scikit-learn, EDA

# Направление
Машинное обучение, аналитика: анализ данных, регрессия, кастомные метрики

# Статус проекта
Завершен