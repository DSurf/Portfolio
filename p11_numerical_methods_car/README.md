# Задача
Разработка системы рекомендации стоимости автомобиля на основе его описания.

# Описание проекта
Сервис по продаже автомобилей с пробегом  разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. На основе исторические данные необходимо построить модель для определения стоимости автомобиля.

# Выводы
Провели предобработку исходного датасета, удалили аномалии и неинформативные признаки, заполнили пропуски.
Подготовили выборки для обучения, валидации и тестирования моделей LightGBM и Ridge.
Подобрали гиперпараметры с помощью кросс-валидации и получили следующие результаты:
LightGBM: обучение - 12.84 сек, RMSE = 1506.39, предсказание (валидация) - 427 мс, RMSE = 1489.43.
Ridge: обучение - 6.92 сек, RMSE = 2266.99, предсказание (валидация) - 208 мс, RMSE = 2232.33.
LightGBM имеет лучшую метрику, но Ridge обучается и предсказывает в 2 раза быстрее.
Рекомендую выбрать LightGBM из-за лучшей метрики.
Провели тестирование LightGBM, результаты: предсказание (тест) - 342 мс, RMSE = 1508.14.

# Стек технологий
Pandas, Python, lightgbm

# Направление
Машинное обучение: градиентный бустинг, регрессия

# Статус проекта
Завершен
