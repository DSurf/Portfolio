# Контакты:
Александр Улиско.
* e-mail: [ulisko.av@gmail.com](mailto:ulisko.av@gmail.com)
* Telegram: [@vandegraff](https://t.me/vandegraff)

[Резюме на hh](https://hh.ru/resume/8e74db96ff0ca2b1500039ed1f4d7042615837)

# Проект для Repetit.ru:
Лучшая достигнутая AUC-ROC: 0.810602150754808

## 
Целевой признак получен из status_id
Дубликаты заявок помечены как 1 в целевом признаке target если status_id = 5, 6, 13, 15
К основной таблице orders были присоединены таблицы teachers_info и агрегированная таблица suitable_teachers.
Внутреннее объединение таблиц дает нам 833524 строк. Из которых 200366 принадлежат классу 1.
Добавляем синтезированные признаки.
Добавляем признаки, полученные с помощью Bertopic из колонки purpose. Обучение на gpu заняло 90 минут. Для получения ембендингов использована модель paraphrase-multilingual-mpnet-base-v2.
Обученная модель занимает 11gb, она доступна по ссылке https://drive.google.com/file/d/1F9KDQaVoN6pUch5X1zOtzRfUH8caCkyB/view?usp=drive_link.
Удаляем колонки, дающие утечку.
Подбираем параметры для Catboost с помощью greedsearch.
Обучаем модель Catboost и проверяем ее на тестовой выборке. AUC-ROC составила 0.81



