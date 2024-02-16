# Контакты:
Александр Улиско
* Email: [ulisko.av@gmail.com](mailto:ulisko.av@gmail.com)
* Telegram: [@vandegraff](https://t.me/vandegraff)
* HH: [Резюме на hh](https://hh.ru/resume/8e74db96ff0ca2b1500039ed1f4d7042615837)

# Проект для Repetit.ru:

## **Шаги выполнения:**

1. **Получение Целевого Признака:**
   - Целевой признак равен 1 из если status_id = 5, 6, 13, 15.
   - Дубликаты заявок помечены как 1 в целевом признаке target, если хоть одна из дублирующих заявок была успешной.

2. **Объединение Таблиц:**
   - К основной таблице orders были присоединены таблицы teachers_info и агрегированная таблица suitable_teachers.

3. **Предобработка Данных:**
   - Внутреннее объединение таблиц дало 833524 строки, из которых 200366 принадлежат классу 1.
   - Добавлены синтезированные признаки.
   - Удалены колонки, вызывающие утечку.

4. **Использование Bertopic для кластеризации текста:**
   - Обучение Bertopic на gpu для столбца purpose заняло 90 минут, используя модель paraphrase-multilingual-mpnet-base-v2.
   - Обученная модель занимает 11 ГБ и доступна по ссылке: [Model Link](https://drive.google.com/file/d/1F9KDQaVoN6pUch5X1zOtzRfUH8caCkyB/view?usp=drive_link).
   - Добавлены признаки, полученные с помощью Bertopic из столбца purpose.

5. **Обучение и Тестирование Модели:**
   - Подобраны параметры для Catboost с помощью GridSearch.
   - Обучаем модель Catboost и проверяем ее на тестовой выборке.
   - **Достигнут AUC-ROC: 0.81** 🚀

Обучение происходило на 92 признаках:

|FIELD1|Feature                          |Importance            |
|------|---------------------------------|----------------------|
|89    |day                              |3.1235259363293166    |
|88    |month                            |3.078186476214065     |
|6     |pupil_category_new_id            |3.030437972867819     |
|68    |rating_for_admin_max             |2.629047574689017     |
|69    |rating_for_admin_mean            |2.568686989620132     |
|0     |subject_id                       |2.47625851838237      |
|86    |probabilities_purpose            |2.2616227151985027    |
|70    |display_days_min                 |2.2088932344138614    |
|60    |rating_mean                      |2.1600752099416596    |
|7     |lessons_per_week                 |2.1410475219174314    |
|75    |star_rating_mean                 |2.1252981744288566    |
|3     |home_metro_id                    |2.0038727503615172    |
|46    |lesson_duration_mean             |1.9317905261112622    |
|79    |order_number                     |1.9296690587433594    |
|1     |lesson_price                     |1.9196626010868836    |
|8     |minimal_price                    |1.8438619282843671    |
|43    |lesson_cost_mean                 |1.7005931383452557    |
|39    |age_in_years_max                 |1.6564055819792673    |
|90    |weekday                          |1.614853065017946     |
|80    |purpose_phrase_count             |1.5997807695296582    |
|36    |teaching_experience_in_years_max |1.5385086642670347    |
|71    |display_days_max                 |1.5044573118911728    |
|62    |effective_rating_max             |1.4906875679009806    |
|14    |teacher_age_from                 |1.471697473657206     |
|40    |age_in_years_mean                |1.4227810463509754    |
|37    |teaching_experience_in_years_mean|1.3875273399220027    |
|91    |hour                             |1.3541591313476748    |
|42    |lesson_cost_max                  |1.3500480021065773    |
|55    |sex_1_sum                        |1.304876451794715     |
|63    |effective_rating_mean            |1.2938460086850536    |
|85    |topics_purpose                   |1.2840742817167965    |
|76    |rating_for_users_yesterday_min   |1.2726091785097189    |
|87    |year                             |1.2474676807689562    |
|49    |teacher_count_sum                |1.246623675344564     |
|77    |rating_for_users_yesterday_max   |1.2205234660514315    |
|12    |lesson_place_new                 |1.203105922267793     |
|72    |display_days_mean                |1.1905262184968946    |
|9     |teacher_sex                      |1.1798532559801405    |
|65    |rating_for_users_max             |1.156561573403771     |
|5     |creator_id                       |1.1164671604140104    |
|38    |age_in_years_min                 |1.089379421586932     |
|56    |sex_2_sum                        |1.0867400519554369    |
|17    |no_teachers_available            |1.0760612344520848    |
|83    |successful_applications          |1.0574370257432153    |
|25    |is_external_lessons_sum          |1.0452782702312249    |
|34    |time_on_platform_in_years_mean   |1.03524731408329      |
|82    |previous_applications            |1.0118181474068921    |
|64    |rating_for_users_min             |0.9898204451630391    |
|18    |source_id                        |0.9855798569158395    |
|24    |is_home_lessons_sum              |0.9377881598843889    |
|10    |teacher_experience_from          |0.9300665532946745    |
|81    |add_info_phrase_count            |0.9189444736891035    |
|15    |teacher_age_to                   |0.9101210319561934    |
|47    |student_count_sum                |0.8221467379325582    |
|33    |time_on_platform_in_years_max    |0.8183303748535885    |
|84    |creator_type                     |0.8160586664065684    |
|50    |prof_count_sum                   |0.7995355437217878    |
|19    |original_order_id                |0.745613295925528     |
|51    |priv_count_sum                   |0.7334077367097053    |
|66    |rating_for_users_mean            |0.6989052043774835    |
|78    |rating_for_users_yesterday_mean  |0.6953712468296385    |
|30    |information_flag_sum             |0.6925578244862114    |
|28    |is_remote_lessons_sum            |0.667141129564311     |
|44    |lesson_duration_min              |0.6352792108402538    |
|35    |teaching_experience_in_years_min |0.6185971482544032    |
|23    |is_email_confirmed_sum           |0.5912167016317343    |
|67    |rating_for_admin_min             |0.568153957198443     |
|73    |star_rating_min                  |0.5045627819847327    |
|59    |rating_max                       |0.48590169766905694   |
|58    |rating_min                       |0.46971835484434854   |
|13    |pupil_knowledgelvl               |0.40703306884087953   |
|4     |planned_lesson_number            |0.4035185577778999    |
|27    |is_cell_phone_confirmed_sum      |0.40332758445927536   |
|61    |effective_rating_min             |0.40263925646094506   |
|32    |time_on_platform_in_years_min    |0.39220706670780897   |
|45    |lesson_duration_max              |0.372430163287291     |
|26    |is_pupils_needed_sum             |0.36823566642975486   |
|31    |photo_path_flag_sum              |0.3346351672647702    |
|48    |aspirant_count_sum               |0.33356527410204423   |
|21    |is_display_to_teachers           |0.3248008094488027    |
|41    |lesson_cost_min                  |0.27246921197023766   |
|53    |russian_level_id_mean            |0.23965053853665907   |
|29    |is_confirmed_sum                 |0.23909376110331496   |
|2     |lesson_duration                  |0.21925773263429607   |
|52    |native_count_sum                 |0.14815447111633462   |
|22    |teacher_id_count                 |0.12057589667558602   |
|20    |max_metro_distance               |0.1157706767877568    |
|16    |chosen_teachers_only             |0.10591913071886758   |
|11    |teacher_experience_to            |0.06738687091255596   |
|74    |star_rating_max                  |0.026722476601296626  |
|57    |sex_3_sum                        |0.025706058338101478  |
|54    |russian_level_id_max             |0.00014980589282553248|
