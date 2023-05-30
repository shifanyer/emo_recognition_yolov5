# Импортируем библиотеки
import os
import cv2

# Задаем путь к датасету и классу fear
dataset_path = 'dataset'
fear_class = 'fear'

# Получаем список файлов в папке класса fear
files = os.listdir(os.path.join(dataset_path, fear_class))

# Цикл по файлам
for f in files:
    # Считываем изображение в оттенках серого
    img = cv2.imread(os.path.join(dataset_path, fear_class, f), cv2.IMREAD_GRAYSCALE)
    # Отражаем изображение по горизонтали с помощью функции flip
    img_flipped = cv2.flip(img, 1)
    # Сохраняем отраженное изображение в ту же папку с префиксом 'flipped_'
    cv2.imwrite(os.path.join(dataset_path, fear_class, 'flipped_' + f), img_flipped)
