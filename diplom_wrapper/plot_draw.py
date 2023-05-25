# импортируем необходимые модули
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QMessageBox
from diplom_wrapper.main_wrapper import run_video_prediction


# создаем класс для нашего окна
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # задаем заголовок и размеры окна
        self.setWindowTitle("Сервис")
        self.resize(300, 100)
        # создаем три кнопки с разными надписями
        self.button1 = QPushButton("Обработать видео с\n веб-камеры")
        self.button2 = QPushButton("Обработать записанное видео")
        self.button3 = QPushButton("Обработать файл")
        # создаем горизонтальный лэйаут для размещения кнопок
        self.layout = QHBoxLayout()
        # добавляем кнопки в лэйаут
        self.layout.addWidget(self.button1)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.button3)
        # устанавливаем лэйаут для окна
        self.setLayout(self.layout)
        # связываем функции с сигналами clicked кнопок
        self.button1.clicked.connect(self.run_camera)
        self.button2.clicked.connect(self.run_video)
        self.button3.clicked.connect(self.handle_file)

    # определяем функции для обработки
    def run_camera(self):
        run_video_prediction()

    def run_video(self):
        run_video_prediction()

    def handle_file(self):
        run_video_prediction()


# создаем экземпляр приложения
app = QApplication([])
# создаем экземпляр окна
window = Window()
# показываем окно на экране
window.show()
# запускаем цикл обработки событий приложения
app.exec_()
