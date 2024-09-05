import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar, QAction, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QCheckBox, QGroupBox, QRadioButton, QTextEdit, QPushButton, QDialog, QLineEdit,QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
from scipy.special import comb
from scipy.interpolate import interp1d 

from pathlib import Path

class Filter:
    # Initialize the buffer at the class level
    buffer = [None, None, None]

    def __init__(self, name, kernel, scaling):
        self.name = name
        self.kernel = kernel
        self.scaling = scaling

    def apply(self, image):
        if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
            if self.name == "Median Filter" and self.kernel is None:
                # Apply the median filter
                filtered_image = self.median(image)
            else:
                # Apply a convolution filter if a kernel is defined
                filtered_image = cv2.filter2D(image, -1, np.array(self.kernel))

                # Scale the image if scaling is defined
                if self.scaling is not None:
                    filtered_image = filtered_image 

            return filtered_image
        else:
            return None

    def serialize(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def median(self, image):
        # Apply median filter using SciPy's ndimage.median_filter
        if image is not None:
            filtered_image = cv2.medianBlur(image, 21)
            return filtered_image
        else:
            return None

def create_default_filters():
    # Папка, где будут храниться файлы фильтров

    parent = Path(__file__)
    filters_folder = parent + '/Filters'

    # Убедитесь, что папка "Filters" существует
    if not os.path.exists(filters_folder):
        os.makedirs(filters_folder)

    # Define the filters
    box_blur_filter = Filter("Box Blur", np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float) / 9, scaling=None)
    gaussian_blur_filter = Filter("Gaussian Blur", np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16, scaling=None)
    sharpen_filter = Filter("Sharpen", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float), scaling=None)
    edge_detection_filter = Filter("Edge Detection", np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float), scaling=None)
    emboss_filter = Filter("Emboss", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float), scaling=None)
    median_filter = Filter("Median Filter", None, scaling=None)  # You can set the kernel in the apply method

    # Serialize and save the filters
    box_blur_filter.serialize(os.path.join(filters_folder, 'box_blur.filter'))
    gaussian_blur_filter.serialize(os.path.join(filters_folder, 'gaussian_blur.filter'))
    sharpen_filter.serialize(os.path.join(filters_folder, 'sharpen.filter'))
    edge_detection_filter.serialize(os.path.join(filters_folder, 'edge_detection.filter'))
    emboss_filter.serialize(os.path.join(filters_folder, 'emboss.filter'))
    median_filter.serialize(os.path.join(filters_folder, 'median_filter.filter'))

    print('Default filters saved to the "Filters" folder.')

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.filters = []  # Список фильтров

        self.initUI()
        self.view = QGraphicsView()

    def initUI(self):
        self.setWindowTitle('Image Processing App')
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.createMenus()
        self.createImageBoxes()


        

        # Create a label and text field for custom filter matrix input
        self.custom_filter_apply_button = QPushButton('Add Custom Filter', self)
        self.custom_filter_apply_button.clicked.connect(self.show_custom_filter_dialog)

        self.plot_bezier_button = QPushButton('Bezier Correction', self)
        self.plot_bezier_button.clicked.connect(self.show_bezier_curve)

        #для кнопок
        channels_and_space_layout = QHBoxLayout()
        self.layout.addLayout(channels_and_space_layout)
        channels_and_space_layout.addWidget(self.plot_bezier_button)

        # Add the label, text field, and button to the layout
        channels_and_space_layout.addWidget(self.custom_filter_apply_button)

    def createMenus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        open_first_image_action = QAction('Open Image', self)
        open_first_image_action.triggered.connect(self.openFirstImage)
        file_menu.addAction(open_first_image_action)

        save_result_action = QAction('Save Result', self)
        save_result_action.triggered.connect(self.saveResult)
        file_menu.addAction(save_result_action)

        # Filters menu
        self.filters_menu = menubar.addMenu('Filters')
        self.createFilterActions(self.filters_menu)

        # Info menu
        info_menu = menubar.addMenu('Info')

        info_action = QAction('Show Info', self)
        info_action.triggered.connect(self.show_info_dialog)
        info_menu.addAction(info_action)
    
    def show_info_dialog(self):
        info_text = """This is the instruction manual for the application.
The following functions are available to the user:
1.Open Image. This is done by clicking on the File button on the top panel of the application and clicking on Open File.
2.Save Image. To do this, click on the File button on the top panel of the application and click on the Save File button.
3.Use the built-in filters. To do this, click on the Filters button on the top panel of the application and select the desired filter.
4.Adding your own filters. To do this, click on the Add Custom Filter button on the bottom panel of the application and enter the necessary data.
5.Bezier curve correction. Click the Bezier Correction button on the bottom panel of the application and correct the image."""
        message_box = QMessageBox(self)
        message_box.setWindowTitle('Information')
        message_box.setText(info_text)
        message_box.addButton(QMessageBox.Ok)
        message_box.exec()

    def createImageBoxes(self):
        # Create QGraphicsView widgets for image display
        self.source_image_view = QGraphicsView(self)
        self.result_image_view = QGraphicsView(self)

        # Create QGraphicsScenes to hold the images
        self.source_scene = QGraphicsScene()
        self.result_scene = QGraphicsScene()

        # Set scenes for the QGraphicsViews
        self.source_image_view.setScene(self.source_scene)
        self.result_image_view.setScene(self.result_scene)

        # Add QGraphicsViews to layout
        image_box_layout = QHBoxLayout()
        image_box_layout.addWidget(self.source_image_view)
        image_box_layout.addWidget(self.result_image_view)
        
        self.layout.addLayout(image_box_layout)

    def createFilterActions(self, menu):
        menu.clear()
        # Создайте действия для фильтров на основе файлов в папке с фильтрами
        parent = Path(__file__).parent
        

        filter_folder = os.path.join(parent, 'Filters')  # Папка, где хранятся файлы фильтров
        filter_files = [f for f in os.listdir(filter_folder) if f.endswith('.filter')]

        for filter_file in filter_files:
            filter_name = os.path.splitext(filter_file)[0]
            filter_action = QAction(filter_name, self)
            filter_action.triggered.connect(lambda _, name=filter_name: self.applyFilterByName(name))
            menu.addAction(filter_action)

    def applyFilterByName(self, filter_name):
        filter_file = os.path.join('Filters', f'{filter_name}.filter')
        if os.path.isfile(filter_file):
            filter_obj = Filter.deserialize(filter_file)
            if hasattr(self, 'source_image'):
                filtered_image = filter_obj.apply(self.source_image)
                cv2.imwrite(self.file_path, filtered_image)
                self.display_cv2_image(self.file_path, self.result_scene)
                self.filtered_image = filtered_image  # Сохраняем результат в атрибуте filtered_image

                # Добавляем фильтр в список фильтров
                self.filters.append(filter_obj)

    def saveResult(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Result', '', 'Bitmap Images (*.bmp)')

        if file_path:
            if hasattr(self, 'filtered_image'):
                cv2.imwrite(file_path, self.filtered_image)
                print(f"Result saved to: {file_path}")
            else:
                # If no filter has been applied, save the original image
                cv2.imwrite(file_path, self.source_image)
                print(f"Original image saved to: {file_path}")

    def openFirstImage(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open First Image', '', 'Images (*.bmp *.jpg *.jpeg *.png)')

        if file_path:
            # Use cv2 to read the image
            self.source_image = cv2.imread(file_path)

            # Display the image in the QGraphicsView
            self.display_cv2_image(file_path, self.source_scene)
            if os.path.isfile(file_path):
                folder_path = os.path.dirname(file_path)
                output_path = os.path.join(folder_path, "filtered_image.bmp")
                self.file_path = output_path
                cv2.imwrite(output_path, self.source_image)

    def display_cv2_image(self, cv2_image, scene):
        if cv2_image is not None:
            pixmap = QPixmap(cv2_image)
            item = QGraphicsPixmapItem(pixmap)
            scene.clear()
            self.source_image_view.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            self.result_image_view.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            scene.addItem(item)
            self.show()

    def show_custom_filter_dialog(self):
        dialog = CustomFilterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name, kernel, scaling = dialog.get_custom_filter_params()

            if name and kernel is not None and scaling is not None:
                custom_filter = Filter(name, np.array(kernel, dtype=float)/scaling, None)
                custom_filter_file = os.path.join('Filters', f'{name.lower()}.filter')
                custom_filter.serialize(custom_filter_file)
                self.createFilterActions(self.filters_menu)  # Update the filters menu

                # Добавляем фильтр в список фильтров
                self.filters.append(custom_filter)

    def show_bezier_curve(self):
        plotter = BezierCurvePlotter(self)
        plotter.display_plot()
    

    def update_brightness(self, curve):
        # Получите координаты кривой Безье
        x_points, y_points = curve.get_bezier_curve_values()

        # Разделите исходное изображение на каналы R, G и B
        #b, g, r = cv2.split(self.source_image)

        # Строим интерполяционный полином
        #poly = np.polyfit(x_points, y_points, deg=len(4)-1)

        # Создаем функцию интерполяции
        f_interpolate = interp1d(x_points, y_points, kind='linear', fill_value='extrapolate')
        corrected_image=f_interpolate(self.source_image/255.0)

        # Вычисляем значения интерполяционного полинома для всех x координат пикселей
        #y_interpolated = f_interpolate(range(r.shape[1]))

        # Примените коррекцию яркости к каждому каналу
        #r_corrected = np.clip((r * y_interpolated/255.0), 0, 255).astype(np.uint8)
        #g_corrected = np.clip((g * y_interpolated/255.0 ) , 0, 255).astype(np.uint8)
        #b_corrected = np.clip((b * y_interpolated/255.0 ), 0, 255).astype(np.uint8)

        # Объедините каналы R, G и B, чтобы получить скорректированное изображение
        #corrected_image = cv2.merge((b_corrected, g_corrected, r_corrected))

        cv2.imwrite(self.file_path, corrected_image*255.0)
        self.display_cv2_image(self.file_path, self.result_scene)




    '''def update_brightness(self, curve):
        if self.source_image is not None:
            # Конвертируем изображение OpenCV в изображение Pillow
            image_pillow = Image.fromarray(cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB))

            # Нормализуем значения яркости на основе кривой Bezier
            normalized_brightness = [int(val*255) for val in curve.control_points[:, 1]]
            normalized_brightness = [val / 255.0 for val in normalized_brightness]


            # Вычисляем среднее значение яркости
            average_brightness = sum(normalized_brightness) / len(normalized_brightness)

            # Коррекция яркости
            enhancer = ImageEnhance.Brightness(image_pillow)
            enhanced_image_pillow = enhancer.enhance(average_brightness)

            # Конвертируем изображение Pillow обратно в OpenCV
            enhanced_image_cv2 = cv2.cvtColor(np.array(enhanced_image_pillow), cv2.COLOR_RGB2BGR)

            # Сохраняем откорректированное изображение
            cv2.imwrite(self.file_path, enhanced_image_cv2)
            self.display_cv2_image(self.file_path, self.result_scene)'''



class BezierCurvePlotter:
    def __init__(self, parent):
        self.control_points = np.array([
            [0, 0],      # Начальная точка
            [0.3, 0.3],  # Промежуточная точка 1
            [0.6, 0.6],  # Промежуточная точка 2
            [1, 1]       # Конечная точка
        ], dtype=float)
        self.t = np.linspace(0, 1, 1000)
        self.parent = parent
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title('Bezier Curve')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True)
        self.fig.canvas.manager.set_window_title('Bezier Curve')
        self.fig.canvas.manager.toolbar.setVisible(False)

        self.draggable_points = [DraggablePoint(self.ax, self.control_points[i, 0], self.control_points[i, 1], self, i) for i in range(1, len(self.control_points) - 1)]

        self.plot_curve()
        self.plot_control_points()

    def plot_curve(self):
        x_points = []
        y_points = []

        for i in self.t:
            b = np.zeros_like(self.control_points[0])
            n = len(self.control_points) - 1
            for j in range(n + 1):
                b += self.control_points[j] * (comb(n, j) * (1 - i)**(n - j) * i**j)
            x_points.append(b[0])
            y_points.append(b[1])

        if hasattr(self, 'curve_line'):
            self.curve_line.set_xdata(x_points)
            self.curve_line.set_ydata(y_points)
        else:
            self.curve_line, = self.ax.plot(x_points, y_points, '-r', label='Bezier Curve')
            self.ax.legend()

    def plot_control_points(self):
        if hasattr(self, 'control_point_scatter'):
            self.control_point_scatter.remove()
        self.control_point_scatter = self.ax.scatter(self.control_points[:, 0], self.control_points[:, 1], c='b', marker='o', label='Control Points')
        self.ax.legend()

    def display_plot(self):
        plt.show()

    def update_control_point(self, x, y, point_index):
        self.control_points[point_index] = [x, y]
        self.plot_curve()
        self.plot_control_points()
        plt.draw()
        self.parent.update_brightness(self)

    def get_bezier_curve_values(self):
        x_points = []
        y_points = []

        for t in self.t:
            b = np.zeros_like(self.control_points[0])
            n = len(self.control_points) - 1
            for j in range(n + 1):
                b += self.control_points[j] * (comb(n, j) * (1 - t)**(n - j) * t**j)
            x_points.append(b[0])
            y_points.append(b[1])

        return x_points, y_points

class DraggablePoint:
    def __init__(self, ax, x=0, y=0, plotter=None, point_index=0):
        self.ax = ax
        self.plotter = plotter
        self.point_index = point_index
        self.point, = ax.plot(x, y, 'bo', markersize=10, picker=5)
        self.x = x
        self.y = y
        self.is_pressed = False
        if point_index == 0 or point_index == len(self.plotter.control_points) - 1:
            self.point.set_visible(False)  # Скрываем точки в начале и в конце
        self.point.figure.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.point.figure.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)


    def on_button_press(self, event):
        if event.inaxes != self.point.axes:
            return
        contains, _ = self.point.contains(event)
        if not contains:
            return
        self.is_pressed = True

    def on_button_release(self, event):
        self.is_pressed = False

    def on_mouse_move(self, event):
        if not self.is_pressed or self.point is None:
            return
        self.x, self.y = event.xdata, event.ydata
        self.point.set_data(self.x, self.y)
        if self.point.figure:
            self.point.figure.canvas.draw()
        if self.plotter:
            self.plotter.update_control_point(self.x, self.y, self.point_index)

class CustomFilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Add Custom Filter')

        self.name_label = QLabel('Filter Name:')
        self.kernel_label = QLabel('Kernel (comma-separated values):')
        self.scaling_label = QLabel('Scaling Factor (optional):')

        self.name_edit = QLineEdit()
        self.kernel_edit = QLineEdit()
        self.scaling_edit = QLineEdit()

        self.ok_button = QPushButton('OK')
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.kernel_label)
        layout.addWidget(self.kernel_edit)
        layout.addWidget(self.scaling_label)
        layout.addWidget(self.scaling_edit)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_custom_filter_params(self):
        name = self.name_edit.text()
        kernel_str = self.kernel_edit.text()
        scaling_str = self.scaling_edit.text()

        # Split the kernel_str by ";" to handle multiple sets of values
        kernel_sets = kernel_str.split(';')

        # Initialize an empty list to store the kernel values
        kernel = []

        # Initialize a variable to track the expected number of columns in the matrix
        expected_columns = None

        for set_str in kernel_sets:
            # Convert each set of values to appropriate formats
            try:
                set_values = [float(val.strip()) for val in set_str.split(',')]

                # Check if this is the first set of values
                if expected_columns is None:
                    expected_columns = len(set_values)
                else:
                    # Check if the number of columns matches the first set
                    if len(set_values) != expected_columns:
                        raise ValueError("Matrix rows have different numbers of columns")
                
                kernel.append(set_values)
            except ValueError:
                kernel = None
                break

        # Convert scaling to the appropriate format
        try:
            scaling = float(scaling_str)
        except ValueError:
            scaling = None

        # Print the matrix to the console
        if kernel:
            print("Matrix:")
            for row in kernel:
                print(row)

        return name, kernel, scaling




def main():
    #create_default_filters()
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()

    # Создаем объект BezierCurvePlotter, передавая в него ссылку на ImageProcessingApp
    #bezier_plotter = BezierCurvePlotter(ex)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
