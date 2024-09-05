import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar, QAction, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QCheckBox, QGroupBox, QRadioButton
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import numpy as np
import colorsys
import shutil
import os
class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processing App')
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.createMenus()
        self.createImageBoxes()

        # Create a horizontal layout for "Apply to Channels" and "Observe Space" widgets
        channels_and_space_layout = QHBoxLayout()
        self.createChannelsPanel(channels_and_space_layout)
        self.createObserveSpace(channels_and_space_layout)
        self.createMonocromicalSettings(channels_and_space_layout)
        self.radioButton_All.setChecked(True)
        self.radioButton_RGB.setChecked(True)
        self.radioButton_All.setChecked(True)
        # Add the horizontal layout to the main vertical layout
        self.layout.addLayout(channels_and_space_layout)
        self.createDiagrams()

    def createMenus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        open_first_image_action = QAction('Open First Image', self)
        open_first_image_action.triggered.connect(self.openFirstImage)
        file_menu.addAction(open_first_image_action)

        open_second_image_action = QAction('Open Second Image', self)
        open_second_image_action.triggered.connect(self.openSecondImage)
        file_menu.addAction(open_second_image_action)

        save_result_action = QAction('Save Result', self)
        save_result_action.triggered.connect(self.saveResult)
        file_menu.addAction(save_result_action)

    def createImageBoxes(self):
        # Create QGraphicsView widgets for image display
        self.source_image_view = QGraphicsView(self)
        self.target_image_view = QGraphicsView(self)
        self.result_image_view = QGraphicsView(self)

        # Create QGraphicsScenes to hold the images
        self.source_scene = QGraphicsScene()
        self.target_scene = QGraphicsScene()
        self.result_scene = QGraphicsScene()

        # Set scenes for the QGraphicsViews
        self.source_image_view.setScene(self.source_scene)
        self.target_image_view.setScene(self.target_scene)
        self.result_image_view.setScene(self.result_scene)

        # Add QGraphicsViews to layout
        image_box_layout = QHBoxLayout()
        image_box_layout.addWidget(self.source_image_view)
        image_box_layout.addWidget(self.target_image_view)
        image_box_layout.addWidget(self.result_image_view)

        self.layout.addLayout(image_box_layout)

    def createChannelsPanel(self, layout):
        panel_group = QGroupBox('Примінити до каналів', self)
        panel_layout = QVBoxLayout()

        self.red_checkbox = QCheckBox('Red', self)
        self.green_checkbox = QCheckBox('Green', self)
        self.blue_checkbox = QCheckBox('Blue', self)

        panel_layout.addWidget(self.red_checkbox)
        panel_layout.addWidget(self.green_checkbox)
        panel_layout.addWidget(self.blue_checkbox)

        self.red_checkbox.stateChanged.connect(self.update_correction)
        self.green_checkbox.stateChanged.connect(self.update_correction)
        self.blue_checkbox.stateChanged.connect(self.update_correction)

        panel_group.setLayout(panel_layout)
        layout.addWidget(panel_group)
    
    def createObserveSpace(self, layout):
        space_group = QGroupBox('Простори', self)
        space_layout = QVBoxLayout() 

        self.radioButton_RGB = QRadioButton('LAB', self)
        self.radioButton_YIQ = QRadioButton('YIQ', self)
        self.radioButton_HSV = QRadioButton('HSV', self)

        space_layout.addWidget(self.radioButton_RGB)
        space_layout.addWidget(self.radioButton_YIQ)
        space_layout.addWidget(self.radioButton_HSV)

        space_group.setLayout(space_layout)
        layout.addWidget(space_group)
    
    def createMonocromicalSettings(self, layout):
        space_group = QGroupBox('Монохром', self)
        space_layout = QVBoxLayout()  # Use QVBoxLayout for radio buttons

        self.radioButton_R = QRadioButton('R', self)
        self.radioButton_G = QRadioButton('G', self)
        self.radioButton_B = QRadioButton('B', self)
        self.radioButton_All = QRadioButton('All', self)

        space_layout.addWidget(self.radioButton_R)  
        space_layout.addWidget(self.radioButton_G)
        space_layout.addWidget(self.radioButton_B)
        space_layout.addWidget(self.radioButton_All)

        self.radioButton_R.toggled.connect(self.updateImage)
        self.radioButton_G.toggled.connect(self.updateImage)
        self.radioButton_B.toggled.connect(self.updateImage)
        self.radioButton_All.toggled.connect(self.updateImage)

        space_group.setLayout(space_layout)

        layout.addWidget(space_group)


    def createDiagrams(self):
        diagrams_layout = QHBoxLayout()

        self.figure_source, self.ax_source = plt.subplots(figsize=(5, 3))
        self.canvas_source = FigureCanvas(self.figure_source)
        diagrams_layout.addWidget(self.canvas_source)
        self.ax_source.set_title('Source Image')

        self.figure_target, self.ax_target = plt.subplots(figsize=(5, 3))
        self.canvas_target = FigureCanvas(self.figure_target)
        diagrams_layout.addWidget(self.canvas_target)
        self.ax_target.set_title('Target Image')

        self.figure_result, self.ax_result = plt.subplots(figsize=(5, 3))
        self.canvas_result = FigureCanvas(self.figure_result)
        diagrams_layout.addWidget(self.canvas_result)
        self.ax_result.set_title('Result Image')

        self.layout.addLayout(diagrams_layout)

    def update_source_histogram(self):
        if hasattr(self, 'source_image'):
            self.plot_histogram(self.ax_source, 'Source Image', self.source_image)
            self.canvas_source.draw()

    def update_target_histogram(self):
        if hasattr(self, 'target_image'):
            self.plot_histogram(self.ax_target, 'Target Image', self.target_image)
            self.canvas_target.draw()

    def update_result_histogram(self):
        if hasattr(self, 'corrected_rgb'):
            self.plot_histogram(self.ax_result, 'Result Image', self.corrected_rgb)
            self.canvas_result.draw()

    def calculate_histogram(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate the histogram
        hist = cv2.calcHist([gray_image], [0.00001], None, [256], [0.00001, 256])
        
        return hist

    def plot_histogram(self, ax, title, image):
        lab_image = self.rgb_to_lab(image)
        ax.clear()
        # Split the Lab image into channels
        lab_channels = cv2.split(lab_image)

        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            histogram = cv2.calcHist([lab_channels[i]], [0], None, [256], [0.00001, 256])
            ax.plot(histogram, color=color, label=f'Channel {i}')
        
        ax.set_title(title)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 256)
        ax.legend()
        ax.grid(True)

    def openFirstImage(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open First Image', '', 'Images (*.bmp *.jpg *.jpeg *.png)')

        if file_path:
            # Use cv2 to read the image
            self.source_image = cv2.imread(file_path)
            
            # Display the image in the QGraphicsView
            self.display_cv2_image(file_path, self.source_scene)

            # Update the histogram for the source image
            self.update_source_histogram()


    def openSecondImage(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Second Image', '', 'Images (*.bmp *.jpg *.jpeg *.png)')
        self.file_path = file_path
        if file_path:
            # Use cv2 to read the image
            self.target_image = cv2.imread(file_path)

            # Display the image in the QGraphicsView
            self.display_cv2_image(file_path, self.target_scene)

            # Update the histogram for the source image
            self.update_target_histogram()

            # Create a copy of self.target_image in the same directory with the name "corrected_rgb.bmp"
            if os.path.isfile(file_path):
                folder_path = os.path.dirname(file_path)
                output_path = os.path.join(folder_path, "corrected_rgb.bmp")
                self.file_path = output_path
                cv2.imwrite(output_path, self.target_image)


    def saveResult(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Result', '', 'Bitmap Images (*.bmp)')
        
        if file_path:
            # Perform image processing and save the result to the specified path
                if self.radioButton_R.isChecked() or self.radioButton_G.isChecked() or self.radioButton_B.isChecked():
                    cv2.imwrite(file_path, self.mono_rgb)
                    print(f"Result saved to: {file_path}")
                else:
            # Save the corrected image using OpenCV
                    cv2.imwrite(file_path, self.corrected_rgb)
                    print(f"Result saved to: {file_path}")
    
    def rgb_to_lab(self, rgb_image):
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
        return lab_image

    def lab_to_rgb(self, lab_image):
        rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)
        return rgb_image
    
    def display_cv2_image(self, cv2_image, scene):
        if cv2_image is not None:
            # Convert the cv2 image to RGB format
            # rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            # Convert the RGB image to QPixmap for display in QGraphicsScene
            # pixmap = QPixmap.fromImage(QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888))
            pixmap = QPixmap(cv2_image)
            item = QGraphicsPixmapItem(pixmap)
            scene.clear()
            scene.addItem(item)
            self.show()

            # Clear the scene and add the new pixmap
            # scene.clear()
            # scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())  # Установите размер сцены
            # scene.addPixmap(pixmap)


    def color_correction(self, target_lab, source_lab, channels, color_space):
        # Convert to the specified color space
        if color_space == 'HSV':
            source_lab = self.rgb_to_hsv(self.lab_to_rgb(source_lab))
            target_lab = self.rgb_to_hsv(self.lab_to_rgb(target_lab))
        elif color_space == 'YIQ':
            source_lab = self.rgb_to_yiq(self.lab_to_rgb(source_lab))
            target_lab = self.rgb_to_yiq(self.lab_to_rgb(target_lab))

        # Calculate the means and standard deviations for each channel
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))

        # Create a copy of the source Lab image for correction
        corrected_lab = np.copy(source_lab)

        # Loop through the selected channels and apply correction
        for channel in channels:
            if channel == 'Red':
                channel_idx = 0
            elif channel == 'Green':
                channel_idx = 1
            elif channel == 'Blue':
                channel_idx = 2
            else:
                channel_idx = None

            if channel_idx is not None:
                # Apply the color correction to the selected channel
                corrected_lab[:, :, channel_idx] = (source_lab[:, :, channel_idx] - source_mean[channel_idx]) * (
                    target_std[channel_idx] / source_std[channel_idx]) + target_mean[channel_idx]

        # Convert back to the original color space
        if color_space == 'YIQ':
            corrected_lab = self.rgb_to_lab(self.yiq_to_rgb(corrected_lab))
        elif color_space == 'HSV':
            corrected_lab = self.rgb_to_lab(self.hsv_to_rgb(corrected_lab))

        return corrected_lab

    def rgb_to_hsv(self, rgb_image):
        # Преобразование RGB в HSV
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        return hsv_image

    def hsv_to_rgb(self, hsv_image):
        # Преобразование HSV в RGB
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return rgb_image

    def rgb_to_yiq(self, rgb_image):
        # Define the RGB to YIQ transformation matrix
        rgb_to_yiq_matrix = np.array([[0.299, 0.587, 0.114],
                                      [0.596, -0.275, -0.321],
                                      [0.212, -0.528, 0.311]])

        # Normalize the input RGB image
        normalized_rgb_image = rgb_image / 255.0

        # Perform the matrix multiplication
        yiq_image = np.dot(normalized_rgb_image, rgb_to_yiq_matrix.T)

        return yiq_image

    def yiq_to_rgb(self, yiq_image):
        # Define the YIQ to RGB transformation matrix
        yiq_to_rgb_matrix = np.array([[1.0, 0.956, 0.621],
                                      [1.0, -0.272, -0.647],
                                      [1.0, -1.106, -1.703]])

        # Perform the matrix multiplication
        rgb_image = np.dot(yiq_image, yiq_to_rgb_matrix.T)

        # Clip the values to the [0, 1] range and scale to [0, 255]
        rgb_image = np.clip(rgb_image, 0, 1) * 255

        # Convert to uint8 data type
        rgb_image = rgb_image.astype(np.uint8)

        return rgb_image
    
    def update_correction(self):
        try:
            # Проверяем, что исходное и целевое изображения загружены
            if not hasattr(self, 'source_image') or not hasattr(self, 'target_image'):
                raise Exception("Please load both source and target images.")
            
            # Get the selected channels from checkboxes
            channels_to_correct = []
            if self.red_checkbox.isChecked():
                channels_to_correct.append('Red')
            if self.green_checkbox.isChecked():
                channels_to_correct.append('Green')
            if self.blue_checkbox.isChecked():
                channels_to_correct.append('Blue')

            # Get the selected color space from radio buttons
            color_space = None
            if self.radioButton_RGB.isChecked():
                color_space = 'RGB'
            elif self.radioButton_YIQ.isChecked():
                color_space = 'YIQ'
            elif self.radioButton_HSV.isChecked():
                color_space = 'HSV'

            # Проверяем, что хотя бы один канал и цветовое пространство выбраны
            if not channels_to_correct or not color_space:
                raise Exception("Please select at least one channel and a color space.")

            # Продолжаем выполнение кода
            self.source_image_lab = self.rgb_to_lab(self.source_image)
            self.target_image_lab = self.rgb_to_lab(self.target_image)
            self.corrected_lab = self.color_correction(self.source_image_lab, self.target_image_lab, channels_to_correct, color_space)
            self.corrected_rgb = self.lab_to_rgb(self.corrected_lab)
            cv2.imwrite(self.file_path, self.corrected_rgb)
            # Display the corrected image in QGraphicsView
            self.display_cv2_image(self.file_path, self.result_scene)
            self.update_result_histogram()

        except Exception as e:
            # Обработка исключения
            print(f"Error: {str(e)}")

    def updateImage(self):
        try:
            # Проверяем, что скорректированное изображение загружено
            if not hasattr(self, 'corrected_rgb'):
                raise Exception("Please perform color correction first.")

            channel = None
            if self.radioButton_R.isChecked():
                channel = 2  # Индекс красного канала в RGB
            elif self.radioButton_G.isChecked():
                channel = 1  # Индекс зеленого канала в RGB
            elif self.radioButton_B.isChecked():
                channel = 0  # Индекс синего канала в RGB

            if channel is not None:
                # Создайте копию тензора RGB
                mono_rgb = np.copy(self.corrected_rgb)

                # Обнулите остальные каналы
                for i in range(3):
                    if i != channel:
                        mono_rgb[:, :, i] = 0
                self.mono_rgb=mono_rgb
                # Отобразите изображение в режиме монохрома в QGraphicsView
                 # Create a copy of self.target_image in the same directory with the name "corrected_rgb.bmp"
                if os.path.isfile(self.file_path):
                    folder_path = os.path.dirname(self.file_path)
                    output_path = os.path.join(folder_path, "corrected_rgb_mono.bmp")
                    self.file_path_mono = output_path
                    cv2.imwrite(output_path, mono_rgb)
                self.display_cv2_image(self.file_path_mono, self.result_scene)
                self.update_result_histogram()
            else:
                # Если не выбран ни один канал, отобразите изначальное скорректированное изображение
                self.display_cv2_image(self.file_path, self.result_scene)
                self.update_result_histogram()

        except Exception as e:
            # Обработка исключения
            print(f"Error: {str(e)}")

    def closeEvent(self, event):
    # Этот метод будет вызван при закрытии приложения
        if hasattr(self, 'file_path') and os.path.isfile(self.file_path):
            os.remove(self.file_path)
        event.accept()
        if hasattr(self, 'file_path_mono') and os.path.isfile(self.file_path_mono):
            os.remove(self.file_path_mono)
        event.accept()


        
            
def main():
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
