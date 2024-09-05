import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QWidget, \
    QVBoxLayout, QPushButton, QLineEdit, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter, QImage


class CrystalAutomaton(QMainWindow):
    def __init__(self, rule_number):
        super().__init__()

        self.grid_size = 200
        self.cell_size = 4
        self.rule_number = rule_number
        self.rules = self.generate_rules(rule_number)
        self.crystal_color = QColor(255, 255, 255)  # Default crystal color

        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid[self.grid_size // 2][self.grid_size // 2] = 1

        self.initUI()
        self.animation_running = False

    def generate_rules(self, rule_number):
        binary_str = f'{rule_number:010b}'
        rules = []
        for i in range(10):
            rules.append(int(binary_str[i]))
        return rules

    def initUI(self):
        self.setGeometry(100, 100, self.grid_size * self.cell_size, self.grid_size * self.cell_size + 150)
        self.setWindowTitle('Crystal Automaton')

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.view)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_automaton)

        button_layout = QHBoxLayout()

        rule_layout = QHBoxLayout()
        self.rule_input = QLineEdit(self)
        self.rule_input.setFixedWidth(100)
        self.rule_input.setText(str(self.rule_number))
        rule_layout.addWidget(self.rule_input)

        self.rule_button = QPushButton("Change Rule", self)
        self.rule_button.clicked.connect(self.change_rule)
        rule_layout.addWidget(self.rule_button)
        button_layout.addLayout(rule_layout)

        color_layout = QHBoxLayout()
        self.color_input = QLineEdit(self)
        self.color_input.setFixedWidth(100)
        self.color_input.setPlaceholderText("255, 125, 100")
        color_layout.addWidget(self.color_input)

        self.color_button = QPushButton("Change Color", self)
        self.color_button.clicked.connect(self.change_color)
        color_layout.addWidget(self.color_button)
        button_layout.addLayout(color_layout)

        self.animation_button = QPushButton("Start Animation", self)
        self.animation_button.clicked.connect(self.toggle_animation)
        button_layout.addWidget(self.animation_button)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

    def start_automation(self):
        self.timer.start(100)

    def stop_automation(self):
        self.timer.stop()

    def toggle_animation(self):
        if self.animation_running:
            self.stop_automation()
            self.animation_button.setText("Start Animation")
        else:
            self.start_automation()
            self.animation_button.setText("Stop Animation")
        self.animation_running = not self.animation_running

    def change_rule(self):
        new_rule_number = int(self.rule_input.text())
        self.rule_number = new_rule_number
        self.rules = self.generate_rules(self.rule_number)
        self.reset_grid()
        self.render_grid()

    def change_color(self):
        color_str = self.color_input.text()
        try:
            r, g, b = map(int, color_str.split(','))
            self.crystal_color = QColor(r, g, b)
            self.render_grid()
        except ValueError:
            print("Invalid color format. Please use 'R, G, B' format.")

    def reset_grid(self):
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.grid[self.grid_size // 2][self.grid_size // 2] = 1

    def save_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "BMP Files (*.bmp)", options=options)
        if file_name:
            image = QImage(self.grid_size, self.grid_size, QImage.Format_RGB32)
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    color = QColor(0, 0, 0) if self.grid[row][col] == 0 else self.crystal_color
                    image.setPixel(col, row, color.rgb())
            image = image.scaled(self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            image.save(file_name)

    def update_automaton(self):
        new_grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                neighbors = self.count_neighbors(row, col)
                state = self.grid[row][col]
                new_state = self.rules[state * 5 + neighbors]
                new_grid[row][col] = new_state

        self.grid = new_grid
        self.render_grid()

    def count_neighbors(self, row, col):
        count = 0
        if row > 0 and self.grid[row - 1][col] == 1:
            count += 1
        if row < self.grid_size - 1 and self.grid[row + 1][col] == 1:
            count += 1
        if col > 0 and self.grid[row][col - 1] == 1:
            count += 1
        if col < self.grid_size - 1 and self.grid[row][col + 1] == 1:
            count += 1
        return count

    def render_grid(self):
        self.scene.clear()
        self.scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == 1:
                    item = QGraphicsRectItem(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                    item.setBrush(QBrush(self.crystal_color))
                    self.scene.addItem(item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    rule_number = 286
    ex = CrystalAutomaton(rule_number)
    ex.show()
    sys.exit(app.exec_())
