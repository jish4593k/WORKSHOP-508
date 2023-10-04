import json
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import cv2
import click
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout, QHeaderView, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QFont
from PyQt5.QtCore import Qt
from skimage import io, color, filters, morphology, measure

class FruitDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Fruit Detection App')
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(['Image', 'Apples', 'Bananas', 'Oranges'])

        self.process_button = QPushButton('Process Images')
        self.process_button.clicked.connect(self.process_images)

        self.export_button = QPushButton('Export Results')
        self.export_button.clicked.connect(self.export_results)

        self.save_images_button = QPushButton('Save Processed Images')
        self.save_images_button.clicked.connect(self.save_processed_images)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.process_button)
        self.layout.addWidget(self.result_table)
        self.layout.addWidget(self.export_button)
        self.layout.addWidget(self.save_images_button)

        self.central_widget.setLayout(self.layout)

        self.results = {}

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)

    def detect_fruits(self, image_path):
        image = io.imread(image_path)

        image_hsv = color.rgb2hsv(image)

        orange_mask = (image_hsv[:, :, 0] >= 0.06) & (image_hsv[:, :, 0] <= 0.12)
        apple_mask = (image_hsv[:, :, 0] >= 0.0) & (image_hsv[:, :, 0] <= 0.05)
        banana_mask = (image_hsv[:, :, 0] >= 0.09) & (image_hsv[:, :, 0] <= 0.15)

        oranges = orange_mask.astype(np.uint8) * 255
        apples = apple_mask.astype(np.uint8) * 255
        bananas = banana_mask.astype(np.uint8) * 255

        oranges_contours = self.find_contours(oranges)
        apples_contours = self.find_contours(apples)
        bananas_contours = self.find_contours(bananas)

        oranges_count = self.count_fruit_contours(oranges_contours, 8000)
        apples_count = self.count_fruit_contours(apples_contours, 7000)
        bananas_count = self.count_fruit_contours(bananas_contours, 8000)

        return {'apple': apples_count, 'banana': bananas_count, 'orange': oranges_count}, oranges, apples, bananas

    def find_contours(self, mask):
        binary_image = mask > filters.threshold_otsu(mask)

        cleaned_image = morphology.remove_small_objects(binary_image, min_size=100)

        contours = measure.find_contours(cleaned_image, 0.5)

        return contours

    def count_fruit_contours(self, contours, area_threshold):
        count = 0
        for contour in contours:
            area = measure.perimeter(contour) ** 2 / (4 * np.pi)
            if area > area_threshold:
                count += 1
        return count

    def process_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Image Files", "", "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)", options=options)

        if not file_paths:
            return

        self.result_table.setRowCount(len(file_paths))

        for i, image_path in enumerate(file_paths):
            image_name = Path(image_path).name
            fruits, oranges, apples, bananas = self.detect_fruits(image_path)

            self.results[image_name] = fruits

            item_image = QTableWidgetItem(image_name)
            item_apples = QTableWidgetItem(str(fruits['apple']))
            item_bananas = QTableWidgetItem(str(fruits['banana']))
            item_oranges = QTableWidgetItem(str(fruits['orange']))

            self.result_table.setItem(i, 0, item_image)
            self.result_table.setItem(i, 1, item_apples)
            self.result_table.setItem(i, 2, item_bananas)
            self.result_table.setItem(i, 3, item_oranges)

            self.display_segmented_images(image_path, oranges, apples, bananas)

    def display_segmented_images(self, image_path, oranges, apples, bananas):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for mask, color, fruit_name in zip([oranges, apples, bananas], [(255, 0, 0), (0, 255, 0), (0, 0, 255)], ['Oranges', 'Apples', 'Bananas']):
            masked_image = cv2.addWeighted(image, 1, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.5, 0)
            for contour in self.find_contours(mask):
                cv2.drawContours(masked_image, [contour.astype(int)], -1, color, 2)

            pixmap = QPixmap.fromImage(QImage(masked_image.data, w, h, w * 3, QImage.Format_RGB888))

            label = QLabel(self)
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            label.setToolTip(f'{fruit_name} in {Path(image_path).name}')

            self.layout.addWidget(label)

    def export_results(self):
        if not self.results:
            QMessageBox.warning(self, "Export Warning", "No results to export.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results JSON", "", "JSON Files (*.json);;All Files (*)", options=options)

        if file_path:
            with open(file_path, 'w') as json_file:
                json.dump(self.results, json_file, indent=4)
                QMessageBox.information(self, "Export Success", "Results exported successfully.")

    def save_processed_images(self):
        if not self.results:
            QMessageBox.warning(self, "Save Warning", "No processed images to save.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save Processed Images", options=options)

        if not folder_path:
            return

        for i, image_path in enumerate(self.results.keys()):
            image_name = Path(image_path).name
            apples, bananas, oranges = self.detect_fruits(image_path)[1:]
            
            for fruit, mask in zip(['apples', 'bananas', 'oranges'], [apples, bananas, oranges]):
                fruit_name = fruit.capitalize()
                masked_image = cv2.bitwise_and(cv2.imread(image_path), cv2.imread(image_path), mask=mask)
                cv2.imwrite(f"{folder_path}/{image_name.split('.')[0]}_{fruit}_{i + 1}.jpg", masked_image)

def run_application():
    app = QApplication(sys.argv)
    window = FruitDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_application()
