import os
import cv2
import numpy as np
import uuid
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout, QFileDialog
from PySide6.QtCore import Qt
from PIL import Image
import torch
import torchvision
import random
import webbrowser
import qdarkstyle

device = torch.device("cpu")
model = torchvision.models.mobilenet_v3_small()
model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = True

class CW(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.epsilon = 0.2
        self.setWindowTitle("FaceGuard")
        self.create_widgets()

    def create_widgets(self):
        layout = QVBoxLayout()

        title_label = QLabel("FaceGuard")
        title_label.setFont("Helvetica[20]")
        layout.addWidget(title_label)

        button_open_image = QPushButton("Select Image", self)
        button_open_image.clicked.connect(self.open_image)
        layout.addWidget(button_open_image)

        intensity_label = QLabel("Attack Intensity")
        layout.addWidget(intensity_label)

        self.epsilon_slider = QSlider(Qt.Horizontal)
        self.epsilon_slider.setRange(1, 50)
        self.epsilon_slider.setValue(20)
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)
        layout.addWidget(self.epsilon_slider)

        button_attack_image = QPushButton("Attack Image", self)
        button_attack_image.clicked.connect(self.attack_image)
        layout.addWidget(button_attack_image)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        button_donate = QPushButton("Donate", self)
        button_donate.clicked.connect(self.open_donation_link)
        layout.addWidget(button_donate)

        self.setLayout(layout)

    def open_donation_link(self):
        webbrowser.open("https://ko-fi.com/beytek")

    def open_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.jpg *.jpeg *.png)")
        file_path, _ = file_dialog.getOpenFileName(self)

        if file_path:
            self.image_path = file_path
            self.status_label.setText(f"Actual Image: {self.image_path}")

    def attack_image(self):
        if self.image_path:
            num_classes = 1000
            target = torch.tensor([random.randint(0, num_classes - 1)]).to(device)
            image = Image.open(self.image_path)
            image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
            image_tensor.requires_grad = True
            output = model(image_tensor)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = image_tensor.grad.data
            data_grad = data_grad.detach()
            sign_data_grad = data_grad.sign()
            perturbed_image = image_tensor + self.epsilon * sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            perturbed_image = perturbed_image.squeeze(0).cpu().detach().numpy()
            perturbed_image = np.transpose(perturbed_image, (1, 2, 0))
            perturbed_image = (perturbed_image * 255).astype(np.uint8)

            if perturbed_image.size > 0:
                input_dir = os.path.dirname(self.image_path)
                random_uuid = str(uuid.uuid1())[:6]
                output_filename = f"{random_uuid}-FaceGuard.jpg"
                output_path = os.path.join(input_dir, output_filename)
        

                perturbed_image_with_filter = Image.fromarray(perturbed_image)
                

                cv2.imwrite(output_path, cv2.cvtColor(np.array(perturbed_image_with_filter), cv2.COLOR_RGB2BGR))
                self.status_label.setText(f"Image saved as {output_path}")

        else:
            self.status_label.setText("Choose a picture first.")

    def update_epsilon(self):
        self.epsilon = self.epsilon_slider.value() / 100.0
        self.status_label.setText(f"Intensity Force: {self.epsilon * 200:.0f}%")


def main():
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
    window = CW()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()
