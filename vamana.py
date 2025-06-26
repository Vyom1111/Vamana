import sys
import cv2
import os
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from scipy.spatial import KDTree
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QSplitter, QListWidget
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import csv
from datetime import datetime
import time
import torch
from Predict import predict_value

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

prev_time = 0 
class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_variables()

    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle('YOLO Substance Detection - Enhanced UI')
        self.setGeometry(100, 100, 1200, 800)

        # Left Panel
        self.file_label = QLabel('No file loaded')
        self.start_button = QPushButton('Upload Folder')
        self.start_button.clicked.connect(self.upload_folder)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_processing)
        self.csv_button = QPushButton('View CSV Updates')
        self.csv_button.clicked.connect(self.view_csv)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.select_file)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(self.stop_button)
        left_layout.addWidget(self.csv_button)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Right Panel
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(800, 600)
        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)

        right_top_layout = QVBoxLayout()
        right_top_layout.addWidget(self.video_label)

        right_bottom_layout = QVBoxLayout()
        right_bottom_layout.addWidget(self.text_output)

        right_widget_top = QWidget()
        right_widget_top.setLayout(right_top_layout)
        right_widget_bottom = QWidget()
        right_widget_bottom.setLayout(right_bottom_layout)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(right_widget_top)
        right_splitter.addWidget(right_widget_bottom)
        right_splitter.setSizes([600, 200])

        # Main Splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([300, 900])

        # Central Widget
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Play Button
        self.play_button = QPushButton('Play')
        self.play_button.clicked.connect(self.start_processing)
        self.play_button.setMaximumWidth(100)

        # Total Volume Button
        self.total_volume_button = QPushButton('Show Total Volume')
        self.total_volume_button.setMaximumWidth(200)
        self.total_volume_button.clicked.connect(self.show_total_volume)

        # Add buttons to layout
        main_layout.addWidget(self.play_button, alignment=Qt.AlignTop | Qt.AlignRight)
        main_layout.addWidget(self.total_volume_button, alignment=Qt.AlignTop | Qt.AlignRight)

    def init_variables(self):
        """Initialize variables."""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.model = YOLO(r'D:\Python\model_use.pt').to(device)  # Load model and move to GPU
        self.filename = None  # CSV filename
        self.color_identified = None 
        self.total_pixel_area = 0
        # self.previous_area_pixels = 0
        # self.area_change_threshold = 0
        self.pixel_normalization_factor = 0.00001
        self.color_ranges = {
            'milk': ([101, 110, 190], [107, 120, 210]),  # White
            'phanta': ([180, 140, 200], [210, 240, 250]),  # Light brown
            'lavanodaka': ([180, 210, 230], [210, 230, 250]),  # Light bluish
            'saliva': ([200, 180, 160], [230, 210, 190]),  # Pale yellow
            'pitta': ([180, 130, 40], [220, 170, 60]),  # Yellow-brown
            'water': ([200, 225, 250], [230, 245, 255]),  # Light blue
            'milk_phanta': ([210, 190, 130], [240, 210, 150]),  # Creamy brown
            'milk_saliva': ([220, 200, 180], [250, 230, 210]),  # Light pinkish white
            'milk_lavanodaka': ([220, 230, 235], [250, 250, 255]),  # Slightly bluish white
            'milk_pitta': ([200, 180, 100], [230, 210, 130]),  # Pale yellowish white
            'milk_water': ([220, 240, 250], [255, 255, 255]),  # Almost white
            'phanta_saliva': ([180, 150, 120], [210, 170, 140]),  # Murky brown
            'phanta_lavanodaka': ([170, 160, 120], [200, 180, 140]),  # Muddy yellow
            'phanta_pitta': ([170, 120, 50], [210, 140, 70]),  # Dark yellow-brown
            'phanta_water': ([190, 160, 100], [210, 180, 120]),  # Diluted tea
            'lavanodaka_saliva': ([190, 190, 200], [220, 210, 220]),  # Pale bluish-pink
            'lavanodaka_pitta': ([160, 140, 80], [200, 160, 100]),  # Greenish-yellow
            'lavanodaka_water': ([200, 220, 240], [230, 250, 255]),  # Clear pale blue
            'saliva_pitta': ([170, 140, 80], [200, 160, 90]),  # Yellowish pale pink
            'saliva_water': ([190, 200, 220], [220, 230, 240]),  # Transparent pale white-blue
            'pitta_water': ([170, 140, 70], [200, 170, 90]),  # Faded yellow-green
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.video_files = []    # List of video paths
        self.current_video_idx = 0  # Index of current video
        self.final_processing_done = False  # To check when all videos are done
        self.folder_name = None  # To store the folder name for CSV naming


    def upload_folder(self):
        """Upload a folder and list video files."""
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.folder_name = os.path.basename(folder)
            self.file_list.clear()
            self.video_files = []  # Reset video list
            for filename in os.listdir(folder):
                if filename.endswith(('.mp4', '.avi', '.MOV')):
                    full_path = os.path.join(folder, filename)
                    self.video_files.append(full_path)
                    self.file_list.addItem(full_path)
            self.file_label.setText("Folder loaded. Ready to process all files.")
            self.current_video_idx = 0
            # Initialize CSV file when folder is loaded
            self.init_csv()

    def select_file(self, item):
        """Handle file selection from the list."""
        self.selected_file = item.text()
        self.file_label.setText(f'Selected File: {os.path.basename(self.selected_file)}')

    def init_csv(self):
        """Initialize CSV file based on the folder name."""
        if self.folder_name:
            self.filename = f'{self.folder_name}_combined.csv'
            try:
                with open(self.filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Video File", "Color (CSS3)", "Substance", "Cumulative Area"])
            except Exception as e:
                self.text_output.append(f"Error initializing CSV: {e}")

    def process_frame(self):
        """Process each frame using the YOLO model."""
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.timer.stop()
            self.current_video_idx += 1
            self.start_next_video()
            return
        try:
            # Resize frame to 640x640 (divisible by 32)
            resized_frame = cv2.resize(frame, (640, 640))

            # Convert frame to CHW format and add batch dimension
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float().to(device)  # HWC to CHW
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension (BCHW)

            # Pass the tensor to the YOLO model
            results = self.model(frame_tensor)
            frame_text = "Results:\n"

            # Detect head position (head down or not)
            head_down, face_bbox = self.detect_head_down(frame)
            if not head_down:
    # If head is up, skip 10 seconds of frames
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                skip_frames = int(fps * 5)  # Number of frames to skip
                new_frame = current_frame + skip_frames
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
            for result in results:
                class_id = 1 # most important - sink id will only be counted  
                for mask, class_id  in zip(result.masks.data, result.boxes.cls):
                    # Prepare the segmentation mask
                    mask_image = mask.cpu().numpy().astype(np.uint8)
                    mask_image = (mask_image > 0.5).astype(np.uint8)
                    mask_image_resized = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    mask_image_resized = np.expand_dims(mask_image_resized, axis=-1)

                    # Overlay the mask on the original frame (color it, here it's green)
                    colored_mask = cv2.applyColorMap(mask_image_resized * 255, cv2.COLORMAP_JET) #normalized 
                    frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

                    # Volume calculation only if the head is down
                    if head_down:
                        average_color = self.identify_color(frame, mask_image_resized)
                        self.color_identified = self.blend_color_name(average_color) #Optimal method for mapping RGB to Color Name very specifically
                        # color_identified = average_color 
                        current_area_pixels = self.calculate_area(mask_image_resized)
                        self.total_pixel_area += current_area_pixels * self.pixel_normalization_factor
                        identified_substance = self.identify_substance(tuple(map(int, average_color)))
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        current_video_name = os.path.basename(self.selected_file)
                        with open(self.filename, mode='a', newline='') as file: #Formatting of CSV File 
                            writer = csv.writer(file)
                            writer.writerow([timestamp, current_video_name, self.color_identified, identified_substance, f'{self.total_pixel_area:.2f}pxx2'])
                        # Display detected info
                        cv2.putText(frame, f'Substance: {identified_substance}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

             
            if face_bbox is not None:
                x, y, w, h = face_bbox
                # Determine color of the bounding box based on head position
                box_color = (0, 255, 0) if head_down else (0, 0, 255)  # Green if head is down, red if up

                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                # Display text at the top-left corner of the bounding box
                text = "Head Down" if head_down else "Head Up"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            self.text_output.setText(frame_text)

            # Convert frame to RGB and display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimage).scaled(self.video_label.size(), Qt.KeepAspectRatio))

        except Exception as e:
            self.text_output.append(f"Error processing frame: {e}")

    def blend_color_name(self,rgb_tuple):
        css3_db = {hex_to_rgb(hex): name for hex, name in CSS3_HEX_TO_NAMES.items()}
        color_tree = KDTree(list(css3_db.keys()))
        distances, indices = color_tree.query(rgb_tuple, k=2)
        
        if distances[0] < 10:  # Threshold for exact match
            return list(css3_db.values())[indices[0]]
        
        name1, name2 = list(css3_db.values())[indices[0]], list(css3_db.values())[indices[1]]
        return f"{name1}-{name2}"

    def detect_head_down(self, frame):
        """
        Detect if the patient's head is down based on face alignment.
        Uses a Haar Cascade or pre-trained DNN for face detection and posture estimation.
        """
        if not hasattr(self, 'previous_posture'):
            self.previous_posture = None  # Start with no previous posture

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]

            if not hasattr(self, 't0_height') or self.t0_height is None:
                self.t0_height = h  # Initialize t0_height with the first detected face height
            
            head_down = h < self.t0_height  # Head down if current height is less than t0_height

            self.previous_posture = head_down

            return head_down, (x, y, w, h)  # Return the current posture and bounding box

        else:
            if self.previous_posture is None:
                return False, None  # Head up by default
            else:
                return self.previous_posture, None

    def start_processing(self):
        """Start processing all videos."""
        if self.video_files:
            self.total_pixel_area = 0
            self.previous_area_pixels = 0
            self.final_processing_done = False
            self.start_next_video()
        else:
            self.text_output.append("Please upload a folder with videos first.")

    def start_next_video(self):
        """Start processing the next video in the list."""
        if self.current_video_idx < len(self.video_files):
            self.selected_file = self.video_files[self.current_video_idx]
            self.file_label.setText(f'Processing File: {os.path.basename(self.selected_file)}')
            self.cap = cv2.VideoCapture(self.selected_file)
            if not self.cap.isOpened():
                self.text_output.append(f"Error: Unable to open {os.path.basename(self.selected_file)}")
                self.current_video_idx += 1
                self.start_next_video()
            else:
                self.text_output.append(f"Started processing: {os.path.basename(self.selected_file)}")
                self.timer.start(30)
        else:
            self.final_processing_done = True
            output = predict_value(self.total_pixel_area)
            self.text_output.append(f'Final Total Volume (mapped): {output} ml')
            self.text_output.append("All videos processed.")
            
            # Add final total volume to CSV
            self.add_final_volume_to_csv(output)

    def add_final_volume_to_csv(self, volume):
        """Add a final line with the total volume to the CSV file."""
        if self.filename:
            try:
                with open(self.filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([])  # Empty line for separation
                    writer.writerow(["Final Total Volume", f"{volume} ml"])
            except Exception as e:
                self.text_output.append(f"Error adding final volume to CSV: {e}")

    def stop_processing(self):
        """Stop the video processing."""
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.video_label.clear()
        self.text_output.append("Processing stopped.")
        self.cap = None

    def show_total_volume(self):
        """Display the total pixel area or final volume."""
        if self.final_processing_done:
            output = predict_value(self.total_pixel_area)
            self.text_output.append(f"Final Total Volume: {output} ml")
        else:
            self.text_output.append(f"Total Pixel_Area Detected so far: {self.total_pixel_area:.2f} pxx2")
            self.text_output.append(f"Color Identified: {self.color_identified}")

    def view_csv(self):
        """View the updates saved in the CSV file."""
        if self.filename and os.path.exists(self.filename):
            os.system(f'notepad {self.filename}')  # Open the CSV file in Notepad (Windows)
        else:
            self.text_output.append("No CSV file found to display.")

    def identify_color(self, frame, mask_image):
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_image) #It isolates the region we care about.
        average_color = cv2.mean(masked_frame, mask=mask_image)[:3]
        return average_color

    def identify_substance(self, color):
        """
        Identify the substance based on the detected average color.
        """
        for substance, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            if np.all(color >= lower) and np.all(color <= upper):
                return substance
        return "Unknown"
    
    def calculate_area(self, mask_image):
        area_pixels = np.count_nonzero(mask_image)
        return area_pixels

    def closeEvent(self, event):
        """
        Ensure resources are released when the application is closed.
        """
        self.stop_processing()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
    print(f"Using device: {device}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")