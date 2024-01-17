import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import imageprocess

class VideoPlayer:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.canvas = tk.Canvas(window, width=450, height=600)
        
        self.canvas.pack(anchor=tk.CENTER, expand=True, padx=10, pady=10)

        # Control buttons
        self.btn_upload = tk.Button(window, text="Upload Video", width=50, command=self.upload_video)
        self.btn_upload.pack(anchor=tk.CENTER, expand=True)

        self.btn_play = tk.Button(window, text="Play", width=50, command=self.play_video)
        self.btn_play.pack(anchor=tk.CENTER, expand=True)

        self.btn_pause = tk.Button(window, text="Pause", width=50, command=self.pause_video)
        self.btn_pause.pack(anchor=tk.CENTER, expand=True)

        self.btn_replay = tk.Button(window, text="Replay", width=50, command=self.replay_video)
        self.btn_replay.pack(anchor=tk.CENTER, expand=True)

        # Labels for displaying time
        self.label_total_time = tk.Label(window, text="Total Time: 00:00")
        self.label_total_time.pack(anchor=tk.CENTER)

        self.label_current_time = tk.Label(window, text="Current Time: 00:00")
        self.label_current_time.pack(anchor=tk.CENTER)

        self.video_source = None
        self.current_frame = None
        self.vid = None
        self.pause = True
        self.frame_rate = 0
        self.total_frames = 0
        self.current_frame_number = 0

    def upload_video(self):
        # Reset the video player state
        self.reset_video_player()

        new_video_source = filedialog.askopenfilename()
        if new_video_source:
            self.video_source = new_video_source
            self.vid = cv2.VideoCapture(self.video_source)
            self.pause = False
            self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS)

            # Calculate total duration of the video
            self.total_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            total_milliseconds = int((self.total_frames / self.frame_rate) * 1000)
            total_time = self.format_time(total_milliseconds)
            self.label_total_time.config(text=f"Total Time: {total_time}")

            self.update_frame()

    def play_video(self):
        if self.pause:
            self.pause = False
            self.update_frame()

    def pause_video(self):
        self.pause = True

    def replay_video(self):
     if self.video_source:
        self.vid.release()  # Release the current video capture object
        self.vid = cv2.VideoCapture(self.video_source)  # Reinitialize it
        self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS)  # Recalculate frame rate
        self.pause = False
        self.current_frame_number = 0
        self.update_frame()
        
    def reset_video_player(self):
        # Close any open cv2 windows
        cv2.destroyAllWindows()

        # Release the video capture object if it exists
        if self.vid is not None:
            self.vid.release()

        # Reset relevant variables
        self.video_source = None
        self.current_frame = None
        self.vid = None
        self.pause = True
        self.frame_rate = 0
        self.total_frames = 0
        self.current_frame_number = 0

        # Clear the canvas
        self.canvas.delete("all")

        # Reset labels
        self.label_total_time.config(text="Total Time: 00:00")
        self.label_current_time.config(text="Current Time: 00:00")
    
        
        
        
        
    def update_frame(self):
     if self.vid.isOpened() and not self.pause:
        ret, frame = self.vid.read()
        if ret:
            original_frame = np.copy(frame)
            # Preprocessing and processing steps
            frame = imageprocess.apply_clahe(frame)
            #cv2.imwrite("org.png", original_frame)
            
            #cv2.imwrite("clahe.png", frame)
            
            frame = imageprocess.apply_bilateral_filter(frame)
            
            #cv2.imwrite("bilateral.png", frame)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            avg_brightness = imageprocess.calculate_average_brightness(hsv)
            avg_contrast = imageprocess.calculate_contrast(hsv)

            frame = imageprocess.isolate_color_range(frame)
            
            #cv2.imwrite("color-isolate.png", frame)
            
            processed_frame = imageprocess.apply_gaussian_blur(frame)
            
            #cv2.imwrite("gaussian.png", processed_frame)
            
            processed_frame = imageprocess.remove_noise(processed_frame)
            
            #cv2.imwrite("removenoise.png", processed_frame)
            
            edges = imageprocess.apply_edge_detection(processed_frame)
            
            #cv2.imwrite("edge.png", edges)
            
            edges = imageprocess.clear_noise(edges)
            
            road_edges,road_mask = imageprocess.detect_road_borders_and_create_mask(edges)
            road_edges = cv2.bitwise_and(edges, edges, mask=road_mask)
            
            #cv2.imwrite("road.png", road_edges)
            
            
            road_edges = self.resize_frame(road_edges, max_width=800, max_height=600)
            edges = self.resize_frame(edges, max_width=800, max_height=600)


            # Display the original frame in a separate window
            cv2.imshow('Final Frame', road_edges)
            cv2.moveWindow('Final Frame', 1300 , 0)
            
            cv2.imshow('Edge Frame', edges)
            cv2.moveWindow('Edge Frame', 700 , 0)
            
            # Display the processed frame in the main window
            self.current_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            self.show_frame()




            self.update_time_labels()
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.pause = True  # Or use a different mechanism to stop playback

        # Schedule the next frame update
        self.window.after(int(1000 / self.frame_rate), self.update_frame)
        self.window.update_idletasks()
        self.window.update()

    

    

    def resize_frame(self, frame, max_width=1000, max_height=800):
    # Check if the frame is grayscale (2D) or color (3D)
     if len(frame.shape) == 2:
        height, width = frame.shape
     else:
        height, width, _ = frame.shape

     video_aspect = width / height
     canvas_aspect = max_width / max_height

     if video_aspect > canvas_aspect:
        # Video is wider than the canvas
        new_width = max_width
        new_height = int(new_width / video_aspect)
     else:
        # Video is taller than the canvas
        new_height = max_height
        new_width = int(new_height * video_aspect)

     return cv2.resize(frame, (new_width, new_height))


    def show_frame(self):
    # Clear the canvas
     self.canvas.delete("all")
     frame_resized = self.resize_frame(self.current_frame)
     imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
     self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
     self.canvas.image = imgtk


    def update_time_labels(self):
        if self.vid.isOpened():
            current_milliseconds = int(self.vid.get(cv2.CAP_PROP_POS_MSEC))
            current_time = self.format_time(current_milliseconds)
            self.label_current_time.config(text=f"Current Time: {current_time}")

    @staticmethod
    def format_time(milliseconds):
        seconds = milliseconds // 1000
        mins, secs = divmod(seconds, 60)
        return f"{mins:02d}:{secs:02d}"

