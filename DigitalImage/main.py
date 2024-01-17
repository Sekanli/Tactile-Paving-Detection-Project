# main.py

import tkinter as tk
from videoPlayer import VideoPlayer

def main():
    root = tk.Tk()
    app = VideoPlayer(root, "Tactile Paving Detector")
    root.mainloop()

if __name__ == "__main__":
    main()
