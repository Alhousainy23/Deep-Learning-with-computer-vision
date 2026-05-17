import time 
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog,messagebox,simpledialog
import threading 
import sys 
import os 
#===========================================================================================
#============================Custom tkinter GUI Configuration===============================
ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (default), "green", "dark-blue", "white"
stop_flag , cap , contours = False , None , []
#===========================================================================================
#image processing functions
def select_image():
    path = filedialog.askopenfilename(title ="choose an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")] )
    if path: threading.Thread(target=process_image, args=(path,)).start()
#===========================================================================================
def process_image(path):
    global contours
    image = cv2.imread(path)
    if image is None:
        messagebox.showerror("Error", "Failed to load the image.")
        return
    image = cv2.resize(image, (500, 500))
    img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(img,(400,400))
    _,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image,contours,-1,(0,255,0),2)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_image.jpg")
    cv2.imwrite(save_path, image)
    print(f"✅ Image saved: {save_path}")
    cv2.imshow("Contours",image)
    cv2.setMouseCallback("Contours", click_event,img)
    print(f"Number of contours found: {len(contours)}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#===========================================================================================
def click_event(event,x,y,flags,param):
    global contours
    if event == cv2.EVENT_LBUTTONDOWN:
        for cnt in contours:
            if cv2.pointPolygonTest(cnt,(x,y),False)>=0:
                label = simpledialog.askstring("Label", "Enter label for this contour:")
                if label:
                    cv2.drawContours(param,[cnt],-1,(0,255,0),2)
                    cv2.putText(param,label,(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                    cv2.imshow("Contours",param)
                break 
#===========================================================================================
def start_video():
    path = filedialog.askopenfilename(title ="choose a video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")] )
    if not path : return
    threading.Thread(target=process_video, args=(path,)).start()
#===========================================================================================
def process_video(path):
    global cap , stop_flag
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open the video.")
        return
    print("Starting video processing...")
    stop_flag = False
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (500, 500))
    while cap.isOpened() and not stop_flag:
        if stop_flag:
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret : break 
        frame = cv2.resize(frame,(400,400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours_vid,_= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours_vid, -1, (0, 255, 0), 2)
        cv2.imshow("Video Contours", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):break
    cap.release()
    cv2.destroyAllWindows()
    print("🎬 The video processing has been completed.")
#===========================================================================================
def stop_video():
    global stop_flag 
    stop_flag = not stop_flag
    if stop_flag:
        btn.configure(text="Resume Video")
        print("⏸️ Video processing paused.")
    else:
        btn.configure(text="Stop Video")
        print("▶️ Video processing resumed.")
#===========================================================================================
def exit_app():
    def safe_exit():
        global cap
        print("Exiting the application...")
        if cap : cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.5)
        root.quit()
        root.destroy()
        sys.exit(0)
    threading.Thread(target=safe_exit,daemon=True).start()
#============================GUI Layout===============================
root = ctk.CTk()
root.title("Contours Detection & Labeling")
root.geometry("400x400")
root.resizable(False, False)
# Buttons color and style
GRAY_FG       = "#6B6B6B" 
GRAY_HOVER    = "#808080"   
TEXT_COLOR    = "#FFFFFF"   
#Title Label
title_label = ctk.CTkLabel(root,text="Contours detection & labeling",font=ctk.CTkFont("Arial",size=14,weight="bold"))
title_label.pack(pady=(28,20))
#Buttons Frame
btn_img = ctk.CTkButton(root,text="📷  Choose image",font=ctk.CTkFont(family="Arial", size=13),width=220,height=42,
    fg_color=GRAY_FG,hover_color=GRAY_HOVER,text_color=TEXT_COLOR,corner_radius=8,command=select_image)
btn_img.pack(pady=8)
btn = ctk.CTkButton(root,text="🎬  Start video",font=ctk.CTkFont(family="Arial", size=13),width=220,height=42,
    fg_color=GRAY_FG,hover_color=GRAY_HOVER,text_color=TEXT_COLOR,corner_radius=8,command=start_video)
btn.pack(pady=8)
btn_stop = ctk.CTkButton(root,text="⏸️  Stop/Resume video",font=ctk.CTkFont(family="Arial", size=13),width=220,height=42,
    fg_color=GRAY_FG,hover_color=GRAY_HOVER,text_color=TEXT_COLOR,corner_radius=8,command=stop_video)
btn_stop.pack(pady=8)
exit_btn = ctk.CTkButton(root,text="❌  Exit",font=ctk.CTkFont(family="Arial", size=13),width=220,height=42,
    fg_color="#FF4B4B",hover_color="#FF6E6E",text_color="#FFFFFF",corner_radius=8,command=exit_app) 
exit_btn.pack(pady=8)
root.mainloop()