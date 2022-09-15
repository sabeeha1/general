# from tkinter import *
#
# def run(input_dir, output_dir,frame_count, conf_thres, no_rotate,no_crop):
#     print("run(input_dir, output_dir,frame_count, conf_thres, no_rotate,no_crop)",input_dir, output_dir,frame_count, conf_thres, no_rotate,no_crop)
#
# main_window.mainloop()
#
#
import tkinter as tk
from tkinter import END


class Window_app(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        #Take entries from user in the form
        tk.Label(self, text="                        Input Directory").grid(row=0, column=0,sticky='e')
        tk.Label(self, text="                        ").grid(row=0, column=3, sticky='e')
        self.input_dir= tk.Entry(self, width=100, borderwidth=5)
        self.input_dir.grid(row=0, column=1)
        self.input_dir.insert(0, 'TT_videos')
        tk.Label(self, text=" Output Directory").grid(row=1, column=0,sticky='e')
        self.output_dir = tk.Entry(self, width=100, borderwidth=5)
        self.output_dir.grid(row=1, column=1)
        tk.Label(self, text="                            Frame count").grid(row=2, column=0,sticky='e')
        self.frame_count = tk.Entry(self, width=100, borderwidth=5)
        self.frame_count.insert(0,'50')
        self.frame_count.grid(row=2, column=1)
        tk.Label(self, text=" Image Size").grid(row=3, column=0,sticky='e')
        self.Image_size = tk.Entry(self, width=100, borderwidth=5)
        self.Image_size.insert(0,'640')
        self.Image_size.grid(row=3, column=1)

        tk.Label(self, text=" No Rotation").grid(row=4, column=0,sticky='e')
        self.no_rotate = tk.BooleanVar()
        tk.Radiobutton(self, text="True", variable=self.no_rotate, value=True).grid(row=4, column=1)
        tk.Radiobutton(self, text="False", variable=self.no_rotate, value=False).grid(row=4, column=2)

        tk.Label(self, text="    No crop").grid(row=5, column=0,sticky='e')
        self.no_crop = tk.BooleanVar()
        tk.Radiobutton(self, text="True", variable=self.no_crop, value=True).grid(row=5, column=1)
        tk.Radiobutton(self, text="False", variable=self.no_crop, value=False).grid(row=5, column=2)
        tk.Label(self, text="Current Status").grid(row=6, column=1)
        self.running_output = tk.Text(self)
        self.running_output.grid(row=7, column=1)
        self.button = tk.Button(self, bg='light gray', text="Submit Entries", command=self.on_click)
        self.button.grid(row=6, column=1)



    def on_click(self):
        input_dir,output_dir,frame_count,image_size,no_rotate,no_crop = self.input_dir.get(), self.output_dir.get(),\
                                                                        self.frame_count.get(), self.Image_size.get(),\
                                                                        self.no_rotate.get(), self.no_crop.get()

        if (len(input_dir) ==False) or (len(output_dir) ==False) or (len(frame_count) ==False) or (len(image_size) ==False):
            print("Entries are not complete")
        for i in range(100):
            current_text = run()
            # running_output.insert(END,f'{i}+{current_text}\n')
        return

def run():
    return("............")

main_window = Window_app()
main_window.geometry("1000x600")
main_window.title("TurtnTableExtractor")
for i in range(20):
    main_window.running_output.insert(END,f'{i}+___________________\n')
main_window.mainloop()

