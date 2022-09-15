#
# import tkinter as tk
# #
# # window = tk()
# # window.title("Pressure Scanner")
# # window.geometry('300x300')
# #
# # tk.Label(window, text="Ports used, separated by spaces:").pack()
# # portlist = tk.Entry(window)
# # portlist.pack()
# #
# # tk.Label(window, text="Time to scan each port, in seconds:").pack()
# # timeduration = tk.Entry(window)
# # timeduration.pack()
# #
# # ##other stuff - includes a button to run the main data collection function
# #
# # # make the current port output box on the GUI window
# # tk.Label(window, text="Current Port").pack()
# # currentport = tk.Text(window)
# # currentport.pack()
# #
# # window.mainloop()
# #
# # # earlier in the function I take the input text from the "portlist" entry window on the GUI and create a list with the port numbers
# # length=10
# # for j in range(length):
# #     currentport.insert("just testing")  # ports[j]+1 because the addressing starts from 0 where the engraved port numbers on the module start from 1
#
#     # create address from port number
#     # send address to module
#     # receive signal, do math
#     # go back to top of loop to read the next port
#
# # from tkinter import *
# #
# # def run(input_dir, output_dir,frame_count, conf_thres, no_rotate,no_crop):
# #     print("run(input_dir, output_dir,frame_count, conf_thres, no_rotate,no_crop)",input_dir, output_dir,frame_count, conf_thres, no_rotate,no_crop)
# #
# # main_window.mainloop()
# #
# #
# import tkinter as tk
#
# class Window_app(tk.Tk):
#     def __init__(self):
#         tk.Tk.__init__(self)
#
#         #Take entries from user in the form
#         tk.Label(self, text="                        Input Directory").grid(row=0, column=0,sticky='e')
#         tk.Label(self, text="                        ").grid(row=0, column=3, sticky='e')
#         self.input_dir= tk.Entry(self, width=100, borderwidth=5)
#         self.input_dir.grid(row=0, column=1)
#         self.input_dir.insert(0, 'TT_videos')
#         tk.Label(self, text=" Output Directory").grid(row=1, column=0,sticky='e')
#         self.output_dir = tk.Entry(self, width=100, borderwidth=5)
#         self.output_dir.grid(row=1, column=1)
#         tk.Label(self, text="                            Frame count").grid(row=2, column=0,sticky='e')
#         self.frame_count = tk.Entry(self, width=100, borderwidth=5)
#         self.frame_count.insert(0,'50')
#         self.frame_count.grid(row=2, column=1)
#         tk.Label(self, text=" Image Size").grid(row=3, column=0,sticky='e')
#         self.Image_size = tk.Entry(self, width=100, borderwidth=5)
#         self.Image_size.insert(0,'640')
#         self.Image_size.grid(row=3, column=1)
#
#         tk.Label(self, text=" No Rotation").grid(row=4, column=0,sticky='e')
#         self.no_rotate = tk.BooleanVar()
#         tk.Radiobutton(self, text="True", variable=self.no_rotate, value=True).grid(row=4, column=1)
#         tk.Radiobutton(self, text="False", variable=self.no_rotate, value=False).grid(row=4, column=2)
#
#         tk.Label(self, text="    No crop").grid(row=5, column=0,sticky='e')
#         self.no_crop = tk.BooleanVar()
#         tk.Radiobutton(self, text="True", variable=self.no_crop, value=True).grid(row=5, column=1)
#         tk.Radiobutton(self, text="False", variable=self.no_crop, value=False).grid(row=5, column=2)
#         tk.Label(self, text="Current Status").grid(row=6, column=1)
#         self.running_output = tk.Text(self)
#         self.running_output.grid(row=7, column=1)
#         self.button = tk.Button(self, bg='light gray', text="Submit Entries", command=self.on_click)
#         self.button.grid(row=6, column=1)
#
#
#
#
#
#     def on_click(self):
#         input_dir,output_dir,frame_count,image_size,no_rotate,no_crop = self.input_dir.get(), self.output_dir.get(),\
#                                                                         self.frame_count.get(), self.Image_size.get(),\
#                                                                         self.no_rotate.get(), self.no_crop.get()
#
#         if (len(input_dir) ==False) or (len(output_dir) ==False) or (len(frame_count) ==False) or (len(image_size) ==False):
#             print("Entries are not complete")
#
#         current_text = run()
#         self.running_output.insert(END,current_text)
#         return
#
# def run():
#     for i in range(20):
#         return("I am {i}............")
#
# main_window = Window_app()
# main_window.geometry("1000x600")
# main_window.title("TurtnTableExtractor")
# main_window.mainloop()
#
from tkinter import *
import time


def collectdata():
    ports = [1, 2, 3, 4, 5]
    length = len(ports)
    for n in range(length):
        currentport.insert(END, f'{ports[n]}\n')
        time.sleep(2)


window = Tk()
window.title("Pressure Scanner")
window.geometry('300x300')

button = Button(window, text="Collect Data", command=collectdata).pack()

Label(window, text="Output below should update every loop cycle")

Label(window, text="Current Port").pack()
currentport = Text(window)
currentport.pack()

window.mainloop()