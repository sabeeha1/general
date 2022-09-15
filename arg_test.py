
import argparse
import tkinter
import tkinter as tk

def run(
        input_dir,  # save results to project/name
        output_dir ,  # save results to project/name
        background= '50',
        image_size = '200',
        ):
    print(output_dir,input_dir,background,image_size)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir',default ='exis', help = "first number")
    # parser.add_argument('--output_dir',default ='yy', help = "second number")
    # parser.add_argument('-b','--background',default = '50', help = "Background contrast")
    # parser.add_argument('-s','--image_size',default= '200', help = "Final image size")
    # opt = parser.parse_args()

    parser.add_argument('input_dir', help="first number")
    parser.add_argument('output_dir', help="second number")
    parser.add_argument('-b', '--background', default='50', help="Background contrast")
    parser.add_argument('-s', '--image_size', default='200', help="Final image size")
    opt = parser.parse_args()
    # first= opt.input_dir
    # print(first,"**********************")
    return opt

def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)




# arr = ["111111111111111111111",'222222222222222222222','3333333333333333333333','444444444444444444444','555555555555555555555']
# window = tk.Tk()
# window.title("GUI")
# label = tk.Label(window,text = arr[0]).pack()
# window.mainloop()




'''# Python program to demonstrate
# command line arguments
 
 
import sys
 
# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)
 
# Arguments passed
print("\nName of Python script:", sys.argv[0])
 
print("\nArguments passed:", end = " ")
for i in range(1, n):
    print(sys.argv[i], end = " ")
     
# Addition of numbers
Sum = 0
# Using argparse module
for i in range(1, n):
    Sum += int(sys.argv[i])
     
print("\n\nResult:", Sum)'''