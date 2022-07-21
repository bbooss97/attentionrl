from tkinter import *
from random import *
import torch
import threading

class GridAttention():
    def rect(self,x,y,color):
        width=height=self.step
        self.canvas.create_rectangle(x, y, x+width, y+height, fill=color)
    def reset(self):
        for i in range(0,self.height,self.step):
            for j in range(0,self.width,self.step):
                self.rect(j,i,"white")
                
    def start(self):

        self.window = Tk()
        self.window.title('Attention')
        self.width = 400
        self.height = 400
        self.step=int(self.width/16)
        self.canvas = Canvas(self.window, background='white', width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)
        # create_grid(self.window)
        self.reset()
        self.window.mainloop()
    def __init__(self):
        threading.Thread(target=self.start).start()
        print("creato")
        pass
    def draw(self,features):
        threading.Thread(target=self.drawThread,args=(features,)).start()
        
    def drawThread(self,features):
        f=(features*16).reshape(-1,2).int().tolist()
        for i in f:
            self.rect(i[1],i[0],"red")
#g=GridAttention()
