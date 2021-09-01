#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:04:55 2020

Run the GUI.

@author: imbroscb
"""

#%%
import tkinter as tk
from tkinter import filedialog
from Gui_vesicle_detection import VesAnalysisGui

        
tool=tk.Tk()

tool.geometry("800x600")
starting=VesAnalysisGui(tool)
tool.mainloop()

