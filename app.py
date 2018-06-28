import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from evaluator import evaluate_comment
from train import train_model
import threading


class TrainerTab(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.learning_rate = tk.StringVar()
        tk.Label(self, text='Learning rate', anchor='w', height=2).pack(fill=tk.X)
        tk.Entry(self, textvariable=self.learning_rate).pack(fill=tk.X)

        self.epochs = tk.StringVar()
        tk.Label(self, text='Epochs', anchor='w', height=2).pack(fill=tk.X)
        tk.Entry(self, textvariable=self.epochs).pack(fill=tk.X)

        self.epoch_print_cost = tk.StringVar()
        tk.Label(self, text='Epoch print cost', anchor='w', height=2).pack(fill=tk.X)
        tk.Entry(self, textvariable=self.epoch_print_cost).pack(fill=tk.X)

        self.minibatch_size = tk.StringVar()
        tk.Label(self, text='Minibatch size', anchor='w', height=2).pack(fill=tk.X)
        tk.Entry(self, textvariable=self.minibatch_size).pack(fill=tk.X)

        self.hidden_units = tk.StringVar()
        tk.Label(self, text='Hidden units', anchor='w', height=2).pack(fill=tk.X)
        tk.Entry(self, textvariable=self.hidden_units).pack(fill=tk.X)

        tk.Frame(self, height=10).pack()
        self.train = tk.Button(self, text='Train', command=self.train_classifier)
        self.train.pack()
        tk.Frame(self, height=10).pack()
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X)

    def train_classifier(self):
        learning_rate = float(self.learning_rate.get())
        epochs = int(self.epochs.get())
        epoch_print_cost = int(self.epoch_print_cost.get())
        minibatch_size = int(self.minibatch_size.get())
        hidden_units = int(self.hidden_units.get())
        filters_size = [(3, 96), (5, 96), (7, 64)]
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = epochs
        self.train['state'] = tk.DISABLED
        def _logger(message):
            print(message)
            if 'epoch' in message:
                self.progress_bar['value'] += epoch_print_cost
            if 'FINISHED' in message:
                self.train['state'] = tk.NORMAL
        thread = threading.Thread(target=lambda: train_model(learning_rate,epochs,epoch_print_cost,minibatch_size,hidden_units,filters_size,_logger))
        thread.start()


class ClassifierTab(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, borderwidth=5)
        self.pack()

        self.scrolled_text = scrolledtext.ScrolledText(self)
        self.scrolled_text.pack(side='top')
        bottom_frame = tk.Frame(self, pady=5)
        self.evaluate = tk.Button(bottom_frame, text='Evaluate', command=self.run_evaluation)
        self.evaluate.pack(side='left')
        self.label = tk.Label(bottom_frame, text='Put a comment an press "Evaluate"')
        self.label.pack(side='right')
        bottom_frame.pack(side='bottom', fill=tk.X)

    def run_evaluation(self):
        comment = self.scrolled_text.get('1.0', tk.END)
        if len(comment) > 1:
            self.label.configure(text=str(evaluate_comment(comment)), fg='green')
        else:
            self.label.configure(text='You should write a comment. I cannot do nothing with blank text.', fg='red')


class App(ttk.Notebook):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        self.trainer = TrainerTab(self)
        self.classifier = ClassifierTab(self)
        
        self.add(self.trainer, text='Trainer')
        self.add(self.classifier, text='Classifier')


root = tk.Tk()
app = App(root)

root.title('Sentiment classifier')
root.minsize(300, 200)

root.mainloop()
