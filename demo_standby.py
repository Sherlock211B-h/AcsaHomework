import tkinter as tk
from tkinter import messagebox
from sentiAnalysisModel import SAModel as Model
from sentiAnalysisModel import *

class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中文文本情感分析系统")
        self.root.geometry('%dx%d' % (800, 600))
        self.model = Model()
        self.create_widgets()

    def create_widgets(self):
        self.intro_label = tk.Label(self.root, text="中文文本情感分析系统（以汽车评价为例）", font=('粗体', 25))
        self.intro_label.pack(pady=10)

        self.text_entry = tk.Text(self.root, font=("Arial", 15), height=10, width=60)
        self.text_entry.pack()

        self.text_entry.insert("1.0", "请输入对汽车的评价...")
        self.text_entry.bind("<FocusIn>", self.clear_entry)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.analyze_button = tk.Button(self.button_frame, width=18, height=2, text="文本分析", bg='white', font=("宋", 15), command=self.analyze_sentiment)
        self.analyze_button.pack(side="left", padx=5)

        self.test_button = tk.Button(self.button_frame, width=18, height=2, text="测试集测试", bg='white', font=("宋", 15), command=self.test_model)
        self.test_button.pack(side="left", padx=5)

        self.result_label = tk.Label(self.root, text="", font=("宋", 15), foreground="green", wraplength=400)
        self.result_label.pack()

    def clear_entry(self, event):
        if self.text_entry.get("1.0", tk.END).strip() == "请输入对汽车的评价...":
            self.text_entry.delete("1.0", tk.END)

    def analyze_sentiment(self):
        input_text = self.text_entry.get("1.0", tk.END).strip()
        if input_text:
            positive_prob = self.model.predict_sentiment(input_text)
            messagebox.showinfo("提示", "分析完成，点击确定刷新结果。")
            if positive_prob > 0.5:
                result = "分析完成，该评论是正面评论。"
            else:
                result = "分析完成，该评论是负面评论。"
            self.result_label.config(text=result)
        else:
            messagebox.showerror("错误", "请输入文本。")

    def test_model(self):
        messagebox.showinfo("提示", "正在测试测试集，请稍后...")
        accuracy = 100 * self.model.test_set()
        self.result_label.config(text=f"测试完成（30000条评论数据），测试集准确率为{accuracy:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()