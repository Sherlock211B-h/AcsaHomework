from sentiAnalysisModel import *
from flask import Flask, render_template, request
from sentiAnalysisModel import SAModel as model

app = Flask(__name__)

def analyze_sentiment(saModel, text):
    print("Running test on the text...")
    positive_prob = saModel.predict_sentiment(text)
    if positive_prob > 0.5:
        return "分析完成，该评论是正面评论。"
    return "分析完成，该评论是负面评论。"

def test_model(saModel):
    print("Running test on the dataset...")
    accuracy = saModel.test_set() * 100
    result = f"测试完成（40000条评论数据），测试集正确率为{accuracy:.2f}%"
    return result

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/execute_function", methods=["POST"])
def execute_function():
    saModel = model()
    data = request.json
    function_name = data["function_name"]
    input_text = data["input_text"]

    result = ""
    if function_name == "analyze":
        result = analyze_sentiment(saModel, input_text)
    elif function_name == "test":
        result = test_model(saModel)

    return result

if __name__ == "__main__":
    app.run(debug=True)
