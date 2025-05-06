from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form.get("msg")
    if "fever" in user_msg.lower():
        response = "It might be a sign of an infection. Please consult a doctor if it persists."
    elif "headache" in user_msg.lower():
        response = "Drink plenty of water and rest. If the headache is severe or recurring, seek medical advice."
    else:
        response = "I'm a simple medical assistant. Could you tell me more about your symptoms?"
    return jsonify({"response": response})
if __name__ == "__main__":
    app.run(debug=True)