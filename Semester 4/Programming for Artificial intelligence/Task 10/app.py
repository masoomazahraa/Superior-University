from flask import Flask, render_template, request, jsonify
app=Flask(__name__)
def get_response(userinput):
    userinput = userinput.lower()
    if 'hours'in userinput or "open" in userinput:
        return "We're open every day from 10am to 11 pm."
    elif "menu" in userinput or "dishes" in userinput:
        return "We serve Italian,Continental, and a variety of desserts. check our full menu on our website."
    elif "reservation" in userinput or "book" in userinput:
        return "You can reserve a table by calling us or using our online reservation system."
    elif "location" in userinput or "where" in userinput:
        return "We're located at 123 Foodie Street, Flavor Town."
    elif "contact" in userinput or "phone" in userinput:
        return "You can reach us at (555) 123-4567 or email us at contact@flavortownbistro.com."
    else:
        return "Sorry, I didn't understand. Please ask about hours, menu, reservations, location, or contact info."
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get", methods=["POST"])
def chatbot_response():
    userinput = request.form["msg"]
    response = get_response(userinput)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
