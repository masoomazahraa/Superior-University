from flask import Flask,render_template,redirect,url_for
import requests
app= Flask(__name__)
def jokegenerator():
    url="https://icanhazdadjoke.com/"
    headers={"Accept":"application/json"}
    response=requests.get(url,headers=headers)
    if response.status_code==200:
        return response.json()['joke']
    return "Try again :D"
@app.route('/')
def home():
    joke=jokegenerator()
    return render_template('index.html',joke=joke)
@app.route('/new')
def new():
    return redirect(url_for('home'))
if __name__ == '__main__':
    app.run(debug=True)
