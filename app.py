from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'Whale, Hello there!'

@app.route('/')
def hello_whale():
    return 'Whale, Hello there!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
