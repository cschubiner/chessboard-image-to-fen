from flask import Flask

from eval_classify import run

app = Flask(__name__)


@app.route('/chess')
def index():
    result = run()
    board_s = '\n'.join(result)
    return 'Whale, Hello there!3\n' + board_s

@app.route('/')
def hello_whale():
    return 'Whale, Hello there!1'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
