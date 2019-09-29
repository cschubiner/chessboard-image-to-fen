# -*- coding: utf-8 -*-
import os

from flask import Flask, request


from eval_classify import run
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Photo Upload</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=photo>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        file_url = photos.url(filename)
        return html + '<br><img src=' + file_url + '>'
    return html


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
