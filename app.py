# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import uuid
# print(os.path.dirname(os.path.realpath(__file__)) + "/../neural-chessboard")
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../neural-chessboard")
# from neural_chessboard.main import detect

# result = subprocess.run(['python3', 'main.py', 'detect', '--input="clayboards/IMG_1353.jpg"', '--output="clayboards_out/board_1353.jpg"'], capture_output=True, cwd="../neural-chessboard/")
# print(result)

from flask import Flask, request
import os, time


from eval_classify import eval_images
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

def wait_for_file():
  path_to_watch = os.getcwd() + '/../neural-chessboard/processed'
  before = dict ([(f, None) for f in os.listdir (path_to_watch)])

  while 1:
    time.sleep(1)
    after = dict ([(f, None) for f in os.listdir (path_to_watch)])
    added = [f for f in after if not f in before]

    if added:
        if len(added) > 1:
          print('ADDED MULTIPLE??')
        print("Added: ", ", ".join (added), '... Processing...')
        return (path_to_watch + '/' + added[0], added[0])


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'], name='upload-' + str(uuid.uuid1())[:8] + '.')

        # Move upload to neural-chessboard for processing
        os.rename('uploads/' + filename, os.getcwd() + '/../neural-chessboard/uploads/' + filename)

        file_url = photos.url(filename)

        # Wait for neural-chessboard to move it to its processed directory
        processed_filepath, processed_filename = wait_for_file()

        print('processed_filepath', processed_filepath)
        print('processed_filename', processed_filename)

        processed_filepath_new = "uploads/" + processed_filename
        os.rename(processed_filepath, processed_filepath_new)

        processed_file_url = photos.url(processed_filename)
        print('filename', filename)
        print('file_url', file_url)
        print('processed_file_url', processed_file_url)
        print('processed_filepath_new', processed_filepath_new)
        result = eval_images([processed_filepath_new])
        board_s = '<br/>'.join(result)

        return html + '<br/>' + board_s + + '<br/><br/><img src=' + processed_file_url + '>'
    return html


@app.route('/detect')
def index():
    board_s = ''
    # result = eval_images()
    # board_s = '\n'.join(result)
    return 'Whale, Hello there!3\n' + board_s

@app.route('/')
def hello_whale():
    return 'Whale, Hello there!1'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
