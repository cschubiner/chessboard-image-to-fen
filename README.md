# chessboard-image-to-fen
Takes an overhead picture of a chessboard and converts it to FEN notation for engine analysis

# Running server
psudo() { sudo env PATH="$PATH" "$@"; }

psudo python3 app.py

# commands to setup ec2 instance:
sudo yum install git -y
y

cd ~

git clone git@github.com:cschubiner/chessboard-image-to-fen.git


git clone git@github.com:cschubiner/neural-chessboard.git
yes


conda create -n tf-cpu tensorflow
y

conda activate tf-cpu
y

cd neural-chessboard

pip3 install --no-cache-dir  -U git+https://github.com/chsasank/image_features.git

pip3 install --no-cache-dir  -r requirements.txt

pip3 install  --no-cache-dir  keras

pip3 install  --no-cache-dir  pandas

pip3 install  --no-cache-dir  flask

python3 -m pip install --no-cache-dir  -U git+https://github.com/chsasank/image_features.git

python3 -m pip install --no-cache-dir  -r requirements.txt

python3 -m pip install  --no-cache-dir  keras

python3 -m pip install  --no-cache-dir  pandas

python3 -m pip install  --no-cache-dir  flask

conda install opencv
y

conda install keras
y

conda install pyclipper
y

conda install numpy
y

conda install scipy
y

python3 -m pip install pyclipper

conda install matplotlib
y

conda install --file requirements.txt
y

conda install scikit-learn
y

conda install pandas
y

pip3 install --no-cache-dir  -U git+https://github.com/chsasank/image_features.git

python3 -m pip install --no-cache-dir  -U git+https://github.com/chsasank/image_features.git

cd ~/neural-chessboard
python3 main.py detect --input="clayboards/IMG_1353.jpg" --output="clayboards_out/board_1353.jpg"

cd ~/chessboard-image-to-fen/
python3 eval_classify.py



## Notes

If you get this error with image_features, follow steps here:
https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org

## Links
https://chsasank.github.io/deep-learning-image-features.html
https://github.com/chsasank/image_features

