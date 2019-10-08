# chessboard-image-to-fen
Takes an overhead picture of a chessboard and converts it to FEN notation for engine analysis

# Running server
screen -S chess_flask

screen -x -R chess_flask

cd ~/chessboard-image-to-fen
mkdir saved_objects
conda activate tf-cpu
psudo() { sudo env PATH="$PATH" "$@"; }
git pull
psudo python3 app.py



screen -x -R file_checker
cd ~/neural-chessboard/
conda activate tf-cpu
git pull
python3 file_checker.py

# steps to annotate new training data:
  take photos
  put photos in neural-chessboard/chess_1857 and edit detect_mass.sh
  run ./detect_mass.sh
  delete all malformed chess boards in output folder
  move files that exist in chessboard-image-to-fen/clayboards_out/ to chessboard-image-to-fen/clayboards_out_processed/
  move files from neural-chessboard output folder to chessboard-image-to-fen/clayboards_out/
  label all images in labeled_boards.py
  run create_training_data_from_boards.py
  ensure new images exist in training folders and validate images there to ensure they're good
  move files that exist in chessboard-image-to-fen/clayboards_out/ to chessboard-image-to-fen/clayboards_out_processed/

# commands to setup ec2 instance:
sudo yum install git -y
y

cd ~

git clone git@github.com:cschubiner/chessboard-image-to-fen.git
mkdir saved_objects

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


## Google Cloud AutoML Vision
make sure you run these commands in the google cloud shell of the project that has autoML in it

PROJECT=$(gcloud config get-value project) && BUCKET="${PROJECT}-vcm"
gsutil mb -p ${PROJECT} -c regional -l us-central1 gs://${BUCKET}

gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
  --role="roles/ml.admin"
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
  --role="roles/storage.admin"
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
  --role="roles/serviceusage.serviceUsageAdmin"

## Notes

If you get this error with image_features, follow steps here:
https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org

## Links
https://chsasank.github.io/deep-learning-image-features.html
https://github.com/chsasank/image_features

