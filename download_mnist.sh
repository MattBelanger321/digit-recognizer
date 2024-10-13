kaggle competitions download -c digit-recognizer
mkdir ./data
unzip digit-recognizer.zip -d ./data
rm digit-recognizer.zip
mkdir test_pngs
mkdir train_pngs
python3 generate_png.py