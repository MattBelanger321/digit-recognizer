kaggle competitions download -c digit-recognizer
mkdir .\data
Expand-Archive -Path "digit-recognizer.zip" -DestinationPath ".\data"
del digit-recognizer.zip
mkdir test_pngs
mkdir train_pngs
python generate_png.py