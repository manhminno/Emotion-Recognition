# Emotion-Recognition
- Emotion recognition using haar-like and CNN model.
- Using haar-like feature to detect face in image
- After detecting face, put it into CNN network to recognize emotions

## 1. Load data:
- Load data from csv file, u can download file here: https://drive.google.com/open?id=1-wHLDkS_CUgKXF9lWcbvimKx917LNxVH
- Run code to read data from csv file:
```
  python load_dataset.py
```
## 2. Train model:
```
  python train.py
```
## 3. Enjoy the result:
```
* Detect from webcam: python main.py --mode Webcam
* Detect from image: python main.py --mode Image --path (Img_file)
```
## 4. Demo: 
#### *Here I am using Vietnamese, you can replace emotions with your language in label2id*
<p align="center"> <img src="https://github.com/manhminno/Emotion-Recognition/blob/master/Result.jpg"></p>
<p align="center"> <img src="https://github.com/manhminno/Emotion-Recognition/blob/master/Demo.jpg"></p>
