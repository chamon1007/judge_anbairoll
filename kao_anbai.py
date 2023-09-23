import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

class_names = ["anpanman","baikinman","rollpanna"]

# Haar Cascade分類器の読み込み
@st.cache_resource
def load_MTCNN():
    return MTCNN()
    
# MTCNNモデルの初期化
detector = load_MTCNN()

# モデルの読み込み
@st.cache_resource
def load_model():
    model_path = r'my_model2'
    return tf.keras.models.load_model(model_path)

model = load_model()

def scale_to_height(img, height):
    """高さが指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))

    return dst

# Streamlit UI
st.title("Image Classification App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    cv2.resize(img, (160, 160))
    img = scale_to_height(img, 2000)
    
    # 顔の検出
    faces = detector.detect_faces(img)
    
    for face in faces:
        x, y, w, h = face['box']
        
        # 顔領域の取得
        face_img = img[y:y+h, x:x+w]
        
        # モデルの入力サイズにリサイズ
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        
        # 予測
        predictions = model.predict(face_img)
        # Softmaxを適用して確率を得る
        probabilities = tf.nn.softmax(predictions).numpy()
        # 各クラスの確率を表示
        #st.write(f"predictions:{predictions}")
        #for class_name, probability in zip(class_names, probabilities[0]):
        #    st.write(f"{class_name}: {probability*100:.2f}%")
        # 最大確率のインデックスを取得
        predicted_index = np.argmax(probabilities)
        
        # 予測結果の表示
        label = "{}".format(class_names[predicted_index])
        if predicted_index == 0:
            cv2.putText(img, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
        elif predicted_index == 1:
            cv2.putText(img, label, (x, y - 10),cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
        else:
            cv2.putText(img, label, (x, y - 10),cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Streamlitに画像を表示
    st.image(img, channels="BGR", use_column_width=True)
    
    