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
st.title("正義の心 Detector")
uploaded_file = st.file_uploader("画像をアップして判定", type=["jpg", "jpeg", "png"])
st.write("アンパンマン：圧倒的、正義の味方")
st.write("バイキンマン：正義の敵、いたずら好き")
st.write("ロールパンナ：優しさと悪 2つの心の間で揺れ動く葛藤")

if uploaded_file is not None:
    # 画像の読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    cv2.resize(img, (160, 160))
    img = scale_to_height(img, 2500)
    
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
        font_scale = 5
        thickness = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # テキストの背景の座標を計算します。
        padding = 20  # 任意のパディングを設定します。
        pt1 = (x, y)  # テキストの左上の点
        pt2 = (x + text_width + padding, y - text_height - padding)  # テキストの右下の点
        
        # 色を選択します。
        color = (0, 0, 0)  # 黒で背景を塗りつぶします。
        if predicted_index == 0:
            text_color = (0, 0, 255)  # テキストの色を赤に設定します。
        elif predicted_index == 1:
            text_color = (255, 0, 0)  # テキストの色を青に設定します。
        else:
            text_color = (0, 255, 0)  # テキストの色を緑に設定します。
        
        # 背景を描画します。
        cv2.rectangle(img, pt1, pt2, color, -1)  # -1 は塗りつぶしを意味します。
        
        # テキストを描画します。
        cv2.putText(img, label, (x + padding, y - padding), font, font_scale, text_color, thickness)
        
        # 最後に、元の矩形を描画します。
        cv2.rectangle(img, (x, y), (x + w, y + h), text_color, 2)

    # Streamlitに画像を表示
    st.image(img, channels="BGR", use_column_width=True)
    
    
