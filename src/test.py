import numpy as np
from tensorflow.keras.models import load_model
import os
import librosa
from sklearn.preprocessing import LabelEncoder
np.complex = complex  
from tensorflow.keras.utils import to_categorical

# 사용자 설정 부분
MODEL_PATH = "model.keras"        # 저장된 모델 경로
OUTPUT_PATH = "dd2025_test_result.txt"     # 출력 파일 이름
# 경로 설정
test_metadata_path = os.path.expanduser('/2501ml_data/label/test_label.txt')
test_data_path = os.path.expanduser('/2501ml_data/test')


# Mel-spectrogram 추출 함수
def extract_mel_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y[:48000], sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec  # shape: (128, 94)

# 데이터셋 로딩 함수
def load_dataset(metadata_path, data_path):
    x_data, y_data, file_names = [], [], []
    with open(metadata_path, 'r') as f:
        for line in f:
            spk, file_name, _, _, label = line.strip().split(' ')
            wav_path = os.path.join(data_path, file_name)
            features = extract_mel_spectrogram(wav_path)
            x_data.append(features)
            y_data.append(label)
            file_names.append(file_name)
    return np.array(x_data), np.array(y_data), file_names

# Mel-spectrogram 데이터셋 생성
test_x, test_y, test_file_names = load_dataset(test_metadata_path, test_data_path)

# 채널 차원 추가 → (128, 400, 1)
test_x = test_x[..., np.newaxis]

# 라벨 인코딩 + 원-핫 인코딩
le = LabelEncoder()
le.fit(test_y)
y_test = le.transform(test_y)
y_test_oh = to_categorical(y_test)


# 1. 모델 불러오기
model = load_model(MODEL_PATH)

# 2. 예측
probs = model.predict(test_x)
preds = (probs >= 0.5).astype(int).flatten()

# 0/1을 fake/real로 변환
pred_labels = ['fake' if p == 0 else 'real' for p in preds]

# 예측 결과 파일로 저장
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for fname, label in zip(test_file_names, pred_labels):
        f.write(f"{fname} {label}\n")
