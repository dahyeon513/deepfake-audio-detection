# 🎤 Deepfake Audio Detection (Mel-Spectrogram + CNN)

딥러닝을 활용하여 **음성이 Real인지 Fake인지 자동으로 판별**하는 프로젝트입니다.  
Mel-spectrogram 기반 CNN 모델을 사용하여 오디오 데이터를 시각적으로 특징화하고 분류합니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🎧 Mel-spectrogram 기반 Feature 추출 | librosa를 사용해 음성을 Mel-spectrogram으로 변환 |
| 🧠 CNN Classification | Mel 이미지를 입력으로 Fake/Real 분류 |
| 🚀 End-to-End 학습 Notebook 제공 | `notebooks/train.ipynb` 에서 바로 학습 가능 |
| 🔍 모델 추론 스크립트 제공 | `src/inference.py` 로 새로운 음성 파일 추론 가능 |
| ✅ GPU 사용 가능 | TensorFlow / CUDA 지원 |

---

## 📁 프로젝트 구조
```
deepfake-audio-detection/
├─ notebooks/
│  └─ train.ipynb          # 학습용 Jupyter Notebook
├─ src/
│  └─ inference.py         # 음성 파일 추론 실행
├─ models/
│  └─ model.keras          # 학습된 모델
├─ data/
│  └─ README.md            # 데이터는 업로드하지 않음 (저작권 이슈)
└─ README.md
```


## 🛠 사용 기술

- Python  
- TensorFlow / Keras  
- Librosa (Mel-Spectrogram extraction)  
- NumPy / Pandas / Scikit-learn / Matplotlib  

---
