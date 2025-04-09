# opic-study

# 🎙️ 오픽 영어 말하기 연습 앱 (Streamlit)

AI 기반 오픽(영어 말하기 시험) 연습 웹앱입니다.  
사용자는 마이크로 직접 응답을 녹음하고, OpenAI의 Whisper 모델로 음성을 텍스트로 변환한 뒤,  
HuggingFace 문법 교정 모델을 통해 피드백을 받을 수 있습니다.

---

## 🚀 기능 요약

- 마이크로 직접 녹음 (streamlit-webrtc)
- Whisper(OpenAI)로 음성 → 텍스트 전사
- HuggingFace 모델로 문법/표현 교정
- Streamlit UI로 실시간 사용 가능

---
##📦 주요 라이브러리
OpenAI Whisper – 음성 인식

HuggingFace Transformers – 문법 피드백

streamlit-webrtc – 마이크 녹음 기능

Streamlit – 웹 UI


---
## 🖥️ 실행 방법 (로컬)

1. 이 저장소를 클론합니다:
```bash
git clone https://github.com/your-id/opic-ai-app.git
cd opic-ai-app
python -m venv venv
source venv/bin/activate        # 윈도우: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
