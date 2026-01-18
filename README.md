\# 🤖 EXAONE 1.2B LLM \- LoRA 경량화

\*\*LG AIMERS 8th Cohort 해커톤 프로젝트\*\*

\#\# 🎯 성과  
\- ✅ 학습 파라미터 99.1% 감소 (1.28B → 1.6M)  
\- ✅ T4 GPU에서 안정적 학습  
\- ✅ 추론 속도: 0.07 it/s  
\- ✅ GSM8K-Ko 데이터 기반 학습

\#\# 🚀 기능  
\- 일반 채팅 (한국어)  
\- 수학 풀이 (단계별)  
\- 실시간 파라미터 조정

\#\# 📊 기술 스택  
\- 모델: EXAONE 4.0 1.2B  
\- 최적화: 4-bit Quantization \+ LoRA  
\- 프레임워크: Streamlit \+ PyTorch \+ Hugging Face

\#\# 설치  
\`\`\`bash  
pip install \-r requirements.txt  
streamlit run streamlit\_app.py

