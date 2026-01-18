import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re
from typing import Tuple

st.set_page_config(
    page_title="EXAONE 1.2B LLM",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .message-user {
        background: #38bdf8;
        color: #0f172a;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
    }
    .message-assistant {
        background: rgba(56, 189, 248, 0.15);
        border-left: 3px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    st.info("🚀 경량화 모델을 로드 중입니다...")
    
    try:
        MODEL_ID = "paddacoco/exaone-1.2b-lora"
        
        st.write(f"📥 로딩 중: {MODEL_ID}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cpu",
            trust_remote_code=True
        )
        base_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        
        st.success("✅ 모델 로드 완료!")
        return base_model, base_tokenizer
    
    except Exception as e:
        st.error(f"❌ 모델 로드 실패!")
        st.error(f"에러: {str(e)}")
        st.warning("확인 사항:")
        st.write("1. 모델 리포가 Public인가?")
        st.write("2. 모델 파일이 완전히 업로드됐나?")
        st.write("3. 모델 ID가 맞나? (paddacoco/exaone-1.2b-lora)")
        return None, None

def extract_math_answer(text: str) -> str:
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        numbers = re.findall(r"\d+", answer_part)
        return numbers[0] if numbers else answer_part[:50]
    numbers = re.findall(r"\d+", text)
    return numbers[-1] if numbers else "답 없음"

def generate_response(model, tokenizer, prompt: str, max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> Tuple[str, float]:
    if model is None or tokenizer is None:
        return "모델이 로드되지 않았습니다.", 0.0
    
    start_time = time.time()
    
    chat = [{'role': 'user', 'content': prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    with __import__('torch').no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    processing_time = time.time() - start_time
    return response.strip(), processing_time

st.markdown("""
# 🤖 EXAONE 1.2B LLM 대시보드
### LoRA 경량화 모델
**LG AIMERS 8th Cohort 해커톤**
""")

with st.sidebar:
    st.markdown("## ⚙️ 설정")
    mode = st.radio("📌 모드", ["일반 채팅", "수학 풀이"], key="mode_selector")
    
    st.markdown("---")
    temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1)
    top_p = st.slider("🎯 Top-P", 0.1, 1.0, 0.9, 0.05)
    max_length = st.slider("📏 길이", 50, 512, 256, 50)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("모델", "EXAONE 1.2B")
    with col2:
        st.metric("최적화", "LoRA")

tab1, tab2 = st.tabs(["💬 채팅", "📊 통계"])

with tab1:
    st.markdown("### 챗봇과 대화하세요")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model, st.session_state.tokenizer = load_model()
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="message-user"><strong>👤 You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message-assistant"><strong>🤖 AI:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("메시지:", placeholder="질문을 입력하세요", key="user_input")
    with col2:
        send_button = st.button("전송", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("응답 생성 중..."):
            if "수학" in mode:
                system_prompt = "수학 문제를 단계별로 풀어주세요.\n최종 답은 #### 뒤에 적어주세요."
                full_prompt = f"{system_prompt}\n\n질문: {user_input}"
            else:
                full_prompt = user_input
            
            response, processing_time = generate_response(
                model, tokenizer, full_prompt, max_length, temperature, top_p
            )
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("길이", f"{len(response)} 자")
            with col2:
                st.metric("시간", f"{processing_time:.2f}초")
            with col3:
                if "수학" in mode:
                    answer = extract_math_answer(response)
                    st.metric("답", answer)
        
        st.rerun()
    
    if not st.session_state.messages:
        st.info("👋 EXAONE 1.2B에 오신 것을 환영합니다!")

with tab2:
    st.markdown("### 📊 모델 성능")
    
    if st.session_state.messages:
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("대화 수", len(st.session_state.messages) // 2)
        with col2:
            avg = sum(len(m["content"]) for m in assistant_msgs) / len(assistant_msgs) if assistant_msgs else 0
            st.metric("평균", f"{int(avg)} 자")
        with col3:
            total = sum(len(m["content"]) for m in assistant_msgs)
            st.metric("총 생성", f"{total:,} 자")

st.markdown("---")
st.markdown("**EXAONE 1.2B - LG AIMERS 8th Cohort 해커톤**")
