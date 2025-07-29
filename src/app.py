
import streamlit as st
import os
import datetime
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. API 키 설정 ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일이 올바르게 생성되었는지 확인해주세요.")
    st.stop()

# --- 2. RAG 체인 로딩 함수 (캐시 사용) ---
@st.cache_resource
def load_rag_chain():
    VECTOR_STORE_DIR = 'data/03_vector_store'
    embeddings_model = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    prompt_template = """
    <지시문>
    당신은 회계감사 전문가입니다.
    <Context>와 <질문>을 바탕으로 답변을 생성하세요.
    답변은 반드시 아래 <답변형식>을 따라야 하며, '답변 근거란', '관련 교재 내용' 같은 문구는 절대 출력하면 안 됩니다.
    '###' 기호는 사용하지 마세요.
    한국어로 작성해 주세요
    </지시문>

    <답변형식>
    ### 정답:
    질문에 대한 명확한 결론을 내립니다. 물어보는 것에 대해 짧게 대답합니다.
    여부를 물었을 때는 O/X로 대답합니다.
    적절한가? 하고 물었을 때는 예, 아니오로 대답합니다.
    잘못된 감사절차를 물었을 때는 본문에서 잘못된 감사절차 내용을 찾습니다.
    수행하여야 할 절차를 물었을 때는 구체적인 사례보다는 context에서 절차 관련된 언급이 있을 경우 기준서를 최대한 준용합니다.
    몇 가지를 물어보는지 파악하고 질문에 맞는 답안을 구성합니다. (2가지를 물었을 경우 2가지로 대답합니다. 첫째 사항/절차(물음에서 물어본것)는, 둘째 사항/절차(물음에서 물어본것)는, 이런 식으로 대답합니다.)

    ### 판단 근거:
    서술형으로 작성합니다.
    구체적인 기준이나 문구를 인용합니다
    최대한 감사기준서나 절차는 제공된 context의 용어와 표현을 그대로 사용합니다.
    사례에서 제공된 문제점을 지적할 수 있습니다.
    다만, context의 회계감사기준, 윤리기준 등 다양한 내용을 근거로 위배된 부분이 있는지 파악합니다.
    제공된 질문에서 잘못된 부분을, context를 근거로 평가해야 합니다.
    당신은 30년 이상의 숙련된 회계사로, 모든 회계감사기준을 명백히 이해하고 있습니다.
    당신에게 제시된 질문의 text에서 감사절차가 적절/부적절한지 판단한 이유를 근거를 들어 말해야 합니다.

    ---
    Context:
    {context}

    Question:
    {question}

    Answer(Korean):
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
        chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
    )
    return rag_chain

# --- 3. 대화 내용을 HTML로 변환하는 함수 ---
def generate_html(history):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content = f"""
    <!DOCTYPE html><html><head><title>회계감사 AI 대화 기록</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 800px; margin: auto; border: 1px solid #ddd; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .qa-pair {{ border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 20px; }}
        .question-block {{ background-color: #e1f5fe; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #0288d1; }}
        .answer-block {{ background-color: #f1f8e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #7cb342;}}
        h1, h2, h3 {{ color: #0277bd; }}
        h1 {{ text-align: center; }}
        .timestamp {{ text-align: right; color: #757575; font-size: 0.9em; }}
        .save-timestamp {{ text-align: center; color: #757575; margin-bottom: 20px; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 14px; }}
    </style></head><body><div class="container">
        <h1>회계감사 AI 대화 기록</h1>
        <p class="save-timestamp">저장 시각: {timestamp}</p>
    """
    for item in history:
        # Markdown을 HTML 줄바꿈으로 변경
        formatted_result = item['result'].replace('\n', '<br>')
        # 질문 시간을 HTML에 추가
        query_timestamp = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        html_content += f"""
        <div class="qa-pair">
            <p class="timestamp">질문 시각: {query_timestamp}</p>
            <div class='question-block'>
                <h3>질문</h3>
                <pre>{item['query']}</pre>
            </div>
            <div class='answer-block'>
                <h3>AI 답변</h3>
                {formatted_result}
            </div>
        </div>
        """
    html_content += "</div></body></html>"
    return html_content

# --- 4. Streamlit 웹 UI 구성 ---
st.set_page_config(page_title="회계감사 RAG AI", layout="wide")
st.title("🤖 회계감사 문제풀이 AI 어시스턴트")

# --- st.session_state 초기화 ---
# 'history': 대화 기록 저장
# 'show_history': 대화 기록 보기/숨기기 상태 저장
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# --- 사이드바 UI ---
with st.sidebar:
    st.header("메뉴")

    # 1. 대화 기록 보기 버튼 (상태 토글)
    if st.button("대화 기록 보기/숨기기"):
        st.session_state.show_history = not st.session_state.show_history

    # 2. 대화 기록 저장 버튼
    if st.session_state.history:
        html_str = generate_html(st.session_state.history)
        file_name = f"audit_conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        st.download_button(
            label="대화 내용 HTML로 저장",
            data=html_str.encode('utf-8'),
            file_name=file_name,
            mime='text/html'
        )

    # 3. 대화 기록 초기화 버튼
    if st.button("대화 기록 초기화"):
        st.session_state.history = []
        st.session_state.show_history = False # 기록 초기화 시 보기 상태도 초기화
        st.success("대화 기록이 초기화되었습니다.")


# --- 메인 화면 UI ---
try:
    rag_chain = load_rag_chain()

    st.caption("회계감사 시험 문제를 입력하세요")
    # 입력란 2개로 분리
    # st.markdown을 사용하여 원하는 크기의 제목을 표시합니다.
    st.markdown("### 문제")
    # 기존 text_area의 label은 숨깁니다 (label_visibility="collapsed").
    problem_context = st.text_area(
        "problem_input_area", # 위젯을 구분하기 위한 고유 키
        height=80,
        placeholder="여기에 문제 내용을 입력하세요",
        label_visibility="collapsed"
    )

    # 두 번째 입력창도 동일하게 수정합니다.
    st.markdown("### 물음")
    specific_question = st.text_area(
        "question_input_area", # 위젯을 구분하기 위한 고유 키
        height=300,
        placeholder="여기에 물음 내용을 입력하세요",
        label_visibility="collapsed"
    )

    if st.button("답변 생성하기"):
        # (이하 로직은 이전과 동일)
        if problem_context and specific_question:
            # 두 입력 내용을 합쳐서 하나의 질문으로 구성
            full_query = f"문제: {problem_context}\n\n물음: {specific_question}"

            with st.spinner('AI가 감사 기준서를 검토하며 답변을 생성 중입니다...'):
                response = rag_chain.invoke(full_query)

                # 대화 기록에 질문, 답변, 참고자료, 시간 정보 모두 저장
                st.session_state.history.append({
                    'query': full_query,
                    'result': response['result'],
                    'sources': response['source_documents'],
                    'timestamp': datetime.datetime.now()
                })
        else:
            st.warning("문제와 물음 내용을 모두 입력해주세요!")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")

# --- 최신 답변을 메인 화면에 즉시 표시 ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 답변")
    latest_item = st.session_state.history[-1]

    response_text = latest_item['result']  # AI가 생성한 전체 텍스트
    parts = response_text.split("### 판단 근거:") # AI 답변을 '### 판단 근거:' 기준으로 분리
    answer_part = parts[0].replace("### 정답:", "").strip()  # '정답' 부분 텍스트 정리

    st.write(answer_part)
    st.markdown("---") # 구분선

    # '판단 근거'가 있는 경우
    if len(parts) > 1:
        reason_part = parts[1].strip()
        # 2. [판단 근거] 헤더와 내용을 올바른 문법으로 표시
        st.markdown("### **[판단 근거]**")
        st.write(reason_part)

    # 3. 참고 자료 표시
    with st.expander("📚 참고 자료 (AI가 검토한 원문)"):
        for doc in latest_item['sources']:
            st.markdown(f"**📖 {doc.metadata.get('source', '출처 없음')}**")
            st.markdown(f"> {doc.page_content}")
            st.markdown("---")


# --- '대화 기록 보기'를 눌렀을 때만 전체 기록 표시 ---
if st.session_state.show_history:
    st.markdown("---")
    st.header("최근 질문과 답변")

    if not st.session_state.history:
        st.info("표시할 대화 기록이 없습니다.")
    else:
        # 오래된 순서대로 정렬하여 표시
        for item in st.session_state.history:
            query_time = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            with st.container():
                st.markdown(f"<small style='color:grey;'>질문 시각: {query_time}</small>", unsafe_allow_html=True)
                # 질문 블록
                with st.chat_message("user", avatar="❓"):
                    st.text(item['query'])
                # 답변 블록
                with st.chat_message("assistant", avatar="🤖"):
                    # '정답'과 '판단 근거'를 분리해서 구조적으로 보여줌
                    parts = item['result'].split("### 판단 근거:")
                    if len(parts) == 2:
                        answer_part = parts[0].replace("### 정답:", "").strip()
                        reason_part = parts[1].strip()
                        st.markdown("**정답**")
                        st.markdown(answer_part)
                        st.markdown("**판단 근거**")
                        st.markdown(reason_part)
                    else: # 분리 실패 시 전체 결과 표시
                        st.markdown(item['result'])
                st.markdown("---") # 세트별 구분선
