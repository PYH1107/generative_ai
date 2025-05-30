import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入環境變數
load_dotenv()

class CustomE5Embedding(HuggingFaceEmbeddings):
    """自訂 E5 Embedding 類別，加入適當的前綴來優化檢索效果"""
    
    def embed_documents(self, texts):
        # 為文檔加上 "passage:" 前綴
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        # 為查詢加上 "query:" 前綴
        return super().embed_query(f"query: {text}")

class MathTeacherRAG:
    """細心溫柔的數學老師 RAG 系統"""
    
    def __init__(self):
        # 初始化 OpenAI 客戶端
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("請在 .env 檔案中設置 OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=api_key)
        self.retriever = None
        self.vector_db_path = "math_teacher_db"
        
        # 數學老師的人設設定
        self.system_prompt = """你是一位細心溫柔的數學老師，擅長用簡單易懂的方式解釋複雜的概念。

你的特質：
- 總是用 ELI5 (Explain Like I'm 5) 的方式來解釋，讓5歲小孩都能理解
- 耐心細心，從不急躁
- 溫柔親切，讓學生感到安心
- 善用比喻和生活化的例子
- 會鼓勵學生，讓他們有信心
- 如果概念複雜，會分步驟慢慢說明
- 經常確認學生是否理解

請用這樣的語氣和方式來回應學生的問題。"""

        self.prompt_template = """親愛的同學，根據我找到的教材內容：

{retrieved_content}

針對你的問題：{question}

讓老師用最簡單的方式來為你解釋～"""

    def load_pdf(self, pdf_path):
        """載入 PDF 檔案並建立向量資料庫"""
        try:
            print("📚 老師正在讀取教材...")
            
            # 載入 PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            print(f"✅ 成功載入 {len(documents)} 頁教材")
            
            # 分割文檔
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # 稍大的 chunk 保持內容完整性
                chunk_overlap=200,  # 足夠的重疊確保資訊不遺失
                separators=["\n\n", "\n", "。", ".", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
            print(f"📄 將教材分成 {len(split_docs)} 個小段落")
            
            # 建立向量資料庫
            print("🧠 正在建立知識索引...")
            embedding_model = CustomE5Embedding(
                model_name="intfloat/multilingual-e5-small"
            )
            
            vectorstore = FAISS.from_documents(split_docs, embedding_model)
            
            # 儲存向量資料庫
            vectorstore.save_local(self.vector_db_path)
            print("💾 知識索引建立完成！")
            
            # 設定檢索器
            self.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}  # 檢索最相關的4個段落
            )
            
            return True, "📚 教材載入成功！老師已經準備好回答你的問題了～"
            
        except Exception as e:
            error_msg = f"😅 抱歉，載入教材時遇到了問題：{str(e)}"
            print(error_msg)
            return False, error_msg

    def load_existing_db(self):
        """載入已存在的向量資料庫"""
        try:
            if os.path.exists(self.vector_db_path):
                print("📖 找到已建立的知識索引，正在載入...")
                embedding_model = CustomE5Embedding(
                    model_name="intfloat/multilingual-e5-small"
                )
                vectorstore = FAISS.load_local(
                    self.vector_db_path, 
                    embedding_model, 
                    allow_dangerous_deserialization=True
                )
                self.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                print("✅ 知識索引載入成功！")
                return True
            return False
        except Exception as e:
            print(f"⚠️ 載入知識索引時發生錯誤：{str(e)}")
            return False

    def answer_question(self, question):
        """回答學生問題"""
        if not self.retriever:
            return "😊 老師還沒有載入教材哦！請先上傳 PDF 檔案～"
        
        if not question.strip():
            return "💭 親愛的同學，請告訴老師你想問什麼問題呢？"
        
        try:
            # 檢索相關內容
            docs = self.retriever.get_relevant_documents(question)
            retrieved_content = "\n\n".join([doc.page_content for doc in docs])
            
            # 建立完整的 prompt
            full_prompt = self.prompt_template.format(
                retrieved_content=retrieved_content,
                question=question
            )
            
            # 呼叫 OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",  # 使用較好的模型來確保回答品質
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,  # 稍微增加創意性，讓解釋更生動
                max_tokens=1500   # 確保有足夠空間詳細解釋
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            return f"😅 抱歉，老師在思考答案時遇到了一點問題：{str(e)}\n請再試一次，或者用不同的方式問問題哦～"

def create_gradio_interface():
    """建立 Gradio 介面"""
    
    # 初始化 RAG 系統
    math_teacher = MathTeacherRAG()
    
    # 嘗試載入已存在的資料庫
    if math_teacher.load_existing_db():
        initial_message = "📚 老師已經準備好了！有什麼數學問題想問老師嗎？"
    else:
        initial_message = "👋 親愛的同學你好！老師需要先載入教材才能回答問題哦～"
    
    def upload_and_process_pdf(pdf_file):
        """處理 PDF 上傳"""
        if pdf_file is None:
            return "📁 請選擇一個 PDF 檔案上傳給老師～"
        
        success, message = math_teacher.load_pdf(pdf_file.name)
        return message
    
    def chat_with_teacher(message, history):
        """與數學老師對話"""
        if not message.strip():
            return history, ""
        
        # 獲得老師的回答
        teacher_response = math_teacher.answer_question(message)
        
        # 更新對話歷史
        history.append([message, teacher_response])
        
        return history, ""
    
    # 建立 Gradio 介面
    with gr.Blocks(
        title="溫柔數學老師", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🧮 溫柔的數學老師
        
        ### 👩‍🏫 嗨！我是你的數學老師
        我會用最簡單、最溫柔的方式來解釋數學概念，就像對5歲小朋友說話一樣～
        
        **使用方法：**
        1. 📤 先上傳你的數學教材 (PDF檔案)
        2. 💬 然後就可以問老師任何數學問題了！
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_upload = gr.File(
                    label="📚 上傳數學教材 (PDF)",
                    file_types=[".pdf"],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="📋 上傳狀態",
                    value=initial_message,
                    interactive=False,
                    lines=2
                )
        
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(
                    label="💬 與數學老師對話",
                    height=400,
                    placeholder="老師會在這裡回答你的問題～",
                    show_label=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="",
                        placeholder="有什麼數學問題想問老師呢？ (例如：什麼是分數？)",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("🚀 發送", scale=1, variant="primary")
        
        gr.Markdown("""
        ### 💡 小提示
        - 可以問概念解釋：「什麼是二次方程式？」
        - 可以問解題方法：「如何解這個方程式？」
        - 可以問應用問題：「這個概念在生活中怎麼用？」
        - 老師會用最簡單的方式解釋，不懂可以繼續問哦！
        """)
        
        # 事件綁定
        pdf_upload.change(
            fn=upload_and_process_pdf,
            inputs=[pdf_upload],
            outputs=[upload_status]
        )
        
        msg_input.submit(
            fn=chat_with_teacher,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        send_btn.click(
            fn=chat_with_teacher,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
    
    return demo

if __name__ == "__main__":
    # 檢查必要套件 - 使用正確的模組名稱
    package_check_map = {
        "gradio": "gradio",
        "openai": "openai", 
        "python-dotenv": "dotenv",  # 套件名 vs 模組名不同
        "langchain": "langchain",
        "langchain-community": "langchain_community",
        "sentence-transformers": "sentence_transformers",
        "faiss-cpu": "faiss",  # 套件名 vs 模組名不同
        "pypdf": "pypdf"
    }
    
    print("🔍 檢查必要套件...")
    missing_packages = []
    
    for package_name, module_name in package_check_map.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ 缺少以下套件，請執行：")
        print(f"pip install {' '.join(missing_packages)}")
        exit(1)
    
    print("✅ 所有套件都已安裝！")
    
    # 檢查 .env 檔案
    if not os.path.exists('.env'):
        print("⚠️ 找不到 .env 檔案，請建立 .env 並加入：")
        print("OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # 啟動應用
    print("🚀 啟動溫柔數學老師...")
    demo = create_gradio_interface()
    demo.launch(
        share=True,  # 產生公開連結
        server_name="0.0.0.0",  # 允許外部存取
        server_port=7860,  # 指定 port
        show_error=True  # 顯示錯誤資訊
    )