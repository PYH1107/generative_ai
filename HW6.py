import gradio as gr
import openai
import requests
import json
import os
import time
from typing import List, Tuple

# 設置 API 密鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

class AICharacter:
    def __init__(self, name: str, personality: str, model_type: str):
        self.name = name
        self.personality = personality
        self.model_type = model_type
        self.conversation_history = []

# 定義兩個尖酸刻薄的角色
character_a = AICharacter(
    name="毒舌評論家艾莉絲",
    personality="""你是一個極其尖酸刻薄的藝術評論家，名叫艾莉絲。你的特點：
    - 總是用最挑剔的眼光看待一切
    - 說話尖銳，喜歡挖苦諷刺
    - 認為自己品味高雅，看不起庸俗的東西
    - 經常使用反諷和挖苦的語調
    - 喜歡引用一些晦澀的文學或藝術典故來顯示自己的學識
    請保持這個人設，用尖酸刻薄但不失機智的方式回應對方。""",
    model_type="openai"
)

character_b = AICharacter(
    name="毒嘴哲學家博士",
    personality="""你是一個憤世嫉俗的哲學家，名叫博士。你的特點：
    - 對現代社會和人性極度悲觀
    - 說話犀利，喜歡戳穿虛偽
    - 經常引用尼采、叔本華等悲觀主義哲學家的觀點
    - 用理性的外衣包裝尖刻的批評
    - 喜歡質疑一切，包括對方的觀點
    請保持這個人設，用哲學式的毒舌方式回應對方。""",
    model_type="grok"
)

def call_openai_api(messages: List[dict]) -> str:
    """調用 OpenAI API"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API 錯誤: {str(e)}"

def call_grok_api(messages: List[dict]) -> str:
    """調用 Grok API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": messages,
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.8,
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Grok API 錯誤: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Grok API 錯誤: {str(e)}"

def get_ai_response(character: AICharacter, conversation_context: str) -> str:
    """獲取 AI 角色的回應"""
    messages = [
        {"role": "system", "content": character.personality},
        {"role": "user", "content": f"對話情境：{conversation_context}\n請以{character.name}的身份回應。"}
    ]
    
    if character.model_type == "openai":
        return call_openai_api(messages)
    else:
        return call_grok_api(messages)

def start_conversation(topic: str, rounds: int) -> List[Tuple[str, str]]:
    """開始對話"""
    conversation = []
    current_topic = topic
    
    for round_num in range(rounds):
        # 艾莉絲 (OpenAI) 先說
        if round_num == 0:
            context = f"話題：{topic}。請開始這個話題的討論。"
        else:
            context = f"話題：{topic}。對方剛才說：'{conversation[-1][1]}'。請回應。"
        
        alice_response = get_ai_response(character_a, context)
        
        # 博士 (Grok) 回應
        context = f"話題：{topic}。對方({character_a.name})剛才說：'{alice_response}'。請回應。"
        doctor_response = get_ai_response(character_b, context)
        
        conversation.append((
            f"🎭 {character_a.name}: {alice_response}",
            f"🧠 {character_b.name}: {doctor_response}"
        ))
        
        # 添加延遲避免 API 限制
        time.sleep(1)
    
    return conversation

def format_conversation(conversation: List[Tuple[str, str]]) -> str:
    """格式化對話結果"""
    formatted = "=== 🔥 毒舌雙雄對話實錄 🔥 ===\n\n"
    
    for i, (alice_msg, doctor_msg) in enumerate(conversation, 1):
        formatted += f"【第 {i} 回合】\n"
        formatted += f"{alice_msg}\n\n"
        formatted += f"{doctor_msg}\n\n"
        formatted += "─" * 50 + "\n\n"
    
    return formatted

def gradio_interface(topic: str, rounds: int) -> str:
    """Gradio 介面函數"""
    if not topic.strip():
        return "❌ 請輸入對話話題！"
    
    if rounds < 1 or rounds > 10:
        return "❌ 對話回合數請設置在 1-10 之間！"
    
    try:
        conversation = start_conversation(topic, rounds)
        return format_conversation(conversation)
    except Exception as e:
        return f"❌ 發生錯誤：{str(e)}"

# 創建 Gradio 介面
with gr.Blocks(
    theme="soft",
    title="🔥 毒舌雙雄對話機器人",
    css="""
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .gr-button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        border: none;
        color: white;
        font-weight: bold;
    }
    .gr-textbox {
        border-radius: 10px;
    }
    """
) as app:
    
    gr.Markdown("""
    # 🔥 毒舌雙雄對話機器人 🔥
    
    讓兩個極其尖酸刻薄的 AI 角色互相對話！
    
    **🎭 毒舌評論家艾莉絲** (GPT-4) VS **🧠 毒嘴哲學家博士** (Grok)
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(
                label="💬 對話話題",
                placeholder="例如：現代藝術的價值、社交媒體的影響、人工智能的未來...",
                lines=2
            )
            
            rounds_input = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="🔄 對話回合數"
            )
            
            start_button = gr.Button(
                "🚀 開始毒舌對話",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 📋 使用說明
            
            1. **輸入話題**：任何你想讓他們討論的話題
            2. **設置回合數**：建議3-5回合
            3. **點擊開始**：坐等精彩的毒舌對戰
            
            ### 🎭 角色介紹
            
            **艾莉絲**：尖酸刻薄的藝術評論家  
            **博士**：憤世嫉俗的哲學家
            
            ---
            *⚠️ 注意：對話內容純屬虛構，請勿當真！*
            """)
    
    output_text = gr.Textbox(
        label="🎪 對話結果",
        lines=20,
        max_lines=30,
        show_copy_button=True
    )
    
    start_button.click(
        fn=gradio_interface,
        inputs=[topic_input, rounds_input],
        outputs=output_text
    )
    
    gr.Markdown("""
    ---
    ### 🔧 技術說明
    - **OpenAI GPT-4**：驅動毒舌評論家艾莉絲
    - **Grok AI**：驅動毒嘴哲學家博士
    - **Gradio**：提供互動介面
    
    *Made with 💀 and a bit of sass*
    """)

if __name__ == "__main__":
    # 檢查環境變數
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 請設置 OPENAI_API_KEY 環境變數")
    if not os.getenv("GROK_API_KEY"):
        print("❌ 請設置 GROK_API_KEY 環境變數")
    
    print("🚀 啟動毒舌雙雄對話機器人...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )