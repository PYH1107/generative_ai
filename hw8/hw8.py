import gradio as gr
import json
import random
from typing import Dict, List, Tuple

class RelationshipClinic:
    def __init__(self):
        self.questionnaire_data = {}
        self.ai_responses = {}
        self.current_step = 1
        
    def reset_data(self):
        """重置所有數據"""
        self.questionnaire_data = {}
        self.ai_responses = {}
        self.current_step = 1
        
    def calculate_infatuation_score(self) -> float:
        """計算暈船指數 (1-10分)"""
        score = 0
        
        # 關係狀態計分
        relationship_scores = {
            '完全陌生人，只是單方面關注': 3.5,
            '認識但不熟，偶爾有互動': 2.0,
            '朋友關係，經常聊天': 1.5,
            '曖昧階段，關係模糊': 3.0,
            '正在交往中': 1.0
        }
        score += relationship_scores.get(self.questionnaire_data.get('relationship', ''), 0)
        
        # 聯絡頻率計分
        contact_scores = {
            '每天都會傳訊息': 3.0,
            '2-3天一次': 2.0,
            '大約一週一次': 1.0,
            '很少主動聯絡': 0.5
        }
        score += contact_scores.get(self.questionnaire_data.get('contact', ''), 0)
        
        # 對方回應頻率計分
        response_scores = {
            '通常很快回覆': 0.5,
            '會回但常常很慢': 1.5,
            '選擇性回覆，有時不回': 2.5,
            '很少回覆我的訊息': 3.5
        }
        score += response_scores.get(self.questionnaire_data.get('response', ''), 0)
        
        # 危險行為計分 (每個行為 +0.8分)
        behaviors = self.questionnaire_data.get('behaviors', [])
        score += len(behaviors) * 0.8
        
        return min(score, 10.0)  # 最高10分
    
    def generate_ai_questions(self, relationship: str, contact: str, response: str, behaviors: List[str]) -> List[str]:
        """根據問卷結果生成客製化AI問題"""
        self.questionnaire_data = {
            'relationship': relationship,
            'contact': contact, 
            'response': response,
            'behaviors': behaviors
        }
        
        questions = []
        
        # 根據關係狀態客製化問題
        if '陌生人' in relationship:
            questions.append("你是透過什麼方式認識這個人的？是什麼特質最吸引你？")
            questions.append("你有沒有想過，你對這個人的了解其實非常有限？")
        elif '曖昧' in relationship:
            questions.append("在這段曖昧關係中，對方有哪些具體行為讓你覺得他/她對你有意思？")
            questions.append("你是否經常在等待對方的明確表態？這種等待對你的日常生活造成什麼影響？")
        elif '朋友' in relationship:
            questions.append("什麼時候開始你對這位朋友產生了超越友誼的感覺？")
            
        # 根據聯絡頻率客製化
        if '每天' in contact:
            questions.append("你每天都要聯絡對方，如果一天沒聯絡會讓你感到焦慮嗎？")
        elif '很少' in contact:
            questions.append("雖然你很少主動聯絡，但是否經常在腦中想著對方？")
            
        # 根據危險行為客製化
        if '經常查看對方的社群動態' in behaviors:
            questions.append("你查看對方社群時，是否會因為看到他/她與別人的互動而感到不開心？")
        if '過度解讀對方的話語和行為' in behaviors:
            questions.append("請舉例最近一次你過度解讀對方行為的情況，事後回想是否合理？")
        if '心情完全取決於對方的回應' in behaviors:
            questions.append("當對方沒有及時回應你時，你會做什麼來轉移注意力嗎？")
            
        # 確保至少有4個問題
        general_questions = [
            "如果這個人從你的生活中消失，你覺得自己會失去什麼？",
            "你覺得你的朋友或家人對你們的關係有什麼看法？",
            "在認識這個人之前，你的生活重心是什麼？現在有改變嗎？"
        ]
        
        for q in general_questions:
            if len(questions) < 4 and q not in questions:
                questions.append(q)
                
        return questions[:4]  # 限制為4個問題
    
    def analyze_cognitive_bias(self) -> str:
        """分析認知偏誤"""
        biases = []
        recommendations = []
        
        behaviors = self.questionnaire_data.get('behaviors', [])
        
        if '過度解讀對方的話語和行為' in behaviors:
            biases.append("確認偏誤：只注意支持自己想法的證據")
            recommendations.append("練習記錄對方的客觀行為 vs 你的主觀解讀")
            
        if '常常幻想跟對方的未來' in behaviors:
            biases.append("理想化偏誤：過度美化對方和關係")
            recommendations.append("列出對方的缺點和你們的實際互動頻率")
            
        if '單方面付出很多' in behaviors:
            biases.append("沉沒成本謬誤：因為投入太多而不願放手")
            recommendations.append("思考繼續投入的實際效益 vs 機會成本")
            
        if '心情完全取決於對方的回應' in behaviors:
            biases.append("外在歸因偏誤：將自我價值依賴在他人身上")
            recommendations.append("培養獨立的興趣和自我價值感")
            
        analysis = f"**主要認知偏誤：**\n"
        for bias in biases:
            analysis += f"• {bias}\n"
        
        analysis += f"\n**建議調整方向：**\n"
        for rec in recommendations:
            analysis += f"• {rec}\n"
            
        return analysis
    
    def generate_detox_plan(self, score: float) -> str:
        """生成個人化勒戒計畫"""
        plan = "## 🎯 個人化三週勒戒計畫\n\n"
        
        # 第一週：斷捨離階段
        plan += "### 第一週：斷捨離階段 🚫\n"
        week1_tasks = [
            "📱 暫停查看對方的社群媒體（可請朋友協助監督）",
            "🗑️ 刪除手機中對方的照片和聊天記錄截圖", 
            "📝 每天記錄3件與對方無關的正面事件",
            "👥 安排2-3個社交活動分散注意力"
        ]
        
        if score >= 8:
            week1_tasks.append("🔕 考慮暫時封鎖對方的聯絡方式")
        if '每天都會傳訊息' in self.questionnaire_data.get('contact', ''):
            week1_tasks.append("⏰ 設定每日聯絡時間限制（如：僅晚上8-9點）")
            
        for task in week1_tasks:
            plan += f"- {task}\n"
        
        # 第二週：重建階段  
        plan += "\n### 第二週：重建階段 🔄\n"
        week2_tasks = [
            "🎨 建立新的日常習慣，培養個人興趣",
            "🤝 主動聯絡被忽略的朋友，重建社交圈",
            "📊 練習客觀記錄：對方實際行為 vs 我的解讀",
            "🏃 每週至少運動3次，釋放情緒能量"
        ]
        
        if '常常幻想跟對方的未來' in self.questionnaire_data.get('behaviors', []):
            week2_tasks.append("💭 每當幻想出現時，立即轉移注意力到具體任務")
            
        for task in week2_tasks:
            plan += f"- {task}\n"
        
        # 第三週：穩固階段
        plan += "\n### 第三週：穩固階段 💪\n"
        week3_tasks = [
            "🧠 反思這段經歷學到的教訓",
            "🛡️ 制定未來感情中的健康界線",
            "❤️ 評估自己的情感需求和依戀模式",
            "🔮 展望沒有這個人的未來生活規劃"
        ]
        
        if score >= 7:
            week3_tasks.append("💬 考慮尋求專業心理諮詢協助")
            
        for task in week3_tasks:
            plan += f"- {task}\n"
            
        return plan
    
    def process_step1(self, relationship: str, contact: str, response: str, *behaviors) -> Tuple[str, gr.update, gr.update]:
        """處理第一步問卷"""
        if not relationship or not contact or not response:
            return "❌ 請完成所有必填問題！", gr.update(), gr.update()
        
        # 處理多選行為
        selected_behaviors = [b for b in behaviors if b]
        
        # 生成AI問題
        questions = self.generate_ai_questions(relationship, contact, response, selected_behaviors)
        
        # 建立問題界面
        question_components = []
        for i, question in enumerate(questions):
            question_components.append(gr.Markdown(f"**問題 {i+1}：** {question}"))
            question_components.append(gr.Textbox(
                label=f"回答 {i+1}",
                placeholder="請詳細描述你的情況和感受...",
                lines=3
            ))
        
        return "✅ 問卷完成！AI正在為你客製化深度問題...", gr.update(visible=False), gr.update(visible=True)

# 初始化診所
clinic = RelationshipClinic()

def step1_submit(relationship, contact, response, beh1, beh2, beh3, beh4, beh5, beh6):
    """第一步提交處理"""
    behaviors = []
    behavior_map = {
        beh1: "經常查看對方的社群動態",
        beh2: "過度解讀對方的話語和行為", 
        beh3: "常常幻想跟對方的未來",
        beh4: "因為對方而忽略其他朋友",
        beh5: "心情完全取決於對方的回應",
        beh6: "單方面付出很多（時間、金錢、關心）"
    }
    
    for checkbox, behavior in behavior_map.items():
        if checkbox:
            behaviors.append(behavior)
    
    # 檢查必填
    if not relationship or not contact or not response:
        return (
            "❌ 請完成所有必填問題！",
            gr.update(visible=True),  # step1
            gr.update(visible=False), # step2  
            gr.update(visible=False), # step3
            "", "", "", ""  # AI問題
        )
    
    # 生成AI問題
    questions = clinic.generate_ai_questions(relationship, contact, response, behaviors)
    
    return (
        "✅ 問卷完成！請回答下方AI客製化問題：",
        gr.update(visible=False), # step1
        gr.update(visible=True),  # step2
        gr.update(visible=False), # step3
        questions[0] if len(questions) > 0 else "",
        questions[1] if len(questions) > 1 else "",
        questions[2] if len(questions) > 2 else "",
        questions[3] if len(questions) > 3 else ""
    )

def step2_submit(ans1, ans2, ans3, ans4):
    """第二步AI問診提交"""
    answers = [ans1, ans2, ans3, ans4]
    
    # 檢查回答完整性
    if not all(ans.strip() for ans in answers):
        return (
            "❌ 請回答所有問題！",
            gr.update(visible=False), # step1
            gr.update(visible=True),  # step2
            gr.update(visible=False), # step3
            "", "", ""  # 診斷結果
        )
    
    # 儲存AI回答
    clinic.ai_responses = {f"answer_{i+1}": ans for i, ans in enumerate(answers)}
    
    # 生成診斷報告
    score = clinic.calculate_infatuation_score()
    cognitive_analysis = clinic.analyze_cognitive_bias() 
    detox_plan = clinic.generate_detox_plan(score)
    
    # 暈船指數描述
    if score >= 8:
        score_desc = f"🚨 **{score:.1f}/10** - 高度暈船警戒！需要立即採取行動"
    elif score >= 6:
        score_desc = f"⚠️ **{score:.1f}/10** - 中度暈船狀態，建議調整心態"
    elif score >= 4:
        score_desc = f"💛 **{score:.1f}/10** - 輕度暈船，保持理性觀察"
    else:
        score_desc = f"💚 **{score:.1f}/10** - 情感狀態健康，繼續保持"
    
    return (
        "✅ 診斷完成！以下是你的個人化報告：",
        gr.update(visible=False), # step1
        gr.update(visible=False), # step2
        gr.update(visible=True),  # step3
        score_desc,
        cognitive_analysis,
        detox_plan
    )

def restart_clinic():
    """重新開始診斷"""
    clinic.reset_data()
    return (
        "🔄 已重置，請重新開始診斷",
        gr.update(visible=True),  # step1
        gr.update(visible=False), # step2
        gr.update(visible=False), # step3
        "", "", "", "", "", "", ""  # 清空所有輸入
    )

# 建立Gradio界面
with gr.Blocks(title="暈船勒戒所 - 互動問診系統") as demo:
    gr.Markdown("# 💔 暈船勒戒所")
    gr.Markdown("### 互動問診式感情分析系統 - 三步驟幫你找回理性")
    
    status = gr.Markdown("👋 歡迎來到暈船勒戒所！請先完成基礎問卷：")
    
    # 第一步：基礎問卷
    with gr.Group(visible=True) as step1:
        gr.Markdown("## 📋 第一步：基礎情況評估")
        
        relationship = gr.Radio(
            choices=[
                "完全陌生人，只是單方面關注",
                "認識但不熟，偶爾有互動", 
                "朋友關係，經常聊天",
                "曖昧階段，關係模糊",
                "正在交往中"
            ],
            label="你們目前的關係狀態是？*",
            info="必填"
        )
        
        contact = gr.Radio(
            choices=[
                "每天都會傳訊息",
                "2-3天一次",
                "大約一週一次", 
                "很少主動聯絡"
            ],
            label="你平均多久會主動聯絡對方？*",
            info="必填"
        )
        
        response = gr.Radio(
            choices=[
                "通常很快回覆",
                "會回但常常很慢",
                "選擇性回覆，有時不回",
                "很少回覆我的訊息"
            ],
            label="對方的回應頻率如何？*", 
            info="必填"
        )
        
        gr.Markdown("**以下哪些行為你有過？（可複選）**")
        beh1 = gr.Checkbox(label="經常查看對方的社群動態")
        beh2 = gr.Checkbox(label="過度解讀對方的話語和行為")
        beh3 = gr.Checkbox(label="常常幻想跟對方的未來")
        beh4 = gr.Checkbox(label="因為對方而忽略其他朋友")
        beh5 = gr.Checkbox(label="心情完全取決於對方的回應") 
        beh6 = gr.Checkbox(label="單方面付出很多（時間、金錢、關心）")
        
        step1_btn = gr.Button("開始AI深度問診 🤖", variant="primary", size="lg")
    
    # 第二步：AI客製化問診
    with gr.Group(visible=False) as step2:
        gr.Markdown("## 🤖 第二步：AI客製化深度問診")
        
        q1 = gr.Markdown()
        a1 = gr.Textbox(label="回答 1", lines=3, placeholder="請詳細描述...")
        
        q2 = gr.Markdown() 
        a2 = gr.Textbox(label="回答 2", lines=3, placeholder="請詳細描述...")
        
        q3 = gr.Markdown()
        a3 = gr.Textbox(label="回答 3", lines=3, placeholder="請詳細描述...")
        
        q4 = gr.Markdown()
        a4 = gr.Textbox(label="回答 4", lines=3, placeholder="請詳細描述...")
        
        step2_btn = gr.Button("生成個人化診斷報告 📊", variant="primary", size="lg")
    
    # 第三步：診斷報告
    with gr.Group(visible=False) as step3:
        gr.Markdown("## 📊 第三步：個人化診斷報告")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💔 暈船指數分析")
                score_result = gr.Markdown()
                
            with gr.Column():
                restart_btn = gr.Button("重新診斷", variant="secondary")
        
        gr.Markdown("### 🔍 認知偏誤診斷")
        cognitive_result = gr.Markdown()
        
        gr.Markdown("### 💊 個人化勒戒計畫") 
        detox_result = gr.Markdown()
    
    # 事件綁定
    step1_btn.click(
        step1_submit,
        inputs=[relationship, contact, response, beh1, beh2, beh3, beh4, beh5, beh6],
        outputs=[status, step1, step2, step3, q1, q2, q3, q4]
    )
    
    step2_btn.click(
        step2_submit,
        inputs=[a1, a2, a3, a4],
        outputs=[status, step1, step2, step3, score_result, cognitive_result, detox_result]
    )
    
    restart_btn.click(
        restart_clinic,
        outputs=[status, step1, step2, step3, relationship, contact, response, a1, a2, a3, a4]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)