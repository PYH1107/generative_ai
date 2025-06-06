# 🎨 AI 日記繪師 (AI Diary Painter)

## 📖 專案簡介

一個結合 **對話式 AI** 和 **圖像生成** 的互動應用，透過溫暖的對話了解使用者的日常感受，並自動生成專屬的個人畫像。

**核心概念**：讓 AI 不只是工具，而是能理解你的創作夥伴。

## 🤖 使用的 AI 技術

### 1. **大型語言模型 (LLM)**
- **技術**：Groq LLaMA-3 70B
- **應用**：
  - 漸進式對話設計（5個引導式問題）
  - 情感識別與同理回應
  - 個人特質分析與提取

### 2. **Stable Diffusion 圖像生成**
- **模型**：Stable Diffusion v1.5
- **選擇理由**：適合日常美學，溫暖親切的視覺風格
- **應用**：根據對話內容生成個人化畫像

### 3. **Prompt 工程**
- **功能**：將中文對話內容轉換為英文視覺描述
- **技術**：多層次的語義轉換
  ```
  用戶情感描述 → AI 理解分析 → 視覺元素標籤 → 圖像生成 Prompt
  ```

### 4. **AI Agent 設計模式**
- **參考**：課程中的 Reflection 模式
- **實現**：狀態管理的對話流程，記憶用戶回應並累積分析

## 👤 使用者體驗流程

### 🗣️ **對話階段**（5個問題）
1. **心情探索**：「今天你的心情如何？」
2. **經歷分享**：「發生什麼特別的事情？」
3. **色彩個性**：「用顏色形容現在的你？」
4. **自我比喻**：「你覺得自己像什麼？」
5. **視覺期望**：「希望畫中的你是什麼樣子？」

### 🎨 **創作階段**
1. **AI 分析**：整合所有對話內容
2. **Prompt 生成**：轉換為圖像描述
3. **畫像生成**：Stable Diffusion 創作
4. **結果展示**：呈現專屬的「我眼中的你」

## 💻 技術實現

### 核心架構
```
前端界面 (Gradio) → AI Agent → LLM 分析 → Prompt 工程 → SD 生成 → 結果展示
```

### 主要模組
- **SimpleAIClient**：統一的 AI API 介面
- **DiaryAgent**：對話管理和狀態追蹤
- **圖像生成器**：SD 模型封裝和優化
- **Web UI**：Gradio 響應式界面

## 🎯 專案特色

### 技術層面
- ✅ **多 AI 整合**：LLM + Diffusion Model 協作
- ✅ **智能對話流程**：階段式信息收集
- ✅ **個性化創作**：每次結果都獨一無二
- ✅ **自適應優化**：CPU/GPU 環境自動調整

### 應用層面
- ✅ **情感計算**：理解並回應用戶情感
- ✅ **創意表達**：將抽象感受轉為視覺藝術
- ✅ **人機互動**：自然的對話式體驗
- ✅ **個人化服務**：基於個體特質的客製化

## 🔧 技術挑戰與解決

### 挑戰 1：中英文語義轉換
**解決**：設計專門的 Prompt 工程，讓 LLM 將中文情感描述準確轉換為英文視覺 Prompt

### 挑戰 2：對話狀態管理
**解決**：實現 Agent 類別，維護對話歷史和用戶回應，確保上下文連貫

### 挑戰 3：資源優化
**解決**：智能記憶體管理，支援 CPU/GPU 自動切換，確保不同環境下都能運行

### 挑戰 4：Web 部署穩定性
**解決**：多層級錯誤處理，Gradio 版本相容性管理

## 📊 成果展示

### 技術成果
- 成功整合 LLM 和 Diffusion Model
- 實現流暢的多輪對話交互
- 達到個性化圖像生成效果

### 學習成果
- 深入理解 AI Agent 設計模式
- 掌握 Prompt 工程技巧
- 學會多模型協作架構設計
- 實踐端到端 AI 應用開發

## 🎓 課程連結

本專案結合了課程中學習的多個概念：
- **Demo07a**：Reflection Agent 設計模式
- **Demo08**：Stable Diffusion 圖像生成
- **Demo08g**：Web UI 開發

展現了從基礎概念到實際應用的完整學習歷程。