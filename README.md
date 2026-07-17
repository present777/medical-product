# AI Clinical OSCE Simulator

A clinical communication training engine powered by Large Language Models, RAG, and FSM architecture.

## Core Features
*   **RAG-Driven Evidence Base**: Integrates `ChromaDB` and HuggingFace to vectorize medical textbooks, avoiding LLM hallucinations via strict evidence-based citations.
*   **Dynamic SP Generation**: Utilizes prompt engineering with random seeds to dynamically synthesize standardized patient (SP) profiles for over 30 core diseases.
*   **FSM Lifecycle Management**: Implements a strict Finite State Machine using Streamlit's `session_state` to strictly separate chatting, diagnosing, and evaluating phases.
*   **Multi-modal Interaction**: Supports real-time voice input (`streamlit-mic-recorder`) and generates high-fidelity patient avatars via the Kolors vision model.

---

## Quick Start

1. Clone the repository:
   ```bash
   git clone [https://github.com/present777/medical-product.git](https://github.com/present777/medical-product.git)
   cd medical-product

```

2. Install dependencies:
```bash
pip install streamlit openai langchain-community langchain-huggingface langchain-chroma pymupdf streamlit-mic-recorder requests

```


3. Configure API Keys:
Create a `.streamlit/secrets.toml` file in the root directory and add your keys:
```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key"
SILICON_FLOW_API_KEY = "your_siliconflow_api_key"

```


4. Prepare Knowledge Base:
Create a `medical_books` folder in the root directory and place your medical PDF textbooks inside.
5. Run the engine:
```bash
streamlit run web+voice.py

```



---

## Future Work

* [ ] Expand the knowledge base with more specialized medical textbooks.
* [ ] Optimize LLM inference latency for smoother real-time voice conversations.
* [ ] Add support for tracking and visualizing user progress over multiple training sessions.

---

## 中文说明

本项目是一个专为医学初学者打造的临床沟通智能训练引擎，基于大语言模型、RAG 技术与状态机架构开发。

### 核心亮点

* **RAG 循证检索**：集成 ChromaDB 与 HuggingFace 将医学教材向量化，通过强制引用本地文献消除大模型幻觉，实现基于医学循证的客观评分。
* **动态 SP 生成**：利用提示词工程与随机因子，动态生成覆盖 30 余种核心急腹症与内科疾病的标准化病人（SP）档案。
* **状态机生命周期**：使用 Streamlit 状态机严格隔离“问诊、诊断、评估”三个阶段的单向流转，防止数据泄露与模型“脑补”。
* **多模态交互**：集成语音识别组件支持真实场景模拟，并调用硅基流动 API（Kolors）实时生成高度还原的患者体征图像。

```

```
