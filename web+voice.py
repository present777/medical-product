import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import glob
from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time
import random
import streamlit as st
from streamlit_mic_recorder import speech_to_text
import requests  # <--- 新增
import json

# ================= 1. 全局配置区域 =================

st.set_page_config(page_title="医循：基于AI与医学循证逻辑的临床沟通智能训练系统", page_icon="🏥", layout="wide")
API_KEY = st.secrets["DEEPSEEK_API_KEY"]# 替换为api—key
silicon_key = st.secrets["SILICON_FLOW_API_KEY"]
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

KB_SOURCE_FOLDER = "./medical_books"
KB_DB_PATH = "./medical_knowledge_db"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ================= 2. RAG 知识库模块 =================

@st.cache_resource(show_spinner=False)
def build_or_load_knowledge_base():

    # 检查数据库文件夹是否存在且不为空
    if os.path.exists(KB_DB_PATH) and os.listdir(KB_DB_PATH):
        print("✅ 检测到本地知识库，跳过构建步骤。")
        return True

    print("📚 正在初始化知识库 (RAG)...")

    # 扫描 PDF
    pdf_files = glob.glob(os.path.join(KB_SOURCE_FOLDER, "*.pdf"))
    if not pdf_files:
        print("⚠️ 警告: 'medical_books' 文件夹为空，将运行在无知识库模式。")
        return True

    documents = []
    for pdf_path in pdf_files:
        print(f"   正在加载教材: {os.path.basename(pdf_path)}...")
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")

    if not documents: return True

    # 切分文本
    print("   正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    # 向量化并存储
    print("   正在生成向量索引 (首次运行较慢，请耐心等待)...")
    embedding_function = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=KB_DB_PATH
    )
    print("✅ 知识库构建完成！")

    return True

def retrieve_evidence(query, k=10):

    if not os.path.exists(KB_DB_PATH) or not os.listdir(KB_DB_PATH):
        return "（本地知识库未构建）"

    try:
        # 1. 加载数据库
        embedding_function = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        vectorstore = Chroma(persist_directory=KB_DB_PATH, embedding_function=embedding_function)

        # 2. 搜索
        results = vectorstore.similarity_search(query, k=k)

        # 3. 格式化输出
        formatted_evidence = ""
        for i, doc in enumerate(results):
            source_file = doc.metadata.get('source', '未知来源').split(os.sep)[-1].replace('.pdf', '')
            page_num = doc.metadata.get('page', 0) + 1

            formatted_evidence += f"\n> 【来源：{source_file} 第 {page_num} 页】\n"
            formatted_evidence += f"{doc.page_content}\n"

        return formatted_evidence

    except Exception as e:
        print(f"检索出错: {e}")
        return f"（检索出错: {e}）"

# ================= 🚀 新增功能：患者图像与体征生成引擎 =================
def generate_patient_profile(disease_name):
    """根据抽中的疾病，生成患者属性、体征描述，并调用硅基流动API生成图像"""
    age = random.randint(25, 75)
    gender_en = random.choice(["male", "female"])
    gender_cn = "男性" if gender_en == "male" else "女性"

    disease_features = {
        # --- 呼吸系统 ---
        "大叶性肺炎": {"prompt": "feverish flush on cheeks, coughing, slightly rapid breathing, tired look", "signs": "高热面容 / 呼吸稍促 / 疲乏"},
        "慢性阻塞性肺疾病 COPD": {"prompt": "barrel chest visible under gown, pursed-lip breathing, slightly cyanotic lips, older patient", "signs": "桶状胸 / 唇甲微发绀 / 喘息"},
        "支气管哮喘": {"prompt": "sitting upright, anxious expression, chest tight, slight wheezing posture", "signs": "端坐呼吸 / 神情焦虑 / 喘憋"},
        "支气管扩张": {"prompt": "pale, holding a tissue, looking exhausted from coughing", "signs": "面色苍白 / 咯血后虚弱"},
        "肺血栓栓塞症": {"prompt": "sudden severe chest pain expression, clutching chest, extremely anxious, pale, sweating", "signs": "极度焦虑 / 呼吸急促 / 大汗"},
        "自发性气胸": {"prompt": "holding one side of the chest, sharp pain expression, difficult breathing", "signs": "捂一侧胸口 / 呼吸困难 / 表情痛苦"},
        "呼吸衰竭": {"prompt": "cyanosis on lips and fingertips, confused or lethargic expression, weak", "signs": "明显发绀 / 神志淡漠 / 极度虚弱"},
        "肺结核": {"prompt": "very thin, pale with slight malar flush (red cheeks), tired, sweating slightly", "signs": "消瘦 / 颧红 / 盗汗虚弱"},

        # --- 循环系统 ---
        "慢性心力衰竭": {"prompt": "sitting up leaning forward, swollen lower legs visible, labored breathing, tired", "signs": "端坐位 / 下肢浮肿 / 气喘"},
        "急性左心衰竭": {"prompt": "extreme respiratory distress, coughing, very anxious, pale and sweaty", "signs": "极度呼吸困难 / 大汗 / 极度烦躁"},
        "原发性高血压": {"prompt": "rubbing back of the neck or temples, flushed face, slight dizziness expression", "signs": "面色潮红 / 揉捏后颈 / 略显头晕"},
        "稳定型心绞痛": {"prompt": "clutching chest with one hand, grimacing slightly, resting", "signs": "按压胸前区 / 痛苦面容 / 休息状"},
        "急性心肌梗死 STEMI": {"prompt": "severe crushing chest pain, holding chest tightly, pale skin, heavy sweating, terrified look", "signs": "濒死感 / 极度痛苦 / 苍白大汗"},
        "心房颤动": {"prompt": "hand on chest feeling heartbeat, slightly anxious, pale", "signs": "手捂心口 / 心悸不适 / 略显焦虑"},
        "感染性心内膜炎": {"prompt": "feverish, pale, small red spots on skin (petechiae) visible", "signs": "发热面容 / 苍白 / 皮肤可见瘀点"},
        "急性心包炎": {"prompt": "leaning forward to relieve pain, clutching chest, pained expression", "signs": "前倾坐位 / 表情痛苦 / 捂胸"},

        # --- 消化系统 ---
        "胃食管反流病 GERD": {"prompt": "touching lower chest/upper abdomen, slightly uncomfortable expression, heartburn", "signs": "手抚胸骨后 / 皱眉 / 反酸不适"},
        "消化性溃疡": {"prompt": "pressing upper abdomen (epigastric area), dull pain expression", "signs": "按压上腹部 / 痛苦面容"},
        "肝硬化失代偿期": {"prompt": "jaundiced (yellow) skin and eyes, distended abdomen (ascites), very thin limbs", "signs": "黄疸 / 腹部膨隆 / 面色晦暗"},
        "肝性脑病": {"prompt": "confused, lethargic, slight tremors in hands, yellow skin", "signs": "神志恍惚 / 黄疸 / 扑翼样震颤"},
        "上消化道出血": {"prompt": "extremely pale, very weak, dizzy, lying flat", "signs": "面色苍白 / 极度虚弱 / 头晕"},
        "溃疡性结肠炎": {"prompt": "holding lower abdomen, pained expression, pale, tired", "signs": "捂下腹部 / 虚弱 / 痛苦面容"},
        "急性胰腺炎": {"prompt": "severe abdominal pain, bending forward, holding upper abdomen, sweating", "signs": "剧烈腹痛 / 弯腰抱腹 / 大汗"},
        "结核性腹膜炎": {"prompt": "thin, holding abdomen, slight feverish look, tired", "signs": "消瘦 / 腹痛不适 / 低热面容"},

        # --- 外科 急腹症 ---
        "急性阑尾炎": {"prompt": "curled up, holding lower right abdomen, grimacing in pain", "signs": "捂右下腹 / 卷曲体位 / 痛苦面容"},
        "急性胆囊炎": {"prompt": "holding upper right abdomen, pain taking a deep breath, sweating", "signs": "捂右上腹 / 不敢深呼吸 / 痛苦大汗"},
        "急性化脓性胆管炎": {"prompt": "jaundiced (yellow) skin, severe abdominal pain, high fever flush, shivering", "signs": "黄疸 / 寒战高热 / 剧烈腹痛"},
        "胃十二指肠溃疡穿孔": {"prompt": "lying perfectly still, intense severe abdominal pain, pale, sweating", "signs": "强迫仰卧位 / 极度痛苦 / 苍白大汗"},
        "肠梗阻": {"prompt": "distended abdomen, vomiting or feeling nauseous, severe crampy pain", "signs": "腹部膨隆 / 恶心呕吐状 / 阵发性痛苦"},
        "腹股沟斜疝嵌顿": {"prompt": "holding groin area, severe pain, sweating, anxious", "signs": "捂住腹股沟 / 剧烈疼痛 / 大汗"},
    }

    current_feature = disease_features.get(disease_name, {
        "prompt": "neutral expression, slightly tired, resting in hospital bed",
        "signs": "体征平稳 / 略显疲惫 / 痛苦面容"
    })

    final_prompt = f"A highly detailed face portrait of a {age}-year-old Asian {gender_en}, {current_feature['prompt']}, wearing everyday casual clothes, professional photography, 8k resolution."

    # --- 调用硅基流动 API ---
    try:
        silicon_key = st.secrets["SILICON_FLOW_API_KEY"]
        url = "https://api.siliconflow.cn/v1/images/generations"

        payload = {
            "model": "Kwai-Kolors/Kolors",  # ✅ 快手可图全名：极度便宜/免费，画质惊艳
            "prompt": final_prompt,
            "image_size": "1024x1024",
            "batch_size": 1,
            "seed": random.randint(1, 9999999)
        }
        headers = {
            "Authorization": f"Bearer {silicon_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        # 成功获取图片
        image_url = response.json()['images'][0]['url']

    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_msg = e.response.text

        # 在网页右下角弹出详细的错误提示！
        st.toast(f"🚨 硅基流动报错: {error_msg}", icon="❌")
        print(f"⚠️ 图像生成失败: {error_msg}")

        # 恢复成最干净的容错捕获机制
        image_url = f"https://dummyimage.com/400x500/cccccc/000000&text=Patient+Sim"

    return {
        "age": age,
        "gender": gender_cn,
        "signs": current_feature["signs"],
        "image_url": image_url
    }

# ================= 3. 病例生成模块 (Generator) =================

def generate_oldcart_case():
    print("🔄 正在全网检索《内科学》与《外科学》全章节，启动深度随机抽取...")

    textbook_catalog = {

        "【内科】呼吸系统": [
            "大叶性肺炎 (铁锈色痰/高热)", "慢性阻塞性肺疾病 COPD (桶状胸/呼气延长)",
            "支气管哮喘 (发作性伴哮鸣音)", "支气管扩张 (大量脓痰/反复咯血)",
            "肺血栓栓塞症 (突发胸痛/呼吸困难/咯血)", "自发性气胸 (突发针刺样胸痛)",
            "呼吸衰竭 (发绀/神志改变)", "肺结核 (午后低热/盗汗)"
        ],
        "【内科】循环系统": [
            "慢性心力衰竭 (端坐呼吸/双下肢水肿)", "急性左心衰竭 (粉红色泡沫痰)",
            "原发性高血压 (头晕/颈板紧)", "稳定型心绞痛 (劳力性胸痛/含服硝酸甘油缓解)",
            "急性心肌梗死 STEMI (濒死感/大汗淋漓/持续胸痛)", "心房颤动 (心悸/脉搏短绌)",
            "感染性心内膜炎 (发热/心脏杂音)", "急性心包炎 (心包摩擦音)"
        ],
        "【内科】消化系统": [
            "胃食管反流病 GERD (反酸/烧心)", "消化性溃疡 (节律性上腹痛)",
            "肝硬化失代偿期 (腹水/肝掌/蜘蛛痣)", "肝性脑病 (扑翼样震颤/意识障碍)",
            "上消化道出血 (呕血/黑便)", "溃疡性结肠炎 (黏液脓血便/里急后重)",
            "急性胰腺炎 (束带状腹痛/进食油腻诱发)", "结核性腹膜炎 (揉面感)"
        ],
        "【内科】泌尿系统": [
            "急性肾小球肾炎 (血尿/蛋白尿/水肿/高血压)", "肾病综合征 (大量蛋白尿/低蛋白血症)",
            "尿路感染 (尿频/尿急/尿痛/肾区叩击痛)", "慢性肾衰竭 (贫血/恶心/夜尿增多)",
            "急性肾损伤 AKI (少尿/无尿)"
        ],
        "【内科】血液系统": [
            "缺铁性贫血 (匙状指/吞咽困难)", "再生障碍性贫血 (全血细胞减少/出血/感染)",
            "急性白血病 (贫血/出血/发热/骨痛)", "原发性血小板减少性紫癜 ITP (皮肤瘀点/紫癜)",
            "多发性骨髓瘤 (骨痛/肾损害/贫血)"
        ],
        "【内科】内分泌与代谢": [
            "甲状腺功能亢进症 Graves病 (突眼/手颤/怕热多汗)", "甲状腺功能减退症 (淡漠/黏液性水肿)",
            "糖尿病酮症酸中毒 DKA (烂苹果味/深大呼吸)", "痛风 (第一跖趾关节红肿热痛)",
            "库欣综合征 Cushing (满月脸/水牛背/紫纹)"
        ],
        "【内科】风湿免疫科": [
            "类风湿关节炎 (晨僵/对称性关节肿痛)", "系统性红斑狼疮 SLE (蝶形红斑/光过敏/口腔溃疡)",
            "强直性脊柱炎 (腰背痛/活动后减轻/HLA-B27阳性)"
        ],

        "【外科】急腹症 (核心考点)": [
            "急性阑尾炎 (转移性右下腹痛)", "急性胆囊炎 (墨菲征阳性)",
            "急性化脓性胆管炎 (腹痛/寒战高热/黄疸 - Charcot三联征)",
            "胃十二指肠溃疡穿孔 (板状腹/膈下游离气体)", "肠梗阻 (痛/吐/胀/闭)",
            "腹股沟斜疝嵌顿 (不可回纳/剧痛)"
        ],
        "【外科】颅脑与神经": [
            "硬脑膜外血肿 (中间清醒期)", "颅底骨折 (熊猫眼征)", "颅内压增高 (头痛/呕吐/视乳头水肿)"
        ],
        "【外科】胸部与血管": [
            "肋骨骨折 (反常呼吸运动)", "张力性气胸 (极度呼吸困难/气管移位)",
            "食管癌 (进行性吞咽困难)", "下肢静脉曲张 (久站酸胀)", "深静脉血栓形成 DVT (下肢肿胀)"
        ],
        "【外科】骨科 (创伤与病)": [
            "桡骨远端骨折 Colles (银叉样畸形)", "锁骨骨折 (患肩下垂)", "股骨颈骨折 (短缩外旋畸形)",
            "腰椎间盘突出症 (直腿抬高试验阳性)", "颈椎病 (上肢麻木)", "膝关节半月板损伤 (关节交锁)"
        ]
    }

    all_diseases = []
    for category, diseases in textbook_catalog.items():
        for disease in diseases:
            parts = disease.split(" (")
            name = parts[0]
            hint = parts[1].replace(")", "") if len(parts) > 1 else "典型临床表现"

            all_diseases.append({
                "category": category,
                "name": name,
                "hint": hint
            })

    # 随机抽取
    target = random.choice(all_diseases)

    # ================= 3. 构造强制命题 Prompt =================
    seed = f"{time.time()}-{random.randint(1, 100000)}"

    prompt = f"""
    【随机干扰因子】: {seed}

    【强制任务】
    你现在是国家执业医师资格考试（实践技能）的出题专家。请务必根据以下指令生成一个标准化病人（SP）的完整剧本。

    **锁定病种**: 【{target['name']}】
    **所属章节**: {target['category']}
    **核心特征参考**: {target['hint']}

    【出题要求】
    1. **严格限制**：你必须生成一个患有“{target['name']}”的病人。**严禁**更换成其他疾病。
    2. **教科书标准**：病例必须符合《内科学》或《外科学》教材描述。
    3. **国家医考标准**：必须包含以下完整维度的病史资料：
       - 主诉 (核心症状 + 持续时间)
       - 现病史：
         ① 发病诱因 (如受凉、饱餐、劳累等)
         ② 主要症状特点 (部位、性质、程度、持续时间、加重/缓解因素)
         ③ 伴随症状 (是否有发热、恶心、呕吐、放射痛等，用于鉴别诊断)
         ④ 一般情况 (发病以来的饮食、睡眠、大便、小便、体重变化情况，必须全部编造具体细节)
         ⑤ 诊疗经过 (是否去过医院？做过什么检查？吃过什么药？)
       - 既往史、个人史、家族史、过敏史 (如高血压、糖尿病、吸烟饮酒、遗传病等)
    4. **拒绝剧透**：在【患者信息】里不要直接写“患有{target['name']}”，只写主诉（如“多饮多尿2个月”）。

    【输出格式】
    【诊断结果】: {target['name']}
    【患者信息】: (简要，包含年龄、性别、主诉)
    【完整病历设定】: (详细列出上述所有维度的设定，分点作答)
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            temperature=1.4,  # 保持高随机性，让细节每次都不同
        )
        return response.choices[0].message.content, target['name']
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        sys.exit(1)


# ================= 4. 评估模块 (Evaluator) =================

def run_evaluation(chat_history, user_diagnosis,correct_disease):
    print("\n" + "=" * 60)
    print("📊 正在生成【教科书级临床能力评估报告】...")
    print("=" * 60)

    print(f"🔍 [系统日志] 标准答案已锁定：{correct_disease}")


    # 为了防止书里叫"胆石症"而AI搜"胆囊炎"导致漏搜，我们先生成搜索词

    keyword_prompt = f"""
    疾病名称：{correct_disease}

    请列出该疾病在中文医学教科书（如《内科学》《外科学》）中可能出现的 3 个标准章节标题或关键词。
    要求：为同义词。

    格式：词1 词2 词3
    (例如：如果是"急性广泛前壁心肌梗死"，输出：心肌梗死 冠心病 ACS)
    """
    try:
        kw_res = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": keyword_prompt}], temperature=0.1
        )
        search_keywords = kw_res.choices[0].message.content.strip()

    except:
        search_keywords = correct_disease

    # 2. 整理原始对话
    raw_dialogue = ""
    for msg in chat_history:
        role = "医生" if msg['role'] == 'user' else "患者"
        raw_dialogue += f"{role}: {msg['content']}\n"



    query = f"{search_keywords} {correct_disease} 临床表现 症状 体征 诊断 鉴别"
    evidence_text = retrieve_evidence(query, k=12)

    # 4. 评估 Prompt
    evaluator_prompt = f"""
    你现在是【国家执业医师资格考试（实践技能）】的考官，同时也是一名精通【循证医学(EBM)】的临床导师。请严格根据《病史采集评分标准》，生成一份含“精准引用”的权威评估报告。

    === 📚 本地教材片段 (Ground Truth) ===
    {evidence_text}
    ===========================================

    === 📝 考生对话记录 ===
    {raw_dialogue}

    【任务要求】
    【判分核心逻辑：严谨逻辑 + 语义宽容】

    1. **诊断评分规则 (30分)**：
       - **数据源**：判定诊断是否正确时，**仅依据上面的“考生提交的最终诊断”**。
       - **禁止行为**：不要去“考生问诊对话流”里找诊断！因为考生是在问诊结束后单独提交的诊断。
       - **判定逻辑**：
         - 如果【{user_diagnosis}】与【{correct_disease}】含义一致（如"阑尾炎" vs "急性阑尾炎"），判 ✅。
         - 如果不一致（如"胰腺炎" vs "阑尾炎"），判 ❌，并在“扣分原因”中明确写：“考生提交了错误的诊断”。
         - 绝对不要写“考生未在对话中给出诊断”。
         
    2. **病史采集核查规则 (50分)**：
       必须严格核查考生是否在对话中提问了以下【全部】国家医考必考维度（不论什么病，这些项都必须问）：
       - **① 发病诱因**：是否问了受凉、饮食、劳累等。
       - **② 主要症状特点**：部位、性质、程度、持续时间等。
       - **③ 伴随症状**：是否问了发热、恶心等其他系统的症状。
       - **④ 一般情况**：是否问了发病以来的**饮食、睡眠、大便、小便、体重变化**。（注意：这是国内必考重点！只要问了其中一两项就算有效得分 ✅）
       - **⑤ 诊疗经过**：是否问了就诊史、用药史。
       - **⑥ 相关病史**：是否问了既往史、个人史、家族史或过敏史。
       
    3. **问诊技巧与医德医风 **：
       - **爱伤观念**：国内医考核心词汇！考生开场是否有问候？过程中是否有安抚患者情绪？
       - **条理性**：问诊是否有逻辑，无诱导性提问。
       
    4. 循证医学思维评估 ：你必须像导师一样，对考生的临床推理进行 EBM 维度的剖析。指出其忽略的个体化因素，并推荐相关的指南或文献方向。

    5. **语义等效性识别 (Semantic Equivalence)**：这是最重要的！
    - 考生的对象是普通患者，**必须使用通俗语言**。
    - **只要考生的提问能获取教材要求的核心信息，必须给分！**
    - ❌ 错误判例：考生问“疼不疼？”，教材要求“压痛”，判错。（这是不对的）
    - ✅ 正确判例：考生问“我按这里你疼吗？”，教材要求“压痛”，**必须判对 (✅)**。
    - ✅ 正确判例：考生问“吃油腻的东西会加重吗？”，教材要求“诱因”，**必须判对 (✅)**。

    6. **证据溯源**：虽然语言可以通俗，但必须真的问了。如果连“按肚子”都没问，就不能算查了“压痛”。

    7. **拒绝送分**：如果对话记录里没有体现，无论标准病例里有没有，一律判错。

    8. **必须揭晓答案**：在表格第一行明确展示“标准诊断”。

    9. **精准引用 (Citation)**：
        - 在“教材来源”一栏，必须包含两部分：
       - **第一行**：复制上方的 `📖 [来源：...  第 ... 页]` 标签。
       - **第二行**：**摘录原文**。从上方片段中复制一句最相关的原话，用引号括起来。
       - **严禁瞎编**：如果上方片段里没有这句话，就不要写。
       - **绝对禁止**：在表格单元格的内容中使用竖线符号（|），否则会破坏表格结构！


    【输出格式】
    ### 一、 诊断结论 (30分)
    | 项目 | 内容 | 判定 |
    |---|---|---|
    | **标准正确诊断** | **{correct_disease}** | (标准答案) |
    | **考生的诊断** | {user_diagnosis} | (✅得分 / ❌不得分) |

    ### 二、 关键特征采集核查 (50分)
    | 核心指征 | 状态 | 现场还原与教材溯源 |
    |---|---|---|
    | **1. 发病诱因** | ✅/❌ | **考生提问**：“...”<br>**判定**：有效采集 / 未涉及。<br>**教材依据**：... |
    | **2. 主要症状特点** | ✅/❌ | 同上 |
    | **3. 伴随症状** | ✅/❌ | 同上 |
    | **4. 一般情况(二便/饮食/睡眠)** | ✅/❌ | 同上 |
    | **5. 诊疗经过** | ✅/❌ | 同上 |
    | **6. 相关病史(既往/家族/过敏)** | ✅/❌ | 同上 |
    | (例: 进食油腻诱发) | ✅ | **考生提问**：“...”<br>**判定**：有效采集。<br>**教材来源**：<br>📖 [来源：外科学  第 465 页]<br>“原文：典型发作常有进食油腻食物史...” |
    | (例: 墨菲征)| ❌ | **考生提问**：(无相关提问)<br>**判定**：未触诊。<br>**教材来源**：<br>📖 [来源：外科学  第 466 页]<br>“原文：Murphy征阳性是急性胆囊炎的特异性体征。” |
    
    ### 三、 🎯 循证医学临床思维评估 (EBM Assessment - 20分)
    | 评估维度 | 专家点评与循证指导 |
    |---|---|
    | **1. 关键证据识别** | (分析考生是否遗漏了影响决策的关键证据。如：是否忽略了患者的高危因素、既往病史等？) |
    | **2. 决策与证据匹配度** | (根据上方知识库，指出考生的诊断/提问是否符合最新指南。若发现指南依据，请标注【强烈推荐查阅：XXX指南】) |
    | **3. 高阶循证反思** | (向考生提出一个高阶假设性问题，例如：“若该患者合并XX疾病/对某药物过敏，你的诊疗重点会有何改变？”以此启发其循证思维。) |
    
    ### 四、 综合评分表
    | 考核维度 | 权重 | 实得分 | 扣分原因 |
    |---|---|---|---|
    | 诊断结论 | 30 | (计算) | ... |
    | 关键特征采集核查 | 50 | (计算) | 漏问了... |
    | 循证医学临床思维评估 | 20 | (计算) | ... |
    | **总分** | **100** | **(总分)** |

    ### 四、 点评与建议
    (在计算【关键特征采集核查】得分时，请严格执行以下数学公式：
    实得分 = 50分 - (漏问项数 × 10分)。
    严禁使用模糊的比例折算或主观给分。（如扣分后为负分，请写0分）
    在计算【循证医学临床思维评估】得分时，请结合关键证据识别，决策与证据匹配度等方面，根据整体表现评分：
    - **优秀 (16-20分)**：提问具有明确的鉴别诊断导向，主动寻找支持或排除特定诊断的证据（如询问红旗征、关键既往史）。
    - **良好 (11-15分)**：问诊逻辑较清晰，能围绕主诉展开，但缺乏对复杂临床证据的深度挖掘。
    - **合格 (6-10分)**：仅完成了基础套路问诊，未体现出根据患者反馈调整提问策略的循证意识。
    - **不合格 (0-5分)**：提问杂乱无章，完全遗漏影响临床决策的核心证据。
    请根据上述表格计算总分，并基于上述分析给出建议)
    """

    # 5. 生成报告
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "system", "content": evaluator_prompt}], stream=True
        )

        full_report = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                c = chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
                full_report += c
        print("\n\n" + "=" * 60)
    except Exception as e:
        print(f"评估失败: {e}")


# ================= 5. 主程序逻辑 (Main) =================

def main():
    st.title("🏥 医循：基于AI与医学循证逻辑的临床沟通智能训练系统")

    if "exam_history" not in st.session_state:
        st.session_state.exam_history = []

    # ================= 新增：侧边栏历史记录与导出 =================
    with st.sidebar:
        st.header("📜 问诊训练档案")

        if not st.session_state.exam_history:
            st.info("尚未完成任何训练记录。")
        else:
            # --- 导出功能 ---
            # 拼接所有历史记录为一段长文本
            export_text = "医循系统 - 临床沟通训练导出报告\n" + "=" * 40 + "\n"
            for i, rec in enumerate(st.session_state.exam_history):
                export_text += f"\n【记录 {i + 1}】 时间: {rec['time']}\n"
                export_text += f"标准诊断: {rec['disease']} | 您的诊断: {rec['user_diag']}\n"
                export_text += f"报告详情:\n{rec['report']}\n"
                export_text += "-" * 40 + "\n"

            st.download_button(
                label="📥 导出所有历史报告 (TXT)",
                data=export_text,
                file_name=f"问诊训练报告_{time.strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

            st.divider()

            # --- 历史列表查看 ---
            st.subheader("历史记录")
            # 使用 reversed 让最新的记录排在最前面
            for i, rec in enumerate(reversed(st.session_state.exam_history)):
                idx = len(st.session_state.exam_history) - i
                with st.expander(f"第 {idx} 次: {rec['disease']}"):
                    st.caption(f"🕒 {rec['time']}")
                    st.markdown(f"**您的诊断**: {rec['user_diag']}")
                    st.markdown("---")
                    st.markdown(rec['report'])
    # =========================================================

    # 1. 准备 RAG
    with st.spinner("正在连接教材知识库..."):
        build_or_load_knowledge_base()

    # 2. 生成病例
    if "status" not in st.session_state:
        # 抽取考题
        secret_case, correct_disease = generate_oldcart_case()
        st.session_state.secret_case = secret_case
        st.session_state.correct_disease = correct_disease

        # 调用引擎，生成患者图片和档案 (增加spinner提示用户等待)
        with st.spinner("✨ 正在生成标准化病人(SP)全息视觉档案，请稍候(约5-10秒)..."):
            st.session_state.patient_profile = generate_patient_profile(correct_disease)

        # 3. 设定病人角色
        patient_system_prompt = f"""
        你现在是一名正在参加国家执业医师客观结构化临床考试（OSCE）的标准化病人（SP）。
        这是你的真实病历设定：
        {secret_case}
        
        【附加人设】
        你的年龄是 {st.session_state.patient_profile['age']} 岁，性别是 {st.session_state.patient_profile['gender']}。请在对话中严格符合该身份的语气，如果医生问起基本信息，请按此回答。

        【表演规则】
        1. **被动触发**：不能像背书一样主动把病情说出。只有当医生具体问到了某一个维度（如诱因、伴随症状、二便情况、既往史），你才回答那一项的具体信息。
       - 例如：如果不问 "怎么痛"，你就只说 "痛"，不要主动说 "绞痛"。
        2. **通俗化**：你是一个普通老百姓。绝不能说“我有墨菲征阳性”、“我右下腹压痛”，必须说“我右边肋骨下面按着疼”、“我肚子右下角疼”。
        3. **针对一般情况的回答**：如果医生问“你最近吃得好吗/睡得好吗/大小便正常吗”，请根据病历设定里的【一般情况】如实自然地回答。
        
        """

        st.session_state.messages = [{"role": "system", "content": patient_system_prompt}]
        st.session_state.history_for_review = []

        # 开场白
        first_msg = "医生...我不太舒服..."
        st.session_state.messages.append({"role": "assistant", "content": first_msg})
        st.session_state.history_for_review.append({"role": "assistant", "content": first_msg})

        # 标记目前进入“聊天状态”
        st.session_state.status = "chatting"

    # ================= 状态一：问诊聊天阶段 =================
    if st.session_state.status == "chatting":

        col_header1, col_header2 = st.columns([4, 1])
        with col_header1:
            st.info("💡 **问诊提示**：请按照《国家执业医师病史采集标准》进行全面问诊。")
        with col_header2:
            if st.button("📝 结束问诊,我要交卷", type="primary", use_container_width=True):
                st.session_state.status = "diagnosing"
                st.rerun()

        st.divider()

        left_col, right_col = st.columns([1, 2.2])

        # --- 左侧栏：患者视觉档案 ---
        with left_col:
            profile = st.session_state.patient_profile

            st.markdown(f"""
                        <div style="
                            background: linear-gradient(90deg, #4A90E2 0%, #35C3A5 100%);
                            padding: 8px 15px;
                            border-radius: 8px 8px 0 0;
                            color: white;
                            font-weight: bold;
                            display: flex;
                            justify-content: space-between;
                        ">
                            <span>👤 患者档案</span>
                            <span>{profile['age']}岁 &nbsp; {profile['gender']}</span>
                        </div>
                    """, unsafe_allow_html=True)

            st.image(profile['image_url'], use_container_width="always")

            st.markdown(f"""
                        <div style="
                            background-color: #6c757d;
                            padding: 8px 15px;
                            border-radius: 0 0 8px 8px;
                            color: white;
                            font-size: 14px;
                            text-align: center;
                        ">
                            ⓘ 当前体征：{profile['signs']}
                        </div>
                    """, unsafe_allow_html=True)

        # --- 右侧栏：聊天互动区 ---
        with right_col:
            # 使用固定高度的容器装聊天记录
            chat_container = st.container(height=520, border=True)
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] != "system":
                        avatar = "👨‍⚕️" if msg["role"] == "user" else "👤"
                        with st.chat_message(msg["role"], avatar=avatar):
                            st.markdown(msg["content"])

            # 底部输入区 (紧贴着聊天框下方)
            input_col1, input_col2 = st.columns([1, 5])
            with input_col1:
                spoken_text = speech_to_text(
                        language='zh',
                        start_prompt="🎤 点击说话",
                        stop_prompt="🛑 停止录音",
                        just_once=True,
                        key='STT'
                    )
            with input_col2:
                written_text = st.chat_input("👨‍⚕️ 按左侧麦克风说话，或在此打字...")

            user_input = spoken_text or written_text

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.history_for_review.append({"role": "user", "content": user_input})

                # 防止重绘闪烁，直接在此处获取回复并 rerun
                with st.spinner("患者正在思考..."):
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=st.session_state.messages,
                        stream=False  # 在复杂布局中使用 False 更稳定
                    )
                    full_reply = response.choices[0].message.content

                st.session_state.messages.append({"role": "assistant", "content": full_reply})
                st.session_state.history_for_review.append({"role": "assistant", "content": full_reply})
                st.rerun()

    # ================= 状态二：填写诊断阶段 =================
    elif st.session_state.status == "diagnosing":
        st.info("🛑 问诊环节已结束，请提交您的最终诊断！")

        # 表单输入 (取代原本的 input)
        with st.form("diagnosis_form"):
            diagnosis_input = st.text_input("👉 我的诊断是：", placeholder="例如：急性阑尾炎")
            submitted = st.form_submit_button("提交诊断并生成报告")

            if submitted:
                if not diagnosis_input.strip():
                    st.warning("⚠️ 医生，请填写您的初步诊断结果后再交卷！")
                else:
                    st.session_state.user_final_diagnosis = diagnosis_input.strip()
                    st.session_state.status = "evaluating"
                    st.rerun()

        # ================= 状态三：生成评估报告阶段 =================
    elif st.session_state.status == "evaluating":
        st.success(f"✅ 已记录诊断：[{st.session_state.user_final_diagnosis}]。系统正在翻阅教材进行评分...")
        st.subheader("📝 教科书级临床能力评估报告")

        with st.container(border=True):
            # 【优化 1：缓存报告】检查是否已经生成过报告
            if "final_report" not in st.session_state:
            # 调用你已经改成 yield 输出的 run_evaluation 函数
                report_stream = run_evaluation(
                    st.session_state.history_for_review,
                    st.session_state.user_final_diagnosis,
                    st.session_state.correct_disease  # 传入标准答案
                )
                full_text = st.write_stream(report_stream)
                st.session_state.final_report = full_text

                new_record = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "disease": st.session_state.correct_disease,
                    "user_diag": st.session_state.user_final_diagnosis,
                    "report": st.session_state.final_report
                }
                st.session_state.exam_history.append(new_record)

            else:
            # 如果已经存过了，直接打印文字，不再调用 AI 重新生成
                st.markdown(st.session_state.final_report)

        # 【优化 2：使用 on_click】定义一个重置函数
        def reset_exam():
            # 1. 备份历史记录
            current_history = st.session_state.exam_history

            # 2. 清空所有状态（包括问诊记录、当前报告、图片等）
            st.session_state.clear()

            # 3. 把历史记录还原回去
            st.session_state.exam_history = current_history

        st.button("🔄 抽取新病例重新考试", on_click=reset_exam)

if __name__ == "__main__":
    main()