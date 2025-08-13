from typing import List
import torch
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 导入令牌认证模块
from token_auth import TokenAuth

# 初始化FastAPI应用
app = FastAPI(
    title="足篮球问题需要的数据支撑",
    description="带令牌认证的大模型推理接口",
    version="1.0.0"
)

# 初始化令牌认证
auth = TokenAuth()

# 定义分类标签列表
static_labels = [
    "联赛基础信息", "比赛的基本信息和比赛事件", "联赛赛季阶段的积分", "两支球队的历史比赛结果",
    "球队的近期比赛结果", "球队的阵容", "球员的基础数据",
    "球员攻防数据统计包含（进球、替补进球、任意球得分、快攻、快攻射门、快攻进球、过人、过人成功、带球摆脱、扑救、铲断（抢断）、封堵、解围、拦截、失去球权、丢失球权）",
    "球员射门和传球技术统计包含（射门、射正、点球、助攻、关键传球、传球、传球成功率、传球成功、传中球、传中球成功、横传、精确横传、长传、精确长传）",
    "球员其他技术统计包含（黄牌、红牌、两黄变红、越位、犯规、被犯规、出场分钟、替补出场分钟、评分、1对1拼抢、1对1拼抢成功）",
    "球队的基础信息",
    "球队攻防数据统计包含（进球、失球、进攻次数、危险进攻、快攻、快攻进球、丢失球权、解围、有效阻挡、拦截、抢断、过人、过人成功）",
    "球队射门和传球统计包含（射门、射正、射偏、击中门框、传球、传球成功、关键传球、传中球、传中球成功、长传、长传成功、助攻）",
    "球队其他技术统计包含（角球、黄牌、红牌、任意球、任意球进球、越位、1对1拼抢、1对1拼抢成功、场均控球率、犯规、被侵犯）",
    "球队中球员的伤停信息", "各家机构欧赔初赔", "各家机构亚盘盘口和水位", "各家机构大小球盘口和水位",
    "球队情报线索", "球队未来的3场赛程", "比赛数据的专家分析评论",
    "关键球员攻防数据统计包含,关键球员射门和传球技术统计,关键球员其他技术统计包含（黄牌、红牌、两黄变红、越位、犯规、被犯规、出场分钟、替补出场分钟、评分、1对1拼抢、1对1拼抢成功）",
    "球队近期比赛亚盘、大小球盘路",
    "球队近期胜平负统计包括（胜率、胜平负、胜平负走势、进球失球；上半场胜率、胜平负、胜平负走势、进球失球；主客场胜率、胜平负、胜平负走势、进球失球；主客场上半场胜率、胜平负、胜平负走势、进球失球）",
    "球队近期盘路统计包括（亚盘盘路、亚盘盘路走势、大小球盘路、大小球盘路走势）",
    "两支球队历史交锋比赛亚盘、大小球盘路", "历史交锋统计包括（胜率、胜平负统计、胜平负走势、进球失球）",
    "比赛数据的专家简要分析评论", "球队近期胜负统计包括（胜率、胜负、胜负走势、场均得分失分）",
    "两队历史交锋胜负统计包括（胜率、胜负、胜负走势、场均得分失分）",
    "球队近3轮、5轮、10轮相同主客场胜平负统计包括（胜率、胜平负统计、胜平负走势、进球失球统计）",
    "球队近3轮、5轮、10轮相同赛事胜平负统计包括（胜率、胜平负统计、胜平负走势、进球失球统计）",
    "球队近3轮、5轮、10轮相同赛事相同主客场胜平负统计包括（胜率、胜平负统计、胜平负走势、进球失球统计）",
    "球队历史交锋近3轮、5轮、10轮相同主客场胜平负统计包括（胜率、胜平负统计、胜平负走势、进球失球）",
    "球队历史交锋近3轮、5轮、10轮相同赛事胜平负统计包括（胜率、胜平负统计、胜平负走势、进球失球）",
    "球队历史交锋近3轮、5轮、10轮相同赛事相同主客场胜平负统计包括（胜率、胜平负统计、胜平负走势、进球失球）",
    "球队近3轮、5轮比赛黄牌、红牌、角球数统计", "同指(历史相同欧赔比赛的赛果统计)",
    "凯利(计算公式：凯利值=某结果欧指*该结果对应的市场各机构的平均概率)",
    "必发(比赛的必发成交分布数据)", "离散(计算公式：百家欧指平均值与每家欧指差值的绝对值的求和)",
    "素材库没有该数据的素材", "问题与足篮球无关"
]

# 标签映射
num_labels = len(static_labels)
label2id = {label: i for i, label in enumerate(static_labels)}
id2label = {i: label for label, i in label2id.items()}
max_length = 512

# 推理优化配置
torch.backends.cudnn.benchmark = True  # 加速固定输入大小的推理
torch.set_grad_enabled(False)  # 全局禁用梯度计算

# 加载模型和分词器
print("正在加载模型...")
try:
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./local_model")

    # 加载模型（使用兼容的参数设置）
    model = AutoModelForSequenceClassification.from_pretrained(
        "./local_model",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16,  # 半精度加速
        device_map="auto"  # 自动选择设备
    )

    # 设置为评估模式
    model.eval()
    print(f"模型加载完成，使用设备: {model.device}")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    raise  # 加载失败时终止程序


# 请求体模型
class GenerateRequest(BaseModel):
    prompt: str
    threshold: float = 0.5  # 概率阈值


# 响应体模型
class GenerateResponse(BaseModel):
    prompt: str
    labels: List[str]
    scores: dict  # 各标签的概率得分


# 根路由
@app.get("/")
def read_root():
    return {
        "message": "欢迎使用足篮球问题数据分类API",
        "提示": "请使用认证令牌访问/generate接口"
    }


# 分类推理接口（需要认证）
@app.post("/generate", response_model=GenerateResponse)
def generate_text(
        request: GenerateRequest,
        token: str = Depends(auth)
):
    try:
        # 文本预处理
        inputs = tokenizer(
            request.prompt,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length"
        ).to(model.device)

        # 模型推理（禁用梯度计算）
        with torch.no_grad():
            outputs = model(**inputs)

        # 处理推理结果
        logits = outputs.logits
        scores = torch.sigmoid(logits).cpu().detach().numpy()[0]  # 多标签分类使用sigmoid

        # 生成标签和得分
        pred_labels = [
            static_labels[j]
            for j, score in enumerate(scores)
            if float(score) > request.threshold
        ]

        score_dict = {
            static_labels[j]: float(score)
            for j, score in enumerate(scores)
        }

        return GenerateResponse(
            prompt=request.prompt,
            labels=pred_labels,
            scores=score_dict
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"推理过程出错: {str(e)}"
        )


# 模型信息接口（需要认证）
@app.get("/model-info")
def get_model_info(token: str = Depends(auth)):
    return {
        "model_name": model.config.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "device": str(model.device),
        "num_labels": num_labels,
        "max_length": max_length
    }


# 健康检查接口（无需认证）
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(model, 'device'),
        "timestamp": torch.datetime.datetime.now().isoformat()
    }
