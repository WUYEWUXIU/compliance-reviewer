"""
Golden Set — 用于回归测试的标注样本集

每条样本结构：
  - text: 待审营销文案
  - expected_compliant: "yes" | "no"
  - expected_violations: 预期触发的违规类型列表（可为空）
  - note: 标注说明，说明判断依据或难点

覆盖维度：
  1. 每个违规类型（V01-V11）的典型正例
  2. 每个违规类型的否定/合规反例（文案表面相似但不违规）
  3. 多违规类型叠加的复合案例
  4. 边界/难判案例（隐式违规、间接暗示）
"""

from __future__ import annotations
from typing import TypedDict


class GoldenCase(TypedDict):
    id: str
    text: str
    expected_compliant: str          # "yes" | "no"
    expected_violations: list[str]   # violation_type_id 列表
    difficulty: str                  # "easy" | "medium" | "hard"
    note: str
    relevant_chunk_ids: list[str]    # chunks expected to be retrieved


GOLDEN_SET: list[GoldenCase] = [

    # ── V01: 承诺本金不受损失 ─────────────────────────────────────────

    {
        "id": "V01-pos-001",
        "text": "购买本产品，本金完全保障，绝对不会损失一分钱！",
        "expected_compliant": "no",
        "expected_violations": ["V01"],
        "difficulty": "easy",
        "note": "直接含'保障本金'+'不损失'，典型V01",
    },
    {
        "id": "V01-pos-002",
        "text": "零风险理财，本金无忧，让您的财富稳稳增长。",
        "expected_compliant": "no",
        "expected_violations": ["V01"],
        "difficulty": "easy",
        "note": "'零风险'+'本金无忧'双关键词命中V01",
    },
    {
        "id": "V01-neg-001",
        "text": "本产品不保本，投资有风险，历史收益不代表未来表现。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "'不保本'是否定表述，V01不触发",
    },
    {
        "id": "V01-neg-002",
        "text": "本金安全是每位投资者的关切，我们的产品虽然追求稳健，但不承诺本金不受损失，请根据自身风险承受能力选择。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "medium",
        "note": "虽含'本金安全'但后文明确否定承诺，整体合规",
    },
    {
        "id": "V01-hard-001",
        "text": "我们精选低风险资产配置，多年来客户本金从未出现亏损。",
        "expected_compliant": "no",
        "expected_violations": ["V01"],
        "difficulty": "hard",
        "note": "未直接说'保本'，但'从未亏损'构成隐式本金承诺",
    },

    # ── V02: 承诺确定收益 ─────────────────────────────────────────────

    {
        "id": "V02-pos-001",
        "text": "年化收益7%，固定到手，每月自动结算，稳赚不赔！",
        "expected_compliant": "no",
        "expected_violations": ["V02"],
        "difficulty": "easy",
        "note": "'固定到手'+'稳赚不赔'直接承诺确定收益",
    },
    {
        "id": "V02-pos-002",
        "text": "保证年化收益不低于4.5%，到期一次性结算，确定回报。",
        "expected_compliant": "no",
        "expected_violations": ["V02"],
        "difficulty": "easy",
        "note": "'保证年化'+'确定回报'，典型V02",
    },
    {
        "id": "V02-neg-001",
        "text": "历史年化收益约6%，仅供参考，实际收益以到期结算为准，投资有风险。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "medium",
        "note": "'历史年化'+风险提示，是合规的历史业绩展示",
    },
    {
        "id": "V02-hard-001",
        "text": "过去三年，我们的产品每年都为客户带来了8%的回报，表现十分稳定。",
        "expected_compliant": "no",
        "expected_violations": ["V02"],
        "difficulty": "hard",
        "note": "以历史业绩暗示未来确定收益，构成隐式V02；无明确免责声明",
    },

    # ── V03: 使用绝对化用语 ───────────────────────────────────────────

    {
        "id": "V03-pos-001",
        "text": "行业最优产品，没有比我们更好的选择，全市场第一！",
        "expected_compliant": "no",
        "expected_violations": ["V03"],
        "difficulty": "easy",
        "note": "'最优'+'第一'，两处绝对化用语",
    },
    {
        "id": "V03-pos-002",
        "text": "顶级专家团队，绝无仅有的投资机会，仅限本周。",
        "expected_compliant": "no",
        "expected_violations": ["V03"],
        "difficulty": "easy",
        "note": "'顶级'+'绝无仅有'命中V03",
    },
    {
        "id": "V03-neg-001",
        "text": "我们追求为客户提供优质服务，持续改进产品体验。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "'优质'是普通形容词，非绝对化表述",
    },
    {
        "id": "V03-hard-001",
        "text": "本产品是目前市场上唯一支持T+0赎回的同类产品。",
        "expected_compliant": "no",
        "expected_violations": ["V03"],
        "difficulty": "hard",
        "note": "'唯一'属于绝对化用语，即使陈述可能为真也需规避",
    },

    # ── V04: 无资质或不当代言 ─────────────────────────────────────────

    {
        "id": "V04-pos-001",
        "text": "知名影星XX倾情推荐，专家团队背书，购买无忧！",
        "expected_compliant": "no",
        "expected_violations": ["V04"],
        "difficulty": "easy",
        "note": "明星推荐+专家背书，典型V04",
    },
    {
        "id": "V04-pos-002",
        "text": "内部人士透露，这是今年最值得购买的产品，仅限内部渠道。",
        "expected_compliant": "no",
        "expected_violations": ["V04"],
        "difficulty": "medium",
        "note": "'内部人士'构成不当代言，且暗示非公开信息",
    },
    {
        "id": "V04-neg-001",
        "text": "由持牌金融机构XX保险公司承保，依法合规销售。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "持牌机构公示属于合规资质展示，非不当代言",
    },

    # ── V05: 缺失风险提示 ─────────────────────────────────────────────

    {
        "id": "V05-pos-001",
        "text": "年化收益高达6%，每月分红，立即购买锁定收益！",
        "expected_compliant": "no",
        "expected_violations": ["V02", "V05"],
        "difficulty": "easy",
        "note": "承诺收益且全文无任何风险提示，V02+V05双违规",
    },
    {
        "id": "V05-neg-001",
        "text": "本产品由XX保险公司承保，存在一定投资风险，犹豫期内可无损退保，退保损失请参阅合同条款。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "包含承保机构、犹豫期、退保损失，风险提示完整",
    },
    {
        "id": "V05-hard-001",
        "text": "稳健型理财产品，适合保守型投资者，预期年化3%-5%。",
        "expected_compliant": "no",
        "expected_violations": ["V05"],
        "difficulty": "hard",
        "note": "虽用'预期'但全文无法定风险提示语句，仍缺失V05要求",
    },

    # ── V06: 误导性产品比较 ───────────────────────────────────────────

    {
        "id": "V06-pos-001",
        "text": "比银行理财收益高50%，完全碾压其他同类产品，不选我们就是吃亏！",
        "expected_compliant": "no",
        "expected_violations": ["V06"],
        "difficulty": "easy",
        "note": "'碾压'+'比其他高'构成贬低竞品的误导性比较",
    },
    {
        "id": "V06-neg-001",
        "text": "本产品预期收益高于当前一年期定期存款基准利率，具体以合同为准。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "medium",
        "note": "与基准利率比较属于客观参照，非贬低竞品",
    },

    # ── V07: 隐瞒/淡化费用 ───────────────────────────────────────────

    {
        "id": "V07-pos-001",
        "text": "零手续费，无管理费，完全免费购买，到手收益更高！",
        "expected_compliant": "no",
        "expected_violations": ["V07"],
        "difficulty": "easy",
        "note": "'零手续费'+'无管理费'隐瞒实际费用结构",
    },
    {
        "id": "V07-neg-001",
        "text": "本产品收取1.2%/年管理费，申购费0.5%，详见费用说明。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "费用完整披露，合规",
    },
    {
        "id": "V07-hard-001",
        "text": "购买本产品无需支付任何额外费用，一分不多收。",
        "expected_compliant": "no",
        "expected_violations": ["V07"],
        "difficulty": "hard",
        "note": "'无需额外费用'可能掩盖内嵌费用，构成淡化费用",
    },

    # ── V08: 诱导退保/转保 ───────────────────────────────────────────

    {
        "id": "V08-pos-001",
        "text": "旧保单收益太低，赶紧退保升级新产品，换了才不亏！",
        "expected_compliant": "no",
        "expected_violations": ["V08"],
        "difficulty": "easy",
        "note": "'退保'+'换了才不亏'直接诱导退保转保",
    },
    {
        "id": "V08-pos-002",
        "text": "您现有保单已不适合当前需求，建议终止后转入我们的升级产品。",
        "expected_compliant": "no",
        "expected_violations": ["V08"],
        "difficulty": "medium",
        "note": "未用'退保'关键词但实质为诱导转保",
    },
    {
        "id": "V08-neg-001",
        "text": "退保会导致损失，请充分了解后谨慎决定，如需帮助请联系顾问。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "medium",
        "note": "提示退保损失是合规做法，非诱导",
    },

    # ── V09: 伪造/篡改备案信息 ────────────────────────────────────────

    {
        "id": "V09-pos-001",
        "text": "本产品已获银保监会特别批准，备案编号XY20240001，独家授权销售。",
        "expected_compliant": "no",
        "expected_violations": ["V09"],
        "difficulty": "medium",
        "note": "备案编号真实性存疑+'特别批准'不存在此类表述，构成V09",
    },
    {
        "id": "V09-neg-001",
        "text": "本产品备案编号请登录银保监会官网查询，销售资质公示于门店。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "引导官方渠道核查，合规资质公示",
    },

    # ── V10: 不当使用客户信息 ─────────────────────────────────────────

    {
        "id": "V10-pos-001",
        "text": "老客户李先生投入50万，三年翻倍，这是他的真实案例！",
        "expected_compliant": "no",
        "expected_violations": ["V10"],
        "difficulty": "medium",
        "note": "展示具名客户案例（即使匿名化也需授权），触发V10",
    },
    {
        "id": "V10-neg-001",
        "text": "根据我们的市场调研，超过80%的用户对产品表示满意。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "统计数据非个人信息，合规",
    },

    # ── V11: 违规承诺增值服务 ─────────────────────────────────────────

    {
        "id": "V11-pos-001",
        "text": "购买即送价值3000元免费体检套餐，还有机会赢取免费旅游大奖！",
        "expected_compliant": "no",
        "expected_violations": ["V11"],
        "difficulty": "easy",
        "note": "'送体检'+'免费旅游'，明确违规增值承诺",
    },
    {
        "id": "V11-pos-002",
        "text": "老客户专享返佣，累计购买可获保单增值礼包。",
        "expected_compliant": "no",
        "expected_violations": ["V11"],
        "difficulty": "medium",
        "note": "'返佣'+'增值礼包'触发V11",
    },
    {
        "id": "V11-neg-001",
        "text": "新客户可享受首月免费咨询服务，专属顾问一对一解答。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "medium",
        "note": "免费咨询属于服务说明，非违规增值承诺",
    },

    # ── 复合违规案例 ──────────────────────────────────────────────────

    {
        "id": "MIX-001",
        "text": "明星XX倾情推荐！年化8%固定收益，本金完全保障，全市场最优产品，立即购买送体检！",
        "expected_compliant": "no",
        "expected_violations": ["V01", "V02", "V03", "V04", "V05", "V11"],
        "difficulty": "easy",
        "note": "集齐多个典型违规的极端反面教材",
    },
    {
        "id": "MIX-002",
        "text": "历史业绩稳健，过去五年无一年亏损，适合追求稳定的您，欢迎咨询。",
        "expected_compliant": "no",
        "expected_violations": ["V01", "V02", "V05"],
        "difficulty": "hard",
        "note": "无任何违规关键词，但隐含本金保障+确定收益暗示，且缺失风险提示",
    },

    # ── 完全合规案例 ──────────────────────────────────────────────────

    {
        "id": "COMP-001",
        "text": (
            "本产品由XX人寿保险股份有限公司承保（备案编号请登录银保监会官网查询）。"
            "产品存在投资风险，历史业绩不代表未来收益，请根据自身风险承受能力谨慎选择。"
            "犹豫期为10个自然日，犹豫期内退保不收取任何费用；犹豫期后退保将产生损失。"
            "管理费率0.8%/年，申购费0.3%，详见合同费用说明。"
        ),
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "包含完整风险提示、费用披露、资质公示，完全合规",
    },
    {
        "id": "COMP-002",
        "text": "本基金产品风险等级为R3（中等），适合平衡型投资者，投资须谨慎，过往业绩不预示未来。",
        "expected_compliant": "yes",
        "expected_violations": [],
        "difficulty": "easy",
        "note": "标准合规表述，无任何违规信号",
    },
]


# Retrieved-chunk annotations for retrieval evaluation.
# Each entry maps a case id to the set of chunks that should be retrieved
# for that case based on its expected violation types.
RELEVANT_CHUNK_IDS: dict[str, list[str]] = {
    "MIX-001": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_15_2', '互联网保险业务监管办法_15_5', '互联网保险业务监管办法_17_2', '互联网保险业务监管办法_7_1', '保险销售行为管理办法_16', '保险销售行为管理办法_17_1', '保险销售行为管理办法_17_4', '保险销售行为管理办法_17_5', '保险销售行为管理办法_25_3', '保险销售行为管理办法_37_2', '保险销售行为管理办法_6', '保险销售行为管理办法_9_1', '金融产品网络营销管理办法_14', '金融产品网络营销管理办法_16_1', '金融产品网络营销管理办法_16_2', '金融产品网络营销管理办法_18', '金融产品网络营销管理办法_22', '金融产品网络营销管理办法_25', '金融产品网络营销管理办法_26', '金融产品网络营销管理办法_33_1', '金融产品网络营销管理办法_8_1', '金融产品网络营销管理办法_9_2', '金融产品网络营销管理办法_9_4', '金融产品网络营销管理办法_9_5'],
    "MIX-002": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_15_5', '互联网保险业务监管办法_17_2', '保险销售行为管理办法_25_3', '保险销售行为管理办法_6', '金融产品网络营销管理办法_8_1', '金融产品网络营销管理办法_9_4'],
    "V01-hard-001": ['金融产品网络营销管理办法_9_4'],
    "V01-pos-001": ['金融产品网络营销管理办法_9_4'],
    "V01-pos-002": ['金融产品网络营销管理办法_9_4'],
    "V02-hard-001": ['互联网保险业务监管办法_15_5', '金融产品网络营销管理办法_9_4'],
    "V02-pos-001": ['互联网保险业务监管办法_15_5', '金融产品网络营销管理办法_9_4'],
    "V02-pos-002": ['互联网保险业务监管办法_15_5', '金融产品网络营销管理办法_9_4'],
    "V03-hard-001": ['互联网保险业务监管办法_15_5', '保险销售行为管理办法_17_4', '保险销售行为管理办法_17_5', '保险销售行为管理办法_37_2', '保险销售行为管理办法_9_1', '金融产品网络营销管理办法_33_1', '金融产品网络营销管理办法_9_2', '金融产品网络营销管理办法_9_5'],
    "V03-pos-001": ['互联网保险业务监管办法_15_5', '保险销售行为管理办法_17_4', '保险销售行为管理办法_17_5', '保险销售行为管理办法_37_2', '保险销售行为管理办法_9_1', '金融产品网络营销管理办法_33_1', '金融产品网络营销管理办法_9_2', '金融产品网络营销管理办法_9_5'],
    "V03-pos-002": ['互联网保险业务监管办法_15_5', '保险销售行为管理办法_17_4', '保险销售行为管理办法_17_5', '保险销售行为管理办法_37_2', '保险销售行为管理办法_9_1', '金融产品网络营销管理办法_33_1', '金融产品网络营销管理办法_9_2', '金融产品网络营销管理办法_9_5'],
    "V04-pos-001": ['互联网保险业务监管办法_15_2', '互联网保险业务监管办法_7_1', '保险销售行为管理办法_16', '保险销售行为管理办法_17_1', '金融产品网络营销管理办法_14', '金融产品网络营销管理办法_16_1', '金融产品网络营销管理办法_16_2', '金融产品网络营销管理办法_18', '金融产品网络营销管理办法_22', '金融产品网络营销管理办法_25', '金融产品网络营销管理办法_26'],
    "V04-pos-002": ['互联网保险业务监管办法_15_2', '互联网保险业务监管办法_7_1', '保险销售行为管理办法_16', '保险销售行为管理办法_17_1', '金融产品网络营销管理办法_14', '金融产品网络营销管理办法_16_1', '金融产品网络营销管理办法_16_2', '金融产品网络营销管理办法_18', '金融产品网络营销管理办法_22', '金融产品网络营销管理办法_25', '金融产品网络营销管理办法_26'],
    "V05-hard-001": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_17_2', '保险销售行为管理办法_25_3', '保险销售行为管理办法_6', '金融产品网络营销管理办法_8_1'],
    "V05-pos-001": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_15_5', '互联网保险业务监管办法_17_2', '保险销售行为管理办法_25_3', '保险销售行为管理办法_6', '金融产品网络营销管理办法_8_1', '金融产品网络营销管理办法_9_4'],
    "V06-pos-001": ['互联网保险业务监管办法_15_5', '互联网保险业务监管办法_23_2', '保险销售行为管理办法_17_5'],
    "V07-hard-001": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_14_6', '互联网保险业务监管办法_23_2', '互联网保险业务监管办法_23_5', '互联网保险业务监管办法_27_3', '互联网保险业务监管办法_40', '保险销售行为管理办法_25_3', '保险销售行为管理办法_9_1', '金融产品网络营销管理办法_17_2', '金融产品网络营销管理办法_8_1'],
    "V07-pos-001": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_14_6', '互联网保险业务监管办法_23_2', '互联网保险业务监管办法_23_5', '互联网保险业务监管办法_27_3', '互联网保险业务监管办法_40', '保险销售行为管理办法_25_3', '保险销售行为管理办法_9_1', '金融产品网络营销管理办法_17_2', '金融产品网络营销管理办法_8_1'],
    "V08-pos-001": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_27_3', '互联网保险业务监管办法_40', '保险销售行为管理办法_25_3', '保险销售行为管理办法_32_2', '保险销售行为管理办法_37_1', '保险销售行为管理办法_39', '保险销售行为管理办法_40_1', '保险销售行为管理办法_40_2', '保险销售行为管理办法_40_3'],
    "V08-pos-002": ['互联网保险业务监管办法_14_2', '互联网保险业务监管办法_27_3', '互联网保险业务监管办法_40', '保险销售行为管理办法_25_3', '保险销售行为管理办法_32_2', '保险销售行为管理办法_37_1', '保险销售行为管理办法_39', '保险销售行为管理办法_40_1', '保险销售行为管理办法_40_2', '保险销售行为管理办法_40_3'],
    "V09-pos-001": ['互联网保险业务监管办法_12_5', '互联网保险业务监管办法_14_1', '互联网保险业务监管办法_7_1'],
    "V10-pos-001": ['互联网保险业务监管办法_13_4', '互联网保险业务监管办法_38_1', '互联网保险业务监管办法_38_2', '互联网保险业务监管办法_38_3', '互联网保险业务监管办法_4_2', '保险销售行为管理办法_7_1', '保险销售行为管理办法_7_2', '金融产品网络营销管理办法_1', '金融产品网络营销管理办法_17_3', '金融产品网络营销管理办法_21_2', '金融产品网络营销管理办法_28_3', '金融产品网络营销管理办法_33_2', '金融产品网络营销管理办法_4'],
}

# Inject relevant_chunk_ids into GOLDEN_SET
for _case in GOLDEN_SET:
    _case["relevant_chunk_ids"] = RELEVANT_CHUNK_IDS.get(_case["id"], [])


def get_cases_by_violation(violation_id: str) -> list[GoldenCase]:
    """按违规类型过滤样本"""
    return [c for c in GOLDEN_SET if violation_id in c["expected_violations"]]


def get_cases_by_difficulty(difficulty: str) -> list[GoldenCase]:
    """按难度过滤样本"""
    return [c for c in GOLDEN_SET if c["difficulty"] == difficulty]


def get_negative_cases() -> list[GoldenCase]:
    """返回所有违规（非合规）案例"""
    return [c for c in GOLDEN_SET if c["expected_compliant"] == "no"]


def get_positive_cases() -> list[GoldenCase]:
    """返回所有合规案例"""
    return [c for c in GOLDEN_SET if c["expected_compliant"] == "yes"]


if __name__ == "__main__":
    from collections import Counter
    violations_count = Counter()
    for case in GOLDEN_SET:
        for v in case["expected_violations"]:
            violations_count[v] += 1

    print(f"总样本数: {len(GOLDEN_SET)}")
    print(f"违规样本: {len(get_negative_cases())}")
    print(f"合规样本: {len(get_positive_cases())}")
    print(f"\n按难度分布:")
    for d in ["easy", "medium", "hard"]:
        print(f"  {d}: {len(get_cases_by_difficulty(d))}")
    print(f"\n各违规类型覆盖数:")
    for v, count in sorted(violations_count.items()):
        print(f"  {v}: {count}")
