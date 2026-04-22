"""
违规类型枚举表（V01-V11 + V00）
"""

VIOLATION_TYPES = {
    "V01": {
        "name": "承诺本金不受损失",
        "keywords": ["保本", "保息", "零风险", "本金无忧", "本金安全", "不损失本金"],
        "articles": ["保险销售第21条", "金融网销第8条"],
        "severity": "critical",
    },
    "V02": {
        "name": "承诺确定收益",
        "keywords": ["年化", "稳稳到手", "固定收益", "确定回报", "保证收益", "稳赚"],
        "articles": ["保险销售第21条", "金融网销第8条"],
        "severity": "critical",
    },
    "V03": {
        "name": "使用绝对化用语",
        "keywords": ["最优", "第一", "最强", "绝无仅有", "最好", "顶级", "唯一"],
        "articles": ["金融网销第11条", "广告法第9条"],
        "severity": "warning",
    },
    "V04": {
        "name": "无资质或不当代言",
        "keywords": ["明星推荐", "KOL", "专家推荐", "内部人士", "权威认证"],
        "articles": ["保险销售第28条", "金融网销第12条"],
        "severity": "critical",
    },
    "V05": {
        "name": "缺失风险提示",
        "keywords": ["风险提示", "保险公司承保", "犹豫期", "退保损失"],
        "articles": ["保险销售第23条", "金融网销第9条"],
        "severity": "critical",
    },
    "V06": {
        "name": "误导性产品比较",
        "keywords": ["比其他", "不如我们", "业内最优", "贬低", "碾压"],
        "articles": ["保险销售第26条", "金融网销第11条"],
        "severity": "warning",
    },
    "V07": {
        "name": "隐瞒/淡化费用",
        "keywords": ["免保费", "零手续费", "无管理费", "无费用", "免费"],
        "articles": ["保险销售第21条", "保险销售第25条"],
        "severity": "critical",
    },
    "V08": {
        "name": "诱导退保/转保",
        "keywords": ["退保", "转保", "升级换代", "旧保单", "不划算", "换了买"],
        "articles": ["保险销售第27条"],
        "severity": "critical",
    },
    "V09": {
        "name": "伪造/篡改备案信息",
        "keywords": ["银保监会批准", "备案编号", "特别批准", "监管备案"],
        "articles": ["互联网保险第34条"],
        "severity": "critical",
    },
    "V10": {
        "name": "不当使用客户信息",
        "keywords": ["客户案例", "内部客户", "专享", "仅限老客户"],
        "articles": ["金融网销第9条", "个人信息保护法"],
        "severity": "warning",
    },
    "V11": {
        "name": "违规承诺增值服务",
        "keywords": ["送体检", "保单变现", "免费旅游", "送礼", "返佣"],
        "articles": ["保险销售第22条"],
        "severity": "warning",
    },
    "V00": {
        "name": "未分类/通用合规要求",
        "keywords": [],
        "articles": [],
        "severity": "info",
    },
}
