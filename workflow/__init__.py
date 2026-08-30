"""工作流子包：LangGraph 图定义、节点与状态契约。

注意：本文件必须保持「无副作用」。
历史上这里没有 __init__.py，导致 `import workflow` 命中的是
workflow/workflow.py 模块而非本包，进而使 `from workflow.schemas import ...`
失败、只能靠 try/except 回退导入。补充 __init__.py 即修复该歧义。

同时不要在 __init__.py 里 re-export create_workflow：
nodes -> skills.test_design_skills -> workflow.schemas 是一条环，
一旦 __init__ 触发 .workflow 的加载，就会出现「部分初始化的模块」导入失败。
保持本文件为空可让 `workflow` 包瞬间完成初始化，从而彻底避免该环。
"""
