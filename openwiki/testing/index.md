# 文件

- [测试与评估：pytest 布局、检索评估与权限冒烟](evaluation.md) - SecRAG 的质量门禁地图：tests/ 单元测试与 tests/e2e/ TC 编号端到端用例的布局与 isolated_stores 隔离 fixture，scripts/evaluate_retrieval.py 的 recall@5/recall@10/MRR/precision@5/覆盖率/权限拦截准确率指标与准入阈值，check_permissions.py 的 RBAC 冒烟检查，LLM-as-Judge 回答质量评估（evaluate_answers_e2e.py + answer_judge.py 四维度），消融实验（evaluate_ablation.py），以及内置评估集规模很小的局限。
