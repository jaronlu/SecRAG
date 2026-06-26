# SecRAG Agent Guide

> 本文件仅做重定向，所有项目指令见 [CLAUDE.md](./CLAUDE.md)。

## 关键强约束

- **禁止使用任何 Deprecation / Sunset 标记 API**；必须使用当前稳定版官方最新 API。
- 遇到 `langchain-community` 被弃用提示时，不能只“忽略警告”，必须迁移到独立集成包或社区最新替代方案。
- 设计文档中的代码示例同样受此规则约束，发现一处即视为待修复问题。
