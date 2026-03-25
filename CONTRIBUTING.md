# **Antool 贡献指南**

感谢你愿意花时间和精力为这个项目做出贡献，你的任何形式的贡献，无论是提交 Issue、修复 Bug 还是完善文档，对它的成长都很有价值！接下来请花上几分钟了解一下一些事。


## 行为准则

我们采用了
[Contributor Covenant](https://www.contributor-covenant.org/)
作为行为准则。参与本项目即表示你同意遵守其条款。请阅读这篇
[贡献行为守则](docs\v1\contributions\CODE_OF_CONDUCT.md)


## 如何提交 Issue

在提交 Issue 之前，请先搜索**已有的** Issues 以避免重复。

- **报告 Bug**：请使用提供的 `Bug 报告` 模板，包含：环境信息、复现步骤、预期结果与实际结果。
- **提出新功能**：请详细描述你的使用场景以及该功能能解决什么问题，而不仅仅是“我想要一个 X 功能”。


## 开始开发

1. **Fork 本仓库** 并克隆到本地。
```bash
git clone
```

2. **安装依赖**：

```bash
# 确保poetry环境已安装完毕
poetry install
poetry lock
```
*_**提示**: 在部分开发环境中包tensorflow很有可能安装失败，请尝试一些网络社区提供的方法，它们大多是有效的。*

### 5. 代码规范与提交规范

明确代码风格和 Commit Message 的格式，这有助于自动化生成 Changelog。


## 代码规范

- **代码风格**

本项目遵循
[PEP 8](https://www.python.org/dev/peps/pep-0008/)
规范。使用 `black` 进行自动格式化，`isort` 管理导入顺序，`flake8` 进行代码检查。请在提交前运行以下命令：

``` bash
black .           # 自动格式化
isort .           # 排序导入
flake8            # 静态检查
```


- **测试**：新功能必须包含相应的单元测试或集成测试。请确保所有测试通过后再提交：

```bash
pytest            # 运行测试
```

- **Commit 信息**：请遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。
  - `feat: add api of get current model information`
  - `fix: fix evaluation-module raise errors on working`
  - `docs: update deployment-guide in README.md`



## 提交 Pull Request (PR)

1. 确保你的 Fork 与主仓库保持同步。
2. 创建一个新的功能分支：`git checkout -b feat/your-feature-name`。
3. 提交你的更改（请遵循上述 Commit 规范）。
4. 推送到你的 Fork 仓库。
5. 在本仓库页面点击 “New Pull Request”。
6. **填写 PR 模板**：
   - 关联相关的 Issue 编号（如 Closes #123）。
   - 描述你做了什么改动。
   - 附上测试结果截图。


### *PR 审查

- 至少需要一名维护者的批准。
- CI（持续集成）检查必须全部通过（包括代码格式、单元测试）。
- 如果 PR 长时间未响应，请礼貌地在评论区 @ 维护者。


## 版本说明
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
- 修复补丁（Patch）：每周末发布（如有）。
- 次要版本（Minor）：每月发布一次。
- 主要版本（Major）：请等待公告。
