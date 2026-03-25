# Antool 🐜 **智能媒体管家**  

> 2026-03-24: **Antool-Next** 火热开发中
>- **媒体文件管理** 高性能数据库检索
>- **可视化管理应用** 基于.Net 8 应用程序提供更好体验
>- **插件系统** 编写/导入插件以保留个性的使用习惯

[English](README-EN.md) | [中文版](README.md)

[//]: # ([![GitHub Stars]&#40;https://img.shields.io/github/stars/yourname/Antool?style=flat-square&#41;]&#40;https://github.com/makerlinck/Antool&#41;)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.13-blue)

<!-- <div align="center">
  <img src="docs/v1/images/eagle_00.png" width="30%" alt="界面预览"/>
  <img src="docs/v1/images/eagle_01.png" width="30%" alt="标签管理"/> 
  <img src="docs/v1/images/eagle_02.png" width="30%" alt="智能分类"/>
</div> -->

---
## 项目概述
基于 ** Deepdanbooru ** 的智能媒体文件标签系统，帮助实现自动化图片媒体分类

### 技术栈与架构

- 后端：基于 Python 3.13 LiteStar 框架开发，集成轻量级图像识别模型，实现高性能标签生成服务。项目整体遵循整洁架构

- 前端：通过 Eagle 媒体管理器内置 Node.js 插件 提供可视化交互界面，支持标签管理、分类浏览等核心功能。


## 快速部署

### 环境准备

>注意:该步骤目前仅适用于开发调试

```bash
# 1.安装依赖
$ poetry install
$ poetry lock
# 2.模型文件
$ wget https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip
# 3.将模型文件解压后的内容放入项目目录/resources/models/中。
# 类似resources\models\v3-20211112-sgd-e28\model-resnet_custom_v3.h5 以及 resources\models\v3-20211112-sgd-e28\tags.txt

```

>

### 基本使用
``` bash
-----------------启动服务----------------
poetry run uvicorn src.apps.main:app
```
### CLI 工具
使用 `antool` 命令行工具批量处理图片：

```bash
# 安装后直接使用
antool image1.jpg image2.png
antool ./images/*.jpg --url ws://localhost:8000/ws/evaluate
antool ./images/ --verbose  # 详细模式

# 或
python -m cli.tagging_request "<相对路径>" --url ws://127.0.0.1:8000/ws/evaluate
```

### WebSocket API
WebSocket 长连接协议，支持心跳、流式返回、任务取消：

```javascript
// Node.js 调用示例
const ws = new WebSocket('ws://127.0.0.1:8000/ws/evaluate');

ws.onopen = () => {
  // 发送图片数据
  ws.send(JSON.stringify({
    type: 'submit',
    images: [
      { path: 'image1.jpg', data: base64Image1 },
      { path: 'image2.png', data: base64Image2 }
    ],
    verbose: false,
    streaming_mode: true  // true=流式返回, false=批量返回
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'result':      // 流式模式：单张图片结果
      console.log(`Path: ${data.path}, UID: ${data.uid}`);
      console.log('Rating:', data.rating);
      console.log('Tags:', data.tags);
      break;
    case 'results':     // 批量模式：所有结果
      data.results.forEach(r => console.log(r.path, r.tags));
      break;
    case 'pong':        // 心跳
      break;
    case 'complete':    // 处理完成
      console.log('All done');
      break;
    case 'error':
      console.error(data.message);
      break;
  }
};

// 取消任务
function cancelTask() {
  ws.send(JSON.stringify({ type: 'stop' }));
}
```


## 目录结构
```
Antool
├─ 📁cli
├─ 📁docs
├─ 📁Eagle-Plugin
│  ├─ 📁js
│  │  └─ 📄plugin.js
│  ├─ 📄index.html
│  ├─ 📄logo.png
│  └─ 📄manifest.json
├─ 📁logs
├─ 📁resources
│  ├─ 📁datas
│  ├─ 📁locales
│  └─ 📁models 模型文件目录/标签文件目录
├─ 📁src
│  ├─ 📁apps
│  │  ├─ 📁ws
│  │  │  └─ 📄image_evaluation.py
│  │  ├─ 📄builder.py
│  │  ├─ 📄main.py
│  │  └─ 📄__init__.py
│  ├─ 📁configs
│  ├─ 📁core
│  │  ├─ 📁entities
│  │  │  └─ 📄__init__.py
│  │  ├─ 📁interfaces
│  │  │  ├─ 📄base_file_downloader.py
│  │  │  ├─ 📄base_net_connection_adapter.py
│  │  │  ├─ 📄base_repository.py
│  │  │  ├─ 📄evaluation.py
│  │  │  ├─ 📄file_provider.py
│  │  │  ├─ 📄image_evaluator.py
│  │  │  └─ 📄__init__.py
│  │  ├─ 📁service
│  │  │  ├─ 📄evaluation.py
│  │  │  └─ 📄__init__.py
│  │  └─ 📄__init__.py
│  ├─ 📁infrastructure
│  │  ├─ 📁common
│  │  │  └─ 📄config_provider.py
│  │  ├─ 📁evaluations
│  │  │  ├─ 📄evaluation_pipeline.py
│  │  │  ├─ 📄filter.py
│  │  │  ├─ 📄image_encoder.py
│  │  │  ├─ 📄model_loader.py
│  │  │  ├─ 📄prediction.py
│  │  │  ├─ 📄preprocess.py
│  │  │  ├─ 📄processor.py
│  │  │  ├─ 📄scheduler.py
│  │  │  └─ 📄__init__.py
│  │  ├─ 📁networks
│  │  ├─ 📁repositories
│  │  ├─ 📄cancel.py
│  │  ├─ 📄cli_tagging_request.py
│  │  ├─ 📄logging.py
│  │  └─ 📄metrics.py
│  ├─ 📁interactors
│  │  ├─ 📄evaluate_images.py
│  │  └─ 📄__init__.py
│  └─ 📁monitoring
│     ├─ 📄benchmark.py
│     └─ 📄performance_metric.py
├─ 📁tests
├─ 📄.gitattributes
├─ 📄.gitignore
├─ 📄poetry.lock
├─ 📄pyproject.toml
└─ 📄README.md
```

## 许可协议
本项目采用 MIT License
