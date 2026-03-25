// ==================== 全局状态控制 ====================
let abortController = null;
let isTaggingActive = false;
let progress = {
  current: 0,
  total: 0,
  cancelled: false
};
let startTime = null; // 任务开始时间戳
let averageTimePerItem = 0; // 每项平均处理时间（秒）
let chunkSize = 32; // 默认分块大小
let maxChunkSize = 192; //
DEFAULT_LANGUAGE = 'en';
SUPPORTED_EXT = ["png","jpg"]

// ==================== UI 模板 ====================
const uiTemplate = () => `
  <div class="container">
    <header class="header">
      <img src="${eagle.plugin.manifest.logo}" class="logo" alt="AntLLM Logo">
      <h1>AntLLM 🐜 智能文件管理 </h1>
      <h1>v${eagle.plugin.manifest.version}</h1>
    </header>

    <div class="config-group">
      <div class="input-group">
        <label>分块大小：</label>
        <input 
          type="number" 
          id="chunkSize" 
          min="1" 
          max="${maxChunkSize}" 
          value="${chunkSize}"
          ${isTaggingActive ? 'disabled' : ''}
          onchange="updateChunkSize(this.value)"
        >
        <span class="hint">(1-${maxChunkSize})</span>
      </div>
    </div>

    <div class="control-group">
      <button class="btn primary" onclick="handleTagging(false)">
        <span class="icon">🏷️</span>智能打标
      </button>
      <button class="btn warning" onclick="confirmForceRefresh()">
        <span class="icon">🔄</span>强制刷新
      </button>
      <button class="btn danger" onclick="handleCancel()" ${!isTaggingActive ? 'disabled' : ''}>
        <span class="icon">⏹️</span>${progress.cancelled ? '正在取消...' : '取消操作'}
      </button>
    </div>
    <div class="control-group-2">
        <button class="btn warning" onclick="confirmRemoveTags()">
            <span class="icon">🔄</span>清除选中标签
        </button>
      </button>
    </div>


    ${progress.total > 0 ? `
    <div class="progress-container">
      <div class="progress-bar" style="width: ${Math.round((progress.current / progress.total) * 100)}%"></div>

      <div class="progress-text">
        ${progress.current}/${progress.total} (${Math.round((progress.current / progress.total) * 100)}%)
      </div>
    </div>` : ''}

    <div class="log-container" id="log">
      ${progress.total === 0 ? '<div class="empty-state">🖼️ 选择文件后开始智能管理</div>' : uiPredictTime()}
    </div>
  </div>

  <style>
    :root {
      --primary-color: #2c3e50;
      --accent-color: #3498db;
      --warning-color: #e67e22;
      --danger-color: #e74c3c;
    }

    .container {
      padding: 24px;
      max-width: 800px;
      margin: 0 auto;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }

    .header {
      text-align: center;
      margin-bottom: 2rem;
    }

    .logo {
      width: 64px;
      height: 64px;
      margin-bottom: 1rem;
    }

    .config-group {
      background: #f8f9fa;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 1.5rem;
    }

    .input-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    input[type="number"] {
      padding: 6px 12px;
      border: 1px solid #ddd;
      border-radius: 4px;
      width: 80px;
    }

    .hint {
      color: #666;
      font-size: 0.9em;
    }

    .control-group {
      display: flex;
      gap: 12px;
      margin: 2rem 0;
      justify-content: center;
    }
    .control-group-2 {
      display: flex;
      gap: 12px;
      margin: 2rem 0;
      justify-content: end;
    }

    .btn {
      padding: 10px 20px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 500;
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .primary { background: var(--accent-color); color: white; }
    .warning { background: var(--warning-color); color: white; }
    .danger { background: var(--danger-color); color: white; }

    .progress-container {
      height: 28px;
      background: #eee;
      border-radius: 14px;
      overflow: hidden;
      position: relative;
      margin: 2rem 0;
      box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
    }

    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-color), #2980b9);
      transition: width 0.3s ease;
    }

    .progress-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: white;
      font-weight: bold;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    .log-container {
      border: 1px solid #eee;
      border-radius: 8px;
      padding: 16px;
      max-height: 300px;
      overflow-y: auto;
      background: white;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .log-container div {
      padding: 8px 12px;
      margin: 4px 0;
      background: #f8f9fa;
      border-radius: 4px;
      font-family: monospace;
    }

    .empty-state {
      text-align: center;
      color: #666;
      padding: 2rem !important;
    }

    .icon {
      font-size: 1.1em;
    }
  </style>
`;

// ==================== 功能函数 ====================
function updateChunkSize(value) {
  const size = Math.min(Math.max(parseInt(value), 1), maxChunkSize);
  chunkSize = isNaN(size) ? 16 : size;
}

function confirmForceRefresh() {
  const confirmed = confirm("⚠️ 强制刷新将覆盖现有标签！\n\n确定要继续吗？");
  if (confirmed) handleTagging(true);
}
function confirmRemoveTags() {
  const confirmed = confirm("⚠️ 将清除现有标签！\n\n确定要继续吗？");
  if (confirmed) removeTags();
}

// ==================== 核心函数 ====================
async function handleTagging(force) {
  if (isTaggingActive) {
    addLog('已有任务进行中，请先取消');
    return;
  }

  try {
    startTime = new Date().getTime(); // 记录任务开始时间
    abortController = new AbortController();
    isTaggingActive = true;
    progress = { current: 0, last_finish: 0, total: 0, cancelled: false };
    updateUI();

    const items = await eagle.item.getSelected();
    const [uris, objs] = processItems(items, force);

    if (uris.length === 0) {
      addLog('没有需要处理的文件');
      return;
    }

    progress.total = uris.length;
    updateUI();

    // 使用动态分块大小
    for (let i = 0; i < uris.length; i += chunkSize) {
      if (progress.cancelled) {
        addLog('操作已取消');
        break;
      }

      const chunkUris = uris.slice(i, i + chunkSize);
      const chunkObjs = objs.slice(i, i + chunkSize);

      await processChunk(chunkUris, chunkObjs);

      progress.current = Math.min(i + chunkSize, uris.length);
      progress.finished = Math.min(i + chunkSize, uris.length);
      updateUI();
    }
    if (progress.cancelled) updateUI();
    addLog('✅ 处理完成');
  } catch (error) {
    addLog(`❌ Error: ${error.message}`);
  } finally {
    cleanup();
  }
}

// ==================== 工具函数 ====================
function uiPredictTime() {
    let remainingText = '';
    if (progress.current > 0 && averageTimePerItem > 0) {
        const remainingItems = progress.total - progress.current;
        const remainingSeconds = remainingItems * averageTimePerItem;
        const minutes = Math.floor(remainingSeconds / 60);
        const seconds = Math.round(remainingSeconds % 60);
        return ` 剩余时间: ${minutes}分${seconds}秒`;
    } else {
        return ' 剩余时间: --';
}}
async function processChunk(uris, objs) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket('ws://127.0.0.1:8000/ws/tagger');
    let results = []; // 存储所有进度结果

    socket.addEventListener('message', (event) => {
      const data = JSON.parse(event.data);
      if (data.error) {
        reject(new Error(data.error));
        return;
      }

      if (data.status === 'progressing') {
        results.push(data.content);
        if (progress.finished + data.progress > progress.current) {
        progress.current = progress.finished + data.progress;

        } // 更新进度并防止远程的进程回溯进度

        // 计算平均时间
        const currentTime = new Date().getTime();
        const elapsedSeconds = (currentTime - startTime) / 1000;
        averageTimePerItem = elapsedSeconds / progress.current;
        updateUI();
      } else if (data.status === 'done') {
        handleResults(results, objs);
        resolve();
      }
    });

    socket.addEventListener('open', () => {
      // 发送请求数据
      socket.send(JSON.stringify({
        tag_language: DEFAULT_LANGUAGE,
        query_uris: uris
      }));
    });

    socket.addEventListener('close', () => {
      if (results.length === 0) {
        reject(new Error('连接意外关闭'));
      }
    });

    socket.addEventListener('error', (error) => {
      reject(error);
    });

    // 取消操作时关闭连接
    abortController.signal.addEventListener('abort', () => {
      socket.close();
      reject(new Error('操作被取消'));
    });
  });
}

// 辅助函数：处理累积的 results 并更新文件标签
function handleResults(results, objs) {
  results.forEach(item => {
    const idx = item.img_seq[0];
    objs[idx].tags = item.img_tags;
    objs[idx].save();
    addLog(`已处理: ${objs[idx].name}`);
  });
}


async function removeTags() {
    const items = await eagle.item.getSelected();
    for(i in items){
        items[i].tags = [];
        items[i].save();
    }
}
function handleCancel() {
  if (isTaggingActive) {
    progress.cancelled = true;
    abortController.abort();
    addLog('操作取消');
    updateUI();
  }
}
function processItems(items, force) {
  const path = require('path');
  return items.reduce((acc, item) => {
    const IS_SUPPORTED = SUPPORTED_EXT.includes(item.ext);
    if ((force || item.tags.length === 0) && IS_SUPPORTED) {
      const posixPath = `${eagle.library.path.split(path.sep).join(path.posix.sep)}/images/${item.id}.info/${item.name}.${item.ext}`;
      acc[0].push(posixPath);
      acc[1].push(item);
    } else if (!IS_SUPPORTED) {
      addLog(`跳过: ${item.name} (不支持的类型:${item.ext})`);
    } else {
      addLog(`跳过: ${item.name} (已存在${item.tags.length}个标签)`);
    }
    return acc;
  }, [[], []]);
}

function addLog(message) {
  const logEl = document.getElementById('log');
  const entry = document.createElement('div');
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

function updateUI() {
  document.querySelector('#message').innerHTML = uiTemplate();
}

function cleanup() {
  isTaggingActive = false;
  abortController = null;
  setTimeout(updateUI, 2000);
}

// ==================== 插件初始化 ====================
eagle.onPluginCreate(async (plugin) => {
  document.querySelector('#message').innerHTML = uiTemplate();
});