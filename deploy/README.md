# 上线指南：让原型在浏览器里用「网址」打开

> 目标：把 `app/app.py` 这个 Streamlit 原型放到网上，任何设备（含手机流量）打开网址就能用。
> 简版说明见根目录 `README.md` 第 1.3 节；本文件是完整操作手册。
> 先用 `python deploy\check_network.py` 看**你这台机器当前网络**适合走哪条路（换 WiFi 请重跑）。

---

## 0. 结论速查（本机实测，2026-10-04）

| 路线 | 你会得到 | 特点 | 本机实测 |
|---|---|---|---|
| **A. 局域网** | `http://192.168.10.160:8501` | 0 配置、立刻可用；但参与者必须与你在同一 WiFi | ✅ 始终可用（`run_app.cmd`） |
| **B. 临时公网网址**（Cloudflare 快速隧道） | `https://xxxx.trycloudflare.com` | 免注册、约 20 秒出网址、手机用流量也能开；行程关闭即失效 | ✅ 隧道 precheck 全 PASS，公网 `/_stcore/health` = 200 |
| **C. 持久公网网址**（Streamlit Community Cloud） | `https://xxxx.streamlit.app` | 你的电脑关机也能访问，适合把链接写进作业；需要 GitHub 账号 | ✅ github 0.3s、share.streamlit.io 0.8s 可达 |
| D. 备选（Hugging Face Spaces） | `https://xxxx.hf.space` | 与 C 类似 | ❌ 本机 `huggingface.co` 探测不通（6.1s 超时） |

结论：**临时演示用 B，长期链接用 C，兜底永远可用 A。**

> ⚠️ **2026-10-10 复测（本机换了网络，上面这张表已过期）：只有 A 仍然可用，B 不可用、C 的远端尚未建**
> （C 的**本地仓库**已在 2026-10-10 建好并首提交，见第 2 节；缺的只是 GitHub 远端与 Streamlit 应用）。
> - **A 可用**：本机 `http://127.0.0.1:8501` / 局域网 `http://10.22.109.176:8501`，`/_stcore/health` 实测 **HTTP 200**
>   （启动 1.1 s 就绪；`deploy\stop_web.cmd` 后端口释放，实测通过）；
> - **B 已不可用（死链）**：本机 DNS 是 `10.10.10.11`，它**不做 SRV 解析**
>   （`nslookup -type=srv _origintunneld._tcp.argotunnel.com` 超时；改查 `1.1.1.1` 立刻返回
>   `region1/region2.argotunnel.com:7844`），cloudflared 取不到 Cloudflare 边缘节点，日志停在
>   `edge discovery: ... DNS query failed` → `Initiating shutdown`；最后一版地址
>   `https://modelling-yorkshire-possession-typing.trycloudflare.com` 本机与公网健康检查**都失败**。
>   另外 `deploy/_tunnel.log` 里 2989 条事件中有 **1775 条 error**（`Register tunnel error from server side` 反复出现）——
>   即便 DNS 修好，免费快速隧道本身也没有 uptime 保证，只能当"演示期间"用。
> - **C 的本地仓库已就绪、远端还没建**：本机已 `git init -b main` 并完成首提交（**150 个文件**、提交 `08c3ead`，
>   两套模型已 `-f` 入库，`git status` 干净、仓库内每个文件与工作区逐字节一致 —— 见第 2 节第 1 步）；
>   但 GitHub 远端仓库与 Streamlit 应用还没建，所以"老师随时能打开"的链接目前**不存在**；
>   GitHub 实测 205 ms、share.streamlit.io 200 可达 → 这条路可做。
>
> ⚠️ `python deploy\check_network.py` **探不到 SRV 这一层**（它只请求 `trycloudflare.com` 的 HTTP 端点，
> 返回 405 就判"可用"），所以它报告"临时公网可用"时，仍要按第 1 节**实跑一次**、看到 `Registered tunnel connection`
> 才算数。

---

## 1. 方案 B：临时公网网址（最快，推荐用于用户测试）

1. **双击 `deploy\share_web.cmd`**
   - 首次运行会自动下载 `cloudflared.exe` 到 `deploy\`（约 53 MB，本机实测 5.9 秒；已下载过则直接跳过）；
   - 脚本会先把原型启动在 `127.0.0.1:8501`，再建隧道；
2. 等它打印出形如 `https://office-plates-hebrew-critics.trycloudflare.com` 的那一行——**这就是公网地址**；
   - 手机（流量、热点都行）打开即可，输入→结果→解释三步完整可用；
3. **忘记地址了**：另开一个窗口运行
   ```cmd
   python deploy\show_public_url.py
   ```
   它会从 `deploy\_tunnel.log` 里读出当前地址，并对本机与公网各做一次健康检查；
4. **结束**：在隧道窗口按 `Ctrl+C`，然后运行 `deploy\stop_web.cmd`（释放 8501 并结束 cloudflared）。

**实测记录（本机）**：隧道出口 `lax10`，协议 `http2`，`Registered tunnel connection`；
`本机健康检查 OK (HTTP 200)`、`公网健康检查 OK (HTTP 200)`。

> **2026-10-04 复测（Step 2 加特征那一轮收尾后重启）**：`share_web.cmd` 已补上
> `--logfile "%~dp0_tunnel.log"`，因此 cloudflared 的输出会落到 `deploy\_tunnel.log`，
> `show_public_url.py` 读到的一定是**当前**地址；本次重启得到
> `https://configuring-logical-areas-specifies.trycloudflare.com`
> （本机与公网 `/_stcore/health` 均 HTTP 200，隧道出口 `lax13`）。
> ⚠️ 地址每次重启都会变，写进报告/发给参与者时都要注明"临时演示地址，可能随时失效"。

> ⚠️ **必读风险**：这个公网地址**没有登录、没有口令**，谁拿到网址谁就能打开。
> 本原型不保存任何输入、也不接触真实个人数据，但仍请：只在测试期间开着；不要让参与者填姓名/联系方式；
> 测试一结束就 `stop_web.cmd`；作业里引用时注明"临时演示地址，可能随时失效"。

**排错**：
| 现象 | 原因 | 处理 |
|---|---|---|
| 网页显示 `530` / `1033` | 隧道刚建立，DNS 还没生效 | 等 10–20 秒刷新；仍不行看 `deploy\_tunnel.log` |
| 隧道窗口报连接失败 | 网络拦了出站 443/7844 | 换网络或改用方案 C |
| 想换更快/更稳的协议 | 默认用 `http2`（兼容优先） | 手动运行 `deploy\cloudflared.exe tunnel --url http://localhost:8501 --protocol quic` |
| 手机打不开，电脑能开 | 你只启动了本地服务 | 必须用 share_web.cmd 给出的 `trycloudflare.com` 地址，不是 `192.168.x.x` |

---

## 2. 方案 C：持久公网网址（Streamlit Community Cloud，免费）

> 得到的是 `https://<名字>.streamlit.app`，**你的电脑关机也能访问**，适合写进作业里的"可访问链接"。

**第 0 步 · 先确认工具与账号**
```cmd
git --version                     :: 没有 git 就先装：https://git-scm.com/download/win
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱"
```
再去 https://github.com 注册/登录（免费的 Public 仓库即可）。

**第 1 步 · 把项目提交成一个仓库（关键：模型必须强制入库）**

> ✅ **本机已完成（2026-10-10 实测）**：`git init -b main` + `git config core.autocrlf false` + 首提交 `08c3ead`，
> 共 **150 个文件**入库（其中 `models/` 13 个 = 两套各 6 个 + `.gitkeep`），提交后 `git status` 干净、
> 仓库里每个文件与工作区**逐字节一致**（`app/app.py` 仍是 `9a9e4a42`、`model.pkl` 仍是 `e4423da8`）；
> 另外**已用 clone 出来的副本跑过界面自检**（`python -m scripts.test_app`）：两条人群线、三个页面全绿
> （`TESTAPP_EXIT_0`），也就是"云端只有仓库内容"时界面确实能跑。
> 下面的命令留作**记录与重建**用，你不必再跑一遍；可以直接从第 2 步开始。

`.gitignore` 排除了 `models/*`，而云端**必须**有模型产物，否则界面会报 `ModelNotTrainedError`：

```cmd
cd /d "E:\人机交互技术\homework_1"
git init -b main
git config core.autocrlf false            :: 让仓库里存的就是原文件字节（CRLF），克隆后与 docs/00 §8.3 的 SHA-256 台账逐字节一致
git add .
git add -f models/model.pkl models/candidates.pkl models/quantiles.pkl models/model_card.json models/history_sample.csv models/feature_importance.csv
git add -f models/sleep_health_lifestyle/model.pkl models/sleep_health_lifestyle/candidates.pkl models/sleep_health_lifestyle/quantiles.pkl models/sleep_health_lifestyle/model_card.json models/sleep_health_lifestyle/history_sample.csv models/sleep_health_lifestyle/feature_importance.csv
git add -f reports/figures/*.png          :: 作业要交图，且 .gitignore 默认排除了它们
git commit -m "sleep duration prototype: streamlit app + trained models (2 populations)"
```
> **为什么必须 `-f`、而且两套都要加**：`.gitignore` 的 `models/*` 会把整个 `models/`
> （**含子目录** `models/sleep_health_lifestyle/`）全部排除，而界面侧栏「预测对象」有**两条人群线**，
> 云端两套模型都得在，否则切到"睡眠健康人群"就会报 `ModelNotTrainedError`。
> 入库后先核对：`git ls-files models/` 应列出 **12 个文件**，`models/model.pkl` 与
> `models/sleep_health_lifestyle/model.pkl` 都要在。
> 界面运行时真正读的是 `model.pkl` / `quantiles.pkl` / `model_card.json` / `history_sample.csv`
> （见 `src/predict.py::load_artifacts`），另外两个（`candidates.pkl` / `feature_importance.csv`）
> 是训练台账，一并入库便于复核。
> 模型一共约 13.3 MB（最大 `candidates.pkl` 4.48 MB），远低于 GitHub 单文件 100 MB 限制，**不需要 Git LFS**。

**第 2 步 · 推到 GitHub**

在 GitHub 网页上新建一个**空**仓库（不要勾选 README/.gitignore），然后：

```cmd
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

**第 3 步 · 固定依赖版本（避免"本地能跑、云端报错"）**

模型是用 `scikit-learn 1.6.1` 训练的，云端若装到别的大版本可能出现反序列化告警或失败，所以用实测版本覆盖：

```cmd
copy /y deploy\requirements_cloud.txt requirements.txt
git add requirements.txt
git commit -m "pin dependency versions for cloud"
git push
```
> 想还原本地的宽松版本：`copy /y deploy\requirements_local.txt requirements.txt`（原文已备份在 `deploy/`）。
> 💡 **想少推一次**：把本步提到第 2 步的 `git push` **之前**做（先 `copy` 再一次性 `git push -u origin main`），
> 效果完全相同，只是省掉第二次 push。

**第 4 步 · 在 Streamlit Cloud 建应用**

1. 打开 https://share.streamlit.io ，用 GitHub 账号登录并授权；
2. **Create app** → **Deploy a public app from GitHub**；
3. 填：Repository = 你的仓库；Branch = `main`；
   **Main file path = `app/app.py`**（写错会报 `ModuleNotFoundError: No module named 'src'`）；
4. **Advanced settings** → Python version 选 **3.12**（3.13 也可）；Secrets 留空（本项目不用密钥）；
5. 点 **Deploy**，等 1–3 分钟。看到 `You can now view your Streamlit app` 即成功；
6. 复制浏览器地址栏的 `https://<名字>.streamlit.app`，这就是可以发给任何人的网页地址。

**第 5 步 · 以后怎么更新**：改完代码 `git add -A && git commit -m "..." && git push`，云端会自动重新部署（约 1 分钟）。

**云端为什么不需要数据**：`data/raw`、`data/processed` 没有入库，云端只用 `app/ + src/ + config/ + models/`。
这是刻意的——上线只是"让界面能被访问"，不重新训练，所以云端结果与本地完全一致。

**常见报错表**

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Main file path 填成了 `app.py` 或别的 | 必须填 `app/app.py`（`app/app.py` 会自动把项目根加进 `sys.path`） |
| `FileNotFoundError: models/model.pkl` 或"尚未训练模型" | 模型没进仓库（被 .gitignore 排除） | 回到第 1 步补 `git add -f models/...` 并 push |
| `InconsistentVersionWarning` / 反序列化报错 | 云端 sklearn 与训练版本不同 | 回到第 3 步固定版本 |
| 打开后长时间显示 "Your app is in the oven" | 首次构建装依赖中 | 等待；或 Manage app → 看构建日志 |
| 过一段时间再打开很慢 | 免费层会休眠 | 首次访问等 20–60 秒，属正常 |
| 想删除/改名 | — | 在 share.streamlit.io 的应用菜单里 Delete app |

---

## 3. 方案 D：备选（Hugging Face Spaces）

- 前提：你的网络能访问 `huggingface.co`（**本机实测 6.1 秒超时，不可达**，所以本项目不推荐这条）；
- 做法：新建 Space → SDK 选 **Docker**（自带 Streamlit 模板）或 **Streamlit** → 把仓库文件（含 `models/`）上传，
  入口文件指向 `app/app.py`（Streamlit SDK 模板里可改 `app_file`）；
- 依赖同样建议用 `deploy/requirements_cloud.txt` 的固定版本。

---

## 4. 方案 A：局域网（0 配置，永远可用的兜底）

```cmd
run_app.cmd
```
手机连**同一个 WiFi** → 打开启动时打印的 `http://192.168.x.x:8501`（首次需允许 Windows 防火墙"专用网络"）。
细节见 `user_test/host_guide.md` 第 2 节。

---

## 5. 本地与云端的一致性（为什么上线不用重新训练）

- 云端只跑**界面链路**：`app/app.py` → `src/predict.py` → 读 `models/{model.pkl, quantiles.pkl, model_card.json, history_sample.csv}`（第二人群是 `models/sleep_health_lifestyle/` 下同名的一组）；
- 界面用哪些字段、要不要显示"就寝时刻"输入框，由 `models/model_card.json` 决定
  （当前 70 个特征、不含就寝时刻）→ 所以云端与本机的行为完全一致；
- 依赖版本固定（`deploy/requirements_cloud.txt` = 本地实测版本），避免 pickle 版本告警；
- `.streamlit/config.toml` 是本地与云端**共用**的一份配置，主题一致；
- 一句话：**上线只改变"谁能打开界面"，不改变模型与结论。**

---

## 6. 上线后怎么用于用户测试

- 方案 B/C/D 都属于 `user_test/host_guide.md` 里的**「方式 C 远程自助」**：参与者自己用手机打开链接；
- **必须承认的取舍**：远程拿不到 T1 用时、T1 提示次数、T3 是否真的打开了解释页
  → 这些字段在 `user_test/summary.md` 里标注"远程自助未采集"，**不要编造**，并把方式 C 的样本与 A/B 分开统计；
- 建议问卷字段（与 `user_test/raw.csv` 对齐）：T1 是否完成、T1 大致用时（自估）、
  T2 三问（是否提到区间 / 是否提到不确定 / 是否当成确定事实）、T3 主要因素与方向、
  T5 的五个评分、两个开放问题（哪一步最费劲、最想改哪一处）；
- 参与者用的是自己的手机，请明确告知：**界面不保存你填的任何内容**（项目没有数据库、没有埋点）；
- 若同时做 A/B（同机）与 C（远程），建议 A/B 的样本用于判定通过线，C 只补充主观评分与文案线索。

---

## 7. 安全与隐私清单

- [ ] 公网地址**无鉴权**：只在测试期间开着；测完立刻 `deploy\stop_web.cmd`；
- [ ] 不要让参与者把姓名、电话、学号等个人信息填进界面（界面本身也没有这类输入框）；
- [ ] 公开仓库只放代码 + 模型；`data/`（原始/处理后数据）已被 `.gitignore` 排除，不要手动强加；
- [ ] 界面底部的"课程作业原型，不构成医疗建议"免责声明必须保留（`app/app.py` 已内置）；
- [ ] 作业/报告里引用链接时写清"临时演示地址，可能随时失效"，避免评委点开时已经过期；
- [ ] 不把 `deploy/cloudflared.exe`（约 53 MB 的第三方程序）推进仓库，`.gitignore` 已排除。

---

## 8. 上线检查清单

| # | 检查项 | 命令 / 位置 | 通过标准 |
|---|---|---|---|
| 1 | 网络适合走哪条路 | `python deploy\check_network.py` | 至少一条"可达" |
| 2 | 本地能跑 | `run_app.cmd` | 浏览器出现三步界面 |
| 3 | 公网地址可用（方案 B） | 双击 `deploy\share_web.cmd` | 打印 `https://xxx.trycloudflare.com` |
| 4 | 地址与健康检查（方案 B） | `python deploy\show_public_url.py` | 本机与公网都 `OK (HTTP 200)` |
| 5 | 手机实测 | 用手机流量打开该网址 | 走完"输入 → 结果 → 解释"三步 |
| 6 | 持久网址（方案 C） | 见第 2 节 | 得到 `https://xxx.streamlit.app` 且电脑关机后可访问 |
| 7 | 测试结束 | `deploy\stop_web.cmd` | `port 8501 is free`、`cloudflared is stopped` |
| 8 | **两套模型都进了仓库**（方案 C 前置条件） | `git ls-files models/` | 列出 **12 个文件**，`models/model.pkl` 与 `models/sleep_health_lifestyle/model.pkl` 都在（漏一个，云端切人群就报错） |
| 9 | 仓库内容与交付台账一致（连换行也不能变） | `git show HEAD:app/app.py \| python -c "import sys,hashlib;print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:8])"` | 打印 **`9a9e4a42`**（与 `docs/00` §8.3 台账一致）；`git status` 干净 |

---

## 附录：`deploy/` 目录里有什么

| 文件 | 作用 |
|---|---|
| `README.md` | 本文件：上线指南 |
| `share_web.cmd` | 一键：本机启动原型 + 建 Cloudflare 隧道 → 打印公网 https 地址 |
| `stop_web.cmd` | 关闭隧道进程并释放 8501 端口 |
| `show_public_url.py` | 从 `_tunnel.log` 读出当前公网地址，并对本机/公网做健康检查 |
| `check_network.py` | 网络自检：判断这台机器适合哪种上线方式 |
| `requirements_cloud.txt` | 云端固定依赖（与本地实测版本一致） |
| `requirements_local.txt` | `requirements.txt` 原始宽松版本备份，随时可还原 |
| `_tunnel.log` | 运行时生成（已被 `.gitignore` 排除），`show_public_url.py` 依赖它 |
| `cloudflared.exe` | 首次运行时自动下载（约 53 MB，已被 `.gitignore` 排除） |
