# 第二阶段数据：当前帧、导航条件文本和GT投影标签

最新代码与指令复核见[复核记录](stage2_code_review_20261107.md)：修复空GT校验形状问题，
第二阶段21项测试、公共流水线28项测试及三个真实场景重投影复核通过；8卡实跑尚未进行。

本文补充离线数据生成实现，不修改原方案，也不指定最终训练工程。原方案见
`/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main/docs/steer_drivor_qwen_gt_grounding_analysis.md`。

## 从原方案到当前约定

- 第一阶段使用独立的简单定位提示词，仅五类有GT框的对象，不输入导航。
- 第二阶段固定使用 `scripts/data_generation/prompts/scene_navigation_v2.txt`：每张当前图像加真实当前导航指令，优先描述与该路线相关的可见车辆、行人、车道边界/箭头、斑马线、信号灯/标志、障碍和交通状况。
- 文本包含物体外观、图内位置及清楚可见的道路关系；不固定物体数量，信息丰富的视图目标50–80词，稀疏视图可以更短，最终按RoBERTa最多128 tokens保留完整短语。
- 不使用GT、未来图像或未来轨迹生成文本；导航只表示路线意图，不能据此编造车道、标志、通行权或未来动作。四视图均使用图内左右，包括后视图。
- 提示词排除背景建筑、树木、广告、装饰等。最新版每张图像只调用Qwen一次，不再请求模型重新描述。对`driving in/on/near...`等可明确保留静态位置的表达，只在本地移除动作词，保留原颜色、类型和位置；其他不安全短语删除，保留同相机剩余短语。原始输出、局部清理及删除原因留档。关键词规则无法保证事实正确，仍需图像抽查。
- Qwen-Drive代码和权重、处理器、VQA解码方式沿用现有源代码：`/mnt/zhaozc_workspace/project/LQH/Qwen-drive`，不新增模型依赖。
- NAVSIM的one-hot顺序是左、直、右、未知；Qwen官方指令词表顺序不同，通过指令词映射，不复用下标。全零/未知仍为UNKNOWN。
- 两阶段原图、GT及标定可以共用，但文本缓存分别生成；第二阶段标签依据其最终文本类别重建，不能直接沿用第一阶段的类别选择结果。

## 划分和用途

| 输出目录 | 原生选择规则 | 用途 |
| --- | --- | --- |
| `train/` | navtrain scene filter ∩ DrivoR train日志 | 规划训练、接地辅助监督 |
| `val/` | navtrain scene filter ∩ DrivoR val日志 | 验证、选checkpoint |
| `test/` | navtest scene filter | 固定测试场景的语言输入与最终评测 |

train/val/test冻结同一个生成模型、提示词、预处理和清洗方式；脚本合并时检查生成配置一致和跨划分token无重复。不能根据navtest结果调提示词或选择生成模型。

这里的val是DrivoR原生训练日志划分，并非额外假定存在一个独立navval目录。如果最终实验另行指定官方场景列表，需在生成前明确采用的列表。

第二阶段保留所有成功读取的原生规划场景：没有接地框不删除。清理按相机进行，单相机问题不清空其余三个相机；部分短语不合格也不删除同相机剩余合格短语。仅在该相机JSON无法解析、生成截断或没有任何可用短语时，该相机最终文本为空并跳过接地。逐相机问题记录在review_queue；整个分片文本均为空时也正常导出零接地目标和完整场景索引。缺图、缺日志、模型运行异常仍会明确停止。

如果Qwen只返回`objects`字符串列表，或对象有phrase但缺失/写错category，保留原句并以`other`表示类别不可用，记录格式清理；这部分短语不提供五类GT接地监督。不会猜测类别、颜色或位置，也不会为修复格式再次调用模型。

test导出的GT投影仅用于离线数值审计，不能作为推理特征、文本生成输入或选择测试样本的条件。无论投影有效与否，测试场景均保留。阶段1的“仅有接地目标的训练索引”不能用于决定阶段2的规划/评测列表。

## 输出格式

每个划分的 `scene_text.json` 保持原方案格式：

```json
{
  "<initial_token>": {
    "cam_f0": {"text": "a white car near the center of the marked lane", "categories": ["vehicle"]},
    "cam_b0": {"text": "a dark car in the lower center of the image", "categories": ["vehicle"]},
    "cam_l0": {"text": "lane markings near the lower left of the image", "categories": ["other"]},
    "cam_r0": {"text": "a pedestrian beside the curb on the right of the image", "categories": ["pedestrian"]}
  }
}
```

上面只是格式示例。模型语言输入是四个`text`，顺序固定f0、b0、l0、r0。`categories`用于选择监督类别，不直接作为RoBERTa输入。

| 产物 | 内容 |
| --- | --- |
| `scene_text.json` | 按initial_token索引的四相机最终文本/类别，仅真正无可用内容的相机为空 |
| `dataset_index.jsonl` | 每行一个规划场景：token、日志、四图路径、当前导航、四句文本、审计路径、框/NPZ路径、有效相机、`view_status`和`text_fallback_cameras` |
| `review_queue.json` | 逐相机问题及其审计路径；有问题但仍保留可用文本的视图也记录 |
| `summary.json`、`COMPLETE.json` | 完成数量、版本、生成/续跑统计及完成标记 |
| `shards/rank_XXXX/prepared/` | 原图身份、当前导航；GT与标定单独存放 |
| `shards/rank_XXXX/captions/scenes/<token>.json` | 逐对象phrase/category、一次原始响应、本地清理`local_edits`和删除记录、最终文本、输入/版本哈希；`repairs=[]` |
| `shards/rank_XXXX/grounding/scenes/<token>.json` | 投影候选框、类别、被选中的GT索引 |
| `shards/rank_XXXX/grounding/masks/<token>.npz` | 2D框、相机/GT索引、类别、`soft_mask(4,3936)`、`valid(4,)`等数值标签 |

2D框坐标为缩放后1148×672图像上的`xmin,ymin,xmax,ymax`。投影沿用50m水平距离、0.1m近平面边裁剪、框覆盖并集、14×14patch、48行82列规则。仍是类别级弱接地，五类之外的信号灯、标志、道路标线/其他上下文保留在语言里，不凭空生成其GT框。完整语言和仅五类框混合时的监督范围，需要在后续损失实现中按原方案讨论处理。

画框图片/HTML只用于检查；训练使用原图、文本和数值标签。离线处理不使用GridMask。

## DrivoR-main-copy接入范围

数据独立于最终训练工程，可由该工程按token读取。原生DrivoR目前没有新方案的`scene_text`读入路径。该目录已有SteerViT扩展使用`QwenTextReader`读取`log/token/qwen_text.gz`，每场景返回一段全局文本，再广播给四相机；与这里的逐相机路由不同，不能直接替换缓存路径或把四句拼起来就认为已实现方案。

后续接入需要Dataset/CacheOnlyDataset按token附加四句文本，模型显式处理default_collate后的转置，并在评测时按场景取文本。b-major顺序是：

```python
texts = [scene_text[n][b] for b in range(B) for n in range(4)]
```

独立数据检查入口可用DrivoR-main-copy原生的`_get_camera_feature`加载图像，验证四视图与上述文本顺序。它不运行规划模型或PDMS：

```bash
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main
/mnt/zhaozc_workspace/envs/drivorpython/bin/python -B \
  scripts/data_generation/stage2/check_feature_input.py \
  --dataset outputs/stage2_navigation_no_rewrite_v2_8gpu_pilot/train \
  --scenes 2 --output outputs/stage2_navigation_no_rewrite_v2_8gpu_pilot/train/feature_input_check.json
```

## DLC运行

可复制命令见 `scripts/data_generation/stage2_8gpu_dlc_command.txt`，提供全量和小样本两个区块，选择一个执行。每卡一个冻结VLM，逐张当前图像处理；不是梯度训练。依次完成train、val、test，使用独立目录。默认复用mot_vega、drivorpython及DrivoR-main-copy/dataset，不修改这些参考工程。

`SCENES_PER_RANK=0`是全量；`DEEP_SCENES=20`只是每卡额外重算20场景投影，不限制数据数量。相同设置和输出目录可续跑，改变提示词、代码、卡数或场景上限要使用新目录。

第二阶段新增Python文件放在`stage2/`子目录，已有第一阶段流水线的Python文件和提示词保持原样，避免改变其缓存版本。第一阶段生成和第二阶段生成应分别占用自己的设备资源。

原方案中的50场景Qwen-Drive与通用Qwen-VL比较、完整模型训练、D+对照及文本消融均未由此数据入口实现。已有导航小样本主要是直行，真实左转/右转质量仍需要抽查。规则检查通过不能替代描述事实正确性的检查。

## 旧版实际验证（保留历史，使用模型改写与场景级回退）

- 新增7项检查通过；另模拟8个rank合并，确认包含一个文本失败样本时仍保留8个规划场景。
- train/val/test的978/214/136个原生日志均可访问，8分片覆盖完整且无重复。
- `outputs/stage2_navigation_verified_20261107/` 在本机单卡为三个划分各生成1场景，共12视图，分别完成GT重新投影和mask重建检查；有效接地视图分别为4、3、2，其他视图未因此删样本。
- 三个划分均通过`DrivoR-main-copy`原生`_get_camera_feature`图文接口检查；真实文本另通过B=2、N=4的default_collate/b-major顺序检查。未运行规划模型或PDMS。
- 训练划分再次执行：复用1份文本和1份接地标签，重新生成0场景；再次校验及跨划分配置检查通过。
- 现有第一阶段流水线16个Python文件内容哈希保持不变。八卡指令的全量/小样本区块均通过dry-run；本机只有1卡，未进行8卡实跑，也未启动全量生成。
- 三个真实样本都是直行；验证集样本仍有一条广告牌描述，说明当前提示词不能保证完全排除背景信息。静态改写也可能改变颜色等表述，规则通过不能证明实例身份正确。

各划分的`preview_2d/index.html`包含四视图原图、选中2D框、文本和patch标签，供抽查。详细机器记录为`outputs/stage2_navigation_verified_20261107/review_report.json`。

## 最新版：按相机保留文本，不调用模型改写

版本为`camera_local_no_model_rewrite_v2`。入口改用`stage2/generate.py`，模型加载和VQA处理复用DrivoR-main-copy的Qwen backend及官方Qwen源代码；`stage2/validate.py`在原数值校验基础上重放本地清理，允许单相机或全部相机为空。最新指令文本已改用包含`no_rewrite_v2`的新输出目录，不能混用旧缓存。

17项检查已覆盖单相机失败时保留其他三句及接地标记、整场景失败仍导出、无模型改写、颜色/位置原样保留、非法对象不牵连其他对象，以及缺失类别或字符串列表的原句保留。真实样例结果另见新版输出目录的review_report。

最终实跑为`outputs/stage2_navigation_no_rewrite_v2_final_20261107/`：train/val/test各1场景、共12视图，模型总调用12次，无模型重述；最终12句全部非空，没有观察到无法解析的JSON内容。前视有一次合法JSON的objects字符串列表缺少类别，经本地格式归一化后保留6条原句。三个划分均通过GT重投影/标签检查；相同命令续跑时复用3个场景，新增模型调用0次。

`outputs/stage2_no_rewrite_fault_checks_20261107/`的“坏JSON”是人为注入`not JSON`的故障测试，不是Qwen真实输出。单相机注入只影响该相机，其余三句保留；四相机注入也能导出完整规划场景和零接地标签，均通过原始GT的深度数值检查。详细结果分别在两个目录的review_report.json和report.json。

12视图零空文本不能据此估计全量空文本概率。真实发生的格式偏差是字段结构与预期不符，不能将其描述成Qwen输出了无法解析的JSON。原始响应、格式归一化、语义过滤和最终空文本应分别统计。

空文本回退不是训练时概率0.2的text dropout，两者独立记录。阶段2按原方案仍保留`0.5*s(t)*L_ground`；没有有效目标只跳过该相机的接地项，规划样本保留。无可用信息时不编造文本填充。统计包含`empty_text_views`、`empty_text_fallback_views`、`all_texts_empty_scenes`，便于全量运行后评估真实比例。
