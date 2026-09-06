# OpenBoost v1 F0.1：算法任务卡、替代方案与接口草图

日期：2026-09-05。代码审阅基线：`7b57436`。
状态：**F0.1 任务规格；v1 实现、独立参考程序和正式评测尚未完成**。
依据 [主计划](agent-boosting-foundation-plan.md)、[应用范围](foundation-application-contracts.md)
和 [v1-plan-r2 eval](openboost-v1-evaluation.md)。A1–A13 全部 required；执行顺序只由依赖决定。

本文件确定任务、数学约定、数据选择、对照路径及可判错的结果。数据下载后的 bytes/
split hash、实际 wheel/driver、16 个搜索配置和资源预算由 F0.3 冻结，不在这里假装
已经测过。F0.2 编写独立参考；F1 编写 production；F2/F3/F4 分别产出作者、GPU、
真实任务证据。本文中的未来测试/模块名是交付约定，不是已存在的 API。
具体怎样构建这些公共组件，见 [foundation 工程构建设计](foundation-construction-design.md)：
数据结构、算子契约、建树/事务/推理/设备执行与 B01–B14 实施切片；本文的任务卡不能替代该设计。

## 1. 共用契约和交付矩阵

### 输入、两轮状态及失败语义

- 原始数据按行对齐：`X[N,F]`、稳定 `row_id[N]`、typed target、可选 `weight[N]`、
  `offset[N,K]`、query/entity ID 与 `structure_input`。权重有限、非负、总和正；
  合法缺失、删失边界与非法 NaN 分别校验。辅助量不会自动转成训练权重。
- Numeric、missing、CPU categorical 按 C1 实现。类别映射由训练集建立，缺失作为
  专用状态，未见类别走预声明 missing route；mapping 与 route 随模型保存。
  F1 的首个类别算法选 one-category-vs-rest 穷举，避免隐藏 target encoding。
  这是起点，若质量不达标可改算法，不能删掉类别任务或放宽 E3。
- 所有小 fixture 至少两轮：`raw0 → geometry0 → learner0 → decision0 → raw1 →
  geometry1 → learner1 → decision1 → raw2`。同时比较中间量、树/系数、训练与验证
  预测、保存后推理。只返回正确 shape、只在第一轮生效、复用旧梯度均判失败。
- 平滑二阶任务使用未加权逐行 g/h，统计入口统一乘训练 weight 一次；曲率若为
  近似必须另命名。分布/Formula 的方向拟合也明确区分方向和其回归权重。
- 最小 scalar Newton 参考：`v=-G/(H+lambda)`；收益为左右和减父节点的
  `0.5*G^2/(H+lambda)`，再减 split penalty。lambda/penalty 不按节点行数隐式缩放。
  零有效质量节点不分裂；分母非法失败。默认 fixture `lambda=1, eta=0.1, depth=2`，
  两轮、全行/全列、无 early stopping；专门反例会显式覆盖这些参数。
- 同收益按 `(feature_id, candidate_id, missing_direction)` 字典序选择；CPU reference
  固定 dtype/归约顺序。Depthwise、best-first 和 symmetric 各有独立 topology 检查，
  不要求不同 policy 生成同一棵树。Symmetric 每层共用一个分裂条件，按活动节点收益
  总和选择；不可行候选不能靠负数/NaN 哨兵误选。
- 拒绝候选不能改变 accepted raw/tree/coefficient/best state；失败记录带组件和字段。
  RNG 按稳定 run ID、seed、round、用途分配，不能依赖 Python `hash()` 或调度位置。
- E1 容差、E3 质量、E4 成本、E5 作者实验和 E6 安装要求沿用 eval。所有 real task
  记录 train/validation/test，模型选择只看 validation；缺项/失败保留。

### 每项的实现、设备与证据责任

以下 CPU 全部 required；CUDA 是主计划已声明的具体子集，不能静默 fallback。
参考文件、real case 和证据目录由未来 runner 以这些 ID 索引。

| 用例 | Recipe | 核心变化 / 独立参考组 | CPU 交付 | Required CUDA | 验收 |
|---|---|---|---|---|---|
| A1 | R1 | scalar regression / `scalar`、`tree` | F1.3/F1.6 | numeric + missing | E0/E1/E2/E3/E4/E6 |
| A2 | R1 | binary link + category / `classification`、`data` | F1.5/F1.6 | numeric encoding；类别原生 CUDA optional | E0/E1/E3/E4/E6 |
| A3 | R1 | multiclass K axis / `classification` | F1.6 | numeric softmax | E0/E1/E3/E4/E6 |
| A4 | R3 | query/pair + lambda / `ranking` | F1.5/F1.6 | 无；CPU required | E0/E1/E2/E3/E6 |
| A5 | R2 | routed residual quantile / `quantile` | F1.5/F1.6 | 无；CPU required | E0/E1/E2/E3/E6 |
| A6 | R8 | shared topology/vector leaf / `vector` | F1.6 | numeric squared error | E0/E1/E2/E3/E4/E6 |
| A7 | R4 | count + offset / `positive` | F1.5/F1.6 | numeric Poisson offset | E0/E1/E3/E4/E6 |
| A8 | R4 | positive mean / `positive` | F1.5/F1.6 | 无；CPU required | E0/E1/E3/E6 |
| A9 | R4 | Tweedie + composed prediction / `positive` | F1.5/F1.6 | 组合中的 Poisson 按 A7；全流程 CUDA optional | E0/E1/E3/E6 |
| A10 | R5 | censored likelihood / `aft` | F1.5/F1.6 | numeric event/right-censor | E0/E1/E3/E4/E6 |
| A11 | R6 | geometry + acceptance / `normal`、`state` | F1.3/F1.6 | numeric Normal 两种步长 | E0/E1/E2/E3/E4/E6 |
| A12 | R7 | formula + coupled direction / `formula` | F1.4/F1.6 | 无；混合执行单列 | E0/E1/E2/E3/E6 |
| A13 | R9 | independent runs + shared data / `runs` | F1.4/F1.6 | 兼容 R1/R4/R8 的一个批量组 | E0/E1/E2/E3/E4/E6 |

R1–R9 各 recipe 的全部 required 行都要闭合。E7 外部采用单列，不把内部作者成功
当作外部采用；也不等待外联才开展数学和真实任务验证。

## 2. A1–A13 任务卡

### A1 — 连续量回归

- **输入/输出：** X、实数 y、weight；raw 与预测均为实数。半平方误差
  `L=(F-y)^2/2, g=F-y, h=1`；base 为训练 weighted mean。
- **数据：** California Housing，沿用 [已冻结定义](../benchmarks/foundation/housing.json)
  的原始 archive/array hash 和目标单位。seeds 0–2 有历史 split；F0.3 以同一规则生成
  3–4 的 hash，禁止把三个旧 split 当作五个。随机切分不代表地理外推。
- **算法/组件：** scalar stats、三种 growth、Newton leaf、predict/add；基准不采样，
  单独启用行/列采样验证 seed。三种 policy 都可通过普通 Python 组合调用。
- **判错：** 手算 weighted base、两轮 G/H、最优 split 与叶值；缺失 routing、零权重、
  tie 和 lambda 缩放。常数目标不能制造 NaN。主质量 RMSE；shape 对而下一轮梯度旧则失败。
- **对照：** XGBoost `reg:squarederror`，LightGBM `regression`，CatBoost `RMSE`；
  使用各自合理生长/叶预算，报告完整预处理和推理。

### A2 — 二分类与类别输入

- **输入/输出：** 两类原始标签、X、weight；保存标签顺序。`p=sigmoid(F)`，
  `L=logaddexp(0,F)-yF, g=p-y, h=p(1-p)`；raw/probability/label 是三个不同输出。
  训练只出现一类时明确拒绝；初始化 logit 的极端概率处理写入配置。
- **数据：** UCI Adult；保留官方 test，官方 train 内按 seeds 0–4 做80/20 stratified
  train/validation；`?` 为缺失，目标去掉官方 test 的尾点。`fnlwgt` 从特征排除，
  不自动视为训练权重。真实主比较权重1；非单位/class weight 在独立 fixture 验证。
- **算法/组件：** CPU native category path 与 numeric encoding path 均有接口；GPU
  用训练集拟合的 numeric 编码，记录维数/unknown 行为和转换成本。对手可使用原生类别。
- **判错：** 标签重新编号后概率含义与模型映射一致；极端 logits 的 loss 有限，
  unseen/missing 不借测试标签编码；weighted loss 与叶统计各只加权一次。主指标 log-loss，
  AUC/分组混淆矩阵辅助；sample/class weight 不等价于已校准概率的保证。
- **对照：** XGBoost `binary:logistic`，LightGBM `binary`，CatBoost `Logloss`。
  缺失/类别处理与 effective class weights 写入每个 artifact。

### A3 — 多分类

- **输入/输出：** class mapping 与 raw `[N,K]`；稳定 log-softmax，输出概率和原标签。
  `g=p-onehot(y)`；数学 Hessian 为 `diag(p)-pp^T`，建树先用声明的对角上界
  `h_k=2*p_k*(1-p_k)`，不能把它称为精确 Hessian。权重作用一次。
- **数据：** UCI Covertype；保留全部原始 numeric/indicator 列，标签映射为0..K-1；
  seeds 0–4 的60/20/20 stratified split，one-hot soil/wilderness 列不私自压回整数。
- **算法/组件：** 每轮由同一 raw 快照计算全部 K 方向，先独立树/联合提交；K 是
  输出参数轴，不是 M 个 run。完整 softmax 更新进入下一轮。
- **判错：** K=3 手算 softmax、梯度行和为0、概率行和为1；精确矩阵与对角上界
  分开核对；交换 class mapping 后可对应还原。缺类别/非法标签不能静默截断。
- **对照/质量：** XGBoost `multi:softprob`、LightGBM `multiclass`、CatBoost `MultiClass`；
  multi-logloss 主指标，各类指标单列。Py-Boost 纳入 GPU/输出轴比较。

### A4 — Group ranking

- **输入/输出：** X、ordinal relevance、query boundaries 和可选 query/pair weights；
  输出 query 内排序 score。v1 不定义通用 row-weight→pair-weight 转换，传入则拒绝，
  或由调用者显式构造 pair weights，不能忽略。
- **数据：** Microsoft MSLR-WEB10K 官方五折，使用给定 train/vali/test；解析为 dense
  时缺省 feature token 为0，不当作 missing。qid 不进特征，全部真实 query 保留。
- **算法：** 独立参考枚举同 query 中 `rel_i>rel_j` 的 pairs；
  `L_ij=logaddexp(0,-(s_i-s_j))`，pair 对两个行的 g 相反；对角曲率为
  `sigmoid(d)*sigmoid(-d)`。Lambda 变体乘冻结当前排序所得 `abs(delta NDCG@10)`；
  此权重不反向求导，不能宣称梯度是 NDCG 的普通导数。
  初始scores=0；默认各query取pair loss的均值再按query weight加和，无pair的query
  贡献零梯度。采样时使用声明过的估计/归一化规则，不能改采样数却隐式改正则强度。
- **规模规则：** 生产可用有 seed 的每 query 有界 pair sampling；小 fixture 使用全 pairs，
  采样数/归一化属于搜索配置并入成本。gain=`2^rel-1`、discount=`1/log2(rank+1)`；
  ties 按稳定 row ID，IDCG=0 的 query 规定 NDCG=1 且单列数量，不静默删除。
- **判错/对照：** 不跨 query；给整个 query 的 score 加常数结果不变；pair梯度总和0。
  两轮 pair/lambda 按新 scores 重建。与 XGBoost `rank:pairwise`/`rank:ndcg`、
  LightGBM `lambdarank`/`rank_xendcg`、CatBoost `PairLogit`/`YetiRank` 比较 NDCG@10。
  对手按其语义使用 group/pair weights，不把优化方法不同视为数值 parity 失败。

### A5 — 加权分位数回归

- **输入/输出：** 实数 y、weight、`q in {0.1,0.5,0.9}`；每个 q 输出一个条件分位数。
  `r=y-F`，pinball=`max(q*r,(q-1)*r)`。分裂可用 pseudo g=`1[y<F]-q`、pseudo h=1；
  叶求解直接访问 routed residual/weight，不能继续用 Newton 均值代替 quantile。
- **数据：** UCI Bike Sharing `hour.csv`；使用预测时已知的 calendar 列，排除 `instant`、
  `casual`、`registered` 与实测 weather/temperature/humidity/windspeed；`dteday`
  用于切分、不进模型。按完整日期定义五个 rolling origins：训练前50/55/60/65/70%，
  紧接10%日期 validation、再10% test；F0.3 固定实际日期边界和 hash。
  Sprint012已冻结[原始文件/数组/五窗口hash](../benchmarks/v1/datasets/bike.json)，
  端点为floor(D*p/100)；仅数据准备，训练预算、基线与质量验收仍未完成。
- **叶契约：** weighted quantile 取最小满足累计正权重 `>=q*sum(w)` 的 residual；
  零权重先过滤，ties 采用该左端约定。raw 加 eta*leaf 后再算下一轮 residual。
  初始化为训练y的同约定weighted quantile，不能使用全数据quantile。
- **判错：** residual `[0,2,10]`、weights `[1,3,1]`、q=.5 的叶为2；改变权重使最优值
  可变化；非光滑点验证 subgradient 最优区间，不用有限差分检验二阶导。
  每个 q 的 pinball 都报告，crossing rate 单列；独立 q 模型不保证不交叉。
- **对照：** XGBoost `reg:quantileerror`（当前为平滑目标）、LightGBM `quantile`、
  CatBoost `Quantile`/`MultiQuantile`。相同 q/权重/预测空间下比较任务质量，不强求
  新版平滑算法与离散叶参考产生相同树。

### A6 — 多输出及向量叶

- **输入/输出：** X、Y `[N,K]`、row weight，K=2 起步；raw/predict `[N,K]`。
  逐目标训练集均值/标准差预处理，保存逆变换；叶输出 schema 与树 topology 分离。
  标准化后base=0；常数目标的标准差按1处理并保留常数标记。
- **数据：** UCI Parkinsons Telemonitoring 的 motor/total UPDRS 两个目标；两者都
  从特征中删除，subject ID 只用于 group split；subject 级60/20/20，seeds 0–4。
  原标签的评分/插值定义保留，任务不作临床有效性声明。
- **算法：** 两条 required recipe：独立树、共享 topology/vector leaf。后者用各输出
  gain 之和评分，叶逐维 `-G_k/(H_k+lambda)`；另以可替换统计投影选择 topology，
  叶仍用完整 K 维统计，检验分裂统计与叶统计可不同。
- **判错：** 手算两维 split 与 leaf，K=1 退化到 A1；输出排列可还原，投影不能
  丢掉叶的目标维。两轮比较全部目标；不得只通过一个标量通道或伪造共享 topology。
- **对照/质量：** XGBoost 独立树/experimental `multi_output_tree`、CatBoost `MultiRMSE`、
  LightGBM 独立目标循环、Py-Boost 原生多输出。报告每个目标原单位 RMSE和标准化平均；
  E3 每个目标都检查，不能平均掩盖某一维失败；两种 OpenBoost 结构分别保留结果。

### A7 — 计数和 exposure

- **输入/输出：** 非负整数 count、`e>0`、独立 sample weight；
  `mu=e*exp(F)`，输出 count mean 或单位 exposure rate，预测时必须声明所需 e。
  `L=mu-y*(F+log(e))+lgamma(y+1), g=mu-y, h=mu`。
- **数据：** freMTPL2freq / OpenML 41214，IDpol 只用于实体切分，ClaimNb 为目标；
  不复制示例对 ClaimNb/Exposure 的裁剪。非法 e/weight/target 记录并拒绝或在
  数据契约中预声明排除，不能训练时偷偷改值。entity split seeds0–4。
- **算法/组件：** rate intercept=`log(sum(w*y)/sum(w*e))`；全零计数按显式最小
  rate 策略处理。offset 是逐行 raw 加量，不是 weight，训练/验证/推理一致。
- **判错：** 固定 F 后 e 翻倍，count 翻倍而 rate 不变；同一个样本重复 weight 次
  与整数加权一致；零 sample weight 不贡献统计；模型保存必须保留 offset约定。
- **对照/质量：** Poisson GLM；XGBoost `count:poisson` + `base_margin=base+log(e)`，
  LightGBM `poisson` + 显式 init-score/prediction adapter，CatBoost `Poisson` + baseline
  adapter。adapter 的 base/offset 保存和推理须先 smoke，不能把 init_score 当作
  自动随模型保存的常量。count Poisson deviance 主指标，rate/总量偏差辅助。

### A8 — 正值金额和严重度

- **输入/输出：** y>0、weight、`mu=exp(F)>0`；固定 shape=1 的 Gamma 均值 recipe，
  训练用 `L=y/mu+log(mu), g=1-y/mu, h=y/mu`。不声称提供估计过的完整分布。
- **数据：** freMTPL2sev / OpenML41215 与 frequency 的 IDpol 关联以获得 covariates；
  正赔付 claim 级训练，weight=1，同保单不跨 split。非正赔款/孤立ID单列排除；
  不把 claim 级样本误当保单平均。分割ID与A7/A9一致。
- **算法/组件：** base=log(weighted mean y)，更新 log mean；float64 参考验证 y/μ
  的极端范围，生产溢出不得以无记录截断掩盖；一行异常能定位到组件/输入。
- **判错/质量：** 导数与独立公式一致；非法零/负 y 拒绝。两轮 weighted leaf
  与预测保存正确；金额单位变化需记录，主指标 Gamma deviance。
- **对照：** Gamma GLM、XGBoost `reg:gamma`、LightGBM `gamma`；CatBoost 公共回归
  目标表未列 Gamma，若做 custom objective 则单列扩展对照，不冒充内置目标。

### A9 — 总损失和纯保费

- **输入/输出：** 保单观察期内正赔付总额 c、exposure e，annualized y=c/e；
  主路径训练 weight=e，`mu=exp(F)` 是 annualized mean；预测观察期金额=e*mu。
  这条路径不再加 log(e) offset。额外业务权重若有须明确乘积并只施加一次。
- **数据：** frequency 左关联 severity，按 IDpol 聚合正赔款；ClaimNb=0 且无赔款
  才填0。ClaimNb>0 却无赔款记录、以及零计数却有赔款的矛盾记录单列排除，不猜
  “缺失就是零”。定义为正赔付损失，不是含退款的净损失；F0.3记录行数与关联质量。
- **算法：** Tweedie variance power p固定于每次fit（默认1.5；调参范围F0.3冻结）。
  `L=-y*mu^(1-p)/(1-p)+mu^(2-p)/(2-p)`，
  `g=mu^(2-p)-y*mu^(1-p)`，`h=(2-p)*mu^(2-p)+(p-1)*y*mu^(1-p)`。
  另交付 frequency×severity：Poisson 用**正赔付记录数**，Gamma用同一定义的
  正赔付均值，不能把含零赔付的 ClaimNb 与正赔付 severity 直接相乘当作同一目标。
- **判错：** 零损失合法；e只进入weight和预测单位转换，不重复当offset。
  手算小型 join/aggregate 和两阶段乘积；改变实体排序不改变配对；保存两阶段
  模型及依赖后预测不变。Tweedie 和组合模型都有真实结果。
- **对照/质量：** Tweedie GLM、XGBoost `reg:tweedie`、LightGBM `tweedie`、CatBoost
  `Tweedie`，以及相同 paid-count 定义的两阶段基线；exposure-weighted annualized
  Tweedie deviance 主指标，aggregate totals辅助；不凭均值宣称尾部分布校准。

### A10 — Censored AFT

- **输入/输出：** 规范 interval target `(lower,upper)`；初始实现为完整事件和右删失。
  选择 log-normal AFT，`log T=F+sigma*Z, Z~Normal(0,1)`；sigma默认1，每次fit固定。
  raw=log-time location，`exp(F)`是median，mean=`exp(F+sigma^2/2)`，另有survival/quantile。
- **数据：** scikit-survival Veterans' Administration lung cancer，按官方
  `Status`/`Survival_in_days` 转换；event和右删失stratified split，seeds0–4。
  小型真实任务不支持规模优势声明。IPCW训练删失估计、时间网格与支持区间由F0.3固定。
- **数学：** `z=(log(t)-F)/sigma`；event NLL=`log(t*sigma)+z^2/2+log(2*pi)/2`，
  g=`-z/sigma`、h=`1/sigma^2`。右删失 NLL=`-log(S_Normal(z))`，
  g=`-mills(z)/sigma`，h=`mills(z)*(mills(z)-z)/sigma^2`；用稳定 log-tail参考。
- **判错：** 改event为censored须改变loss/更新；合法inf上界保留，非法NaN/反向区间拒绝；
  left/interval/truncation依已声明范围显式拒绝。两轮 likelihood、输出单位、单调生存函数
  与round trip一致，不能把右删失时刻当死亡时刻算RMSE。
- **对照/质量：** XGBoost `survival:aft` Normal 与相同sigma、CatBoost `SurvivalAft`
  `dist=Normal;scale=sigma`（CPU，+inf→-1适配）、参数log-normal AFT。
  统一时间密度Jacobian/likelihood常数后censored NLL主指标，IPCW Brier与C-index辅助。
  LightGBM没有已核对的同名内置AFT；custom loss路径可做对照，状态单列。

### A11 — Distributional / NaturalBoost

- **输入/输出：** 实数 y、weight；raw=(mu,log_sigma)，输出两个参数及Normal分布。
  `L=log_sigma+(y-mu)^2/(2*sigma^2)+log(2*pi)/2`。
  普通gradient为 `((mu-y)/sigma^2, 1-(y-mu)^2/sigma^2)`；Fisher为
  `diag(1/sigma^2,2)`，natural direction为 `Fisher^-1*g`，不是随意Hessian命名。
- **数据：** California Housing；数据和split与A1复用，仍单独产出分布任务结果。
  原始目标单位固定；不能用A1的RMSE代替NLL/CRPS。
- **算法：** 拟合 negative ordinary/natural direction 的独立参数树；方向回归weight
  与训练weight明确对应。固定步长和有限回溯都required；联合提交的两参数由同一快照
  计算；D4另检验有序参数更新。初始化训练weighted mean/scale，scale下限预声明。
- **判错：** 手算Fisher solve、两轮梯度及accepted状态，拒绝无残留；非单位weight
  不得自然梯度里乘一次、树回归又乘一次。NLL与CRPS由独立 evaluator 计算。
- **对照/质量：** NGBoost Normal + LogScore/natural；CatBoost `RMSEWithUncertainty`；
  同数据的全局Normal及允许外层循环的树基线。CatBoost预测方差/scale及raw转换要
  smoke后才比较。NLL主指标，CRPS、coverage+width、PIT辅助；不是只提高coverage。

### A12 — FormulaBoost 与真实结构任务

- **输入/输出：** 树只读配方 Z；结构输入 `x=age_days/28>0`；目标为MPa强度。
  `a=softplus(u), b=softplus(v), f=a*(1-exp(-b*x))`；输出f以及参数a/b。
  此饱和单调公式是本次**待检验假设**，不是数据源证明的物理定律。
- **数据：** UCI Concrete Compressive Strength，七个材料用量列作为Z、Age作为x。
  相同七列配方的记录按group整体分割，seeds0–4的60/20/20；Age、目标不进树特征。
  对完全相同输入的重复记录保持同group。实际group数/各龄期支持由F0.3验证。
- **数学：** 用 `-expm1(-b*x)` 稳定求值；analytic raw Jacobian
  `J_u=sigmoid(u)*(1-exp(-b*x))`，`J_v=a*x*exp(-b*x)*sigmoid(v)`。
  半平方误差的 g=`J*(f-y)`；GGN=`J^T J`，方向solve加入显式damping。
  一行的GGN秩最多1；full矩阵不自动证明两个参数可识别。
  默认初始化 `a0=max(weighted_mean(y_train),1e-6), b0=1`，用稳定softplus逆映射
  得raw；独立fixture可显式指定raw0。初始化不能读取validation/test。
- **算法/判错：** ordinary/diagonal/full方向用独立参考；双参数至少两轮且line search
  正确提交。合成实验必须有重复Z/不同x及已知a/b，另有单一x不可识别反例、公式错设反例。
  真实数据只评价预测和结构约束，不声称恢复“真实参数”；没有真实结果A12不完成。
- **对照/质量：** 全局相同公式非线性拟合、旧Formula固定revision、使用XGBoost/
  LightGBM/Py-Boost树弱学习器的外层耦合更新、以及可读Z+x的三大库普通回归。
  不能以XGBoost只接收对角h就断言它无法在外层算full方向。RMSE主指标；结构内/外
  支持区间误差和参数稳定性辅助，不从test重挑公式。任意推理formula依赖明确打包。

### A13 — Train-many 与模型选择

- **输入/输出：** prepared-data identity、稳定run IDs、各自recipe/config/seed/round预算、
  validation metric；返回逐run状态/模型/成本与按validation选定的模型，不只一张均值表。
- **数据：** Covertype 作为主要真实集合，Housing作为另一来源；dataset/split沿用A3/A1。
  同一split/分箱设置可共享；跨fold、权重改变分箱、不同mapping时不得错误复用。
- **算法：** M=1/8/32，顺序参考与兼容组批量执行；run独立early stop、best iteration、
  seed和失败。不同K/目标在CPU组合检查；GPU先选R1兼容组，不强制异构run融合。
  每个候选都算完整model-selection成本；32个配置属于E4集合实验，不能冒充E3的16-trial预算。
- **判错：** 独立训练 vs 顺序复用 vs 批量逐run对照；交换顺序、重分组后结果按ID一致。
  注入一个失败run其余继续，最终汇总保留失败；不能把失败run忽略后宣布整个required集合通过。
  选模型只读validation，提交/保存后的best state必须一致。
- **对照/质量：** 三大库常规循环，分别记录其可用的prepared-data复用；Py-Boost循环
  纳入GPU候选。不能把对手反复分箱却让自己缓存作为唯一速度胜利。所选模型通过
  原任务E3，全部集合通过E4；旧ConfigBatch顺序实现只作语义参考。

## 3. 竞争方案审计：扩展入口、设备与缺口

三大库版本按 [release核对](boosting-release-review-2026-09-05.md)：XGBoost3.4.1、
CatBoost1.2.10、LightGBM4.7.0。这里是**文档/代码审阅**，所有新版本 runtime smoke
状态均为 `not_run`，不据此给CPU/CUDA打pass。NGBoost、Py-Boost、GLM/AFT依赖的
精确版本由F0.3固定；Py-Boost没有取得有效latest-release页，不编造tag。

| 路径 | 已核对的入口 | 任务与成本边界 |
|---|---|---|
| XGBoost | built-in/custom objective、grow policy、vector tree、外层逐轮/源代码修改 | A1–A10多种现成任务；3.4 vector hist仍experimental。自定义Hessian限制不阻止外层预条件方向；C++改动的构建/排错成本须计入 |
| LightGBM | objectives、`Booster.update(fobj)`、`rollback_one_iter`、`set_leaf_output` | D1/D3/D4可先组合公共接口；leaf替换必须发生在下一轮梯度前。对逐候选新增统计的改动仍需核对source路径，不能只调一个总min_child_weight |
| CatBoost | native categorical、symmetric/depthwise/lossguide、custom loss、pair/group目标 | A2/A4/A6/A11是重要强对照；MultiRMSE有GPU而MultiRMSEWithMissingValues无。SurvivalAft CPU；GPU custom loss存在不代表所有自定义算法都走GPU |
| NGBoost | distribution/score/metric、natural gradient与训练循环 | A11/D4必须纳入，不能将已有几何/line-search能力算作新发明；修改训练策略的具体成本实测 |
| Py-Boost | Python/CuPy、callback、loss/metric、sampling、multioutput sketch | 直接foundation对照，训练需GPU；适用题必须作为候选。选中则在GPU环境执行该arm，OpenBoost可先CPU；按E5预算记录实际设备，CPU不支持不能记为失败 |
| GLM/参数AFT/全局formula | 同目标的简单统计模型 | 检验新增框架是否解决真实任务，保持单位/link/数据输入一致；不能只比较弱默认树配置 |

F0.3 capability smoke逐路径至少验证非单位weight、prediction space、base/offset、
保存/加载、最终metric及reported backend。baseline adapter失败标记error并修复；
不能悄悄将该强对照改成unsupported。GPU安装失败单列环境事实，不得伪装成算法不支持。
GPU作者arm与CPU作者arm的wall/token预算相同，设备成本另列；各arm工具和源码可达性相当。

进一步读到 Py-Boost 的 `DepthwiseTreeBuilder.build_tree`：它已对建树G/H使用
`multioutput_sketch`，随后以原始grad/hess调用`calc_node_values`求叶。分裂统计与叶
统计分开本身也不是独有能力。D2的具体source审计入口是`depthwise_grow_tree`中的
候选选择；D3可以利用返回的leaf indices，但替换叶后必须同步训练/验证prediction
cache，不能只改导出的模型。以上是调用路径证据，不是修改工时或runtime成功证明。
[官方tree实现](https://raw.githubusercontent.com/sb-ai-lab/Py-Boost/master/py_boost/gpu/tree.py)。

公开plans维持已发布/roadmap/RFC/request区分：XGBoost多输出tracker、LightGBM预测与
类别mapping工作、CatBoost模型导出任务用于设计和风险审阅，不拿未发布承诺当运行基线。

### 已声明的范围边界

| 非当前required能力 | 现成方案审阅 | v1处理理由与行为 |
|---|---|---|
| native CSR/CSC、外存、分布式/多GPU | 三大库已有不同稀疏/扩展路径；LightGBM4.7新增GPU能力见release | 本轮先验证单设备组件语义；数据可在预算内显式dense转换并计成本，原生参数不支持则拒绝 |
| 全部categorical CTR/ordered boosting组合 | CatBoost已有成熟类别处理 | v1先一个明确可测方法；保留真实类别任务质量gate，不能因实现小而删任务 |
| 完整Cox/competing risks/truncation、所有AFT删失族 | XGBoost/CatBoost已有Cox与多种AFT标签 | 当前R5范围事件/右删失；schema保留区别，未实现类型拒绝；接口不能写死逐行独立目标 |
| 全部分布族、DART/GOSS、linear leaf与全部constraints | 上游目标/参数和本仓库历史能力可供后续比较 | 不用功能目录替代组合语义；保留typed payload与统计入口，准确拒绝未支持参数 |
| 任意Python自动编译/自动序列化 | 不能由竞争库的Python接口推出此保证 | bulk ops显式设备边界；formula依赖显式声明，未支持执行明确报错 |

这些是既有R/C边界的落实，不删除A1–A13；新增需求可修订scope，不允许在评测失败后
静默把required降级为optional。

## 4. D1–D5：作者修改任务

每题通过安装后的public API交付扩展包/recipe，不改core/private module；对手可用其
公共入口、外层循环或源码。执行要求沿E5，不只看是否调用过自定义函数。

| ID / 变化轴 | 明确改动与独立判错 | 最强候选替代路径与可比较成本 |
|---|---|---|
| D1 objective / control | 加入tau=.8 expectile：残差r=y-F，loss=`abs(tau-I[r<0])*r^2`；解析g/h、weighted base、两轮更新与raw round trip | XGBoost/ CatBoost已列expectile，LightGBM custom loss，Py-Boost loss。现成内置也允许，OpenBoost不预设获胜 |
| D2 split + statistics | 每个child的每个预声明cohort信息质量>=1；额外per-cohort sum与训练weight分开，普通gain内选最优可行split；无可行候选则不分裂 | 从Py-Boost build_tree→depthwise_grow_tree改候选统计/选择；也允许XGBoost/LightGBM source。仅总min_child_weight不能替代此约束；允许任何数学等价实现 |
| D3 leaf solver | 叶上最小化weighted pinball + `lambda*(v-anchor)^2/2`，lambda>0；公开row view读取residual/weight，替换solver后下一轮预测使用新叶 | LightGBM路由+`set_leaf_output`，Py-Boost callback/source；比较真正所需工作，不把现成leaf hook藏起来 |
| D4 update / acceptance | A11的joint更新改成预声明的有序参数更新；每步最多6次alpha=`0.1*0.5^j`，有限且loss下降才提交；全部拒绝时无状态残留，下一参数读取已接受状态 | NGBoost训练循环、多个LightGBM/XGBoost weak learner的外层循环、Py-Boost callbacks；已有line search必须计入baseline能力 |
| D5 run scheduling | 共享prepared data的异构K=1/K=2、独立预算/early stop/RNG，重排不改变逐run结果；一个run报错不污染其余，汇总报告失败 | 常规独立模型循环+安全数据复用；Py-Boost GPU循环。调度本身不预设需要fusion；记录准备/执行/恢复成本 |

最小独立反例：

- D1：r正/负/零及weight0；非光滑点不用普通二阶有限差分，tau=.5回到对称平方损失。
- D2：固定bins的6行，cohort交替A/B，g=`[-6,1,1,1,1,2]`、h=1；最优总gain
  分裂可违反cohort约束。逐候选枚举查最优合法项，再用全部A在左、B在右的数据查
  “无可行split”行为。cohort不是树feature，不能通过泄漏它绕过约束。
- D3：枚举residual断点及断点间驻点验证唯一最优；检查subgradient包含0；
  lambda变大时靠近anchor。weighted quantile默认解不能冒充带二次惩罚的解。
- D4：有效下降候选、方向反号导致全部拒绝、NaN候选、第二参数只在第一参数成功后
  读取新状态；检查tree/coefficient/raw/best/RNG的声明语义，而不只看最终loss。
- D5：run顺序交换、失败重试、不同stop轮数、同seed不同run IDs、改变data identity。
  “结果shape一样”或“失败run被跳过”不算通过。

H1/H2保留题的内容不在接口设计材料展开。F0.3由评测侧冻结独立任务卡/verifier与hash，
F1接口作者只看到开发集D1–D5；不能以未填内容的两个ID宣布E5已ready。若后来用保留题
改接口，按E5转开发集并补新的保留题，完整记录。

## 5. 最小接口草图与C1–C7追踪

名称仍是草图；选择这些边界是因为上面的实际调用方需要它们。

```python
prepared = prepare(training_rows, feature_schema, binning, device=device)
problem = bind(prepared, target=target, weight=weight, offset=offset,
               query=query, structure_input=structure_input)
with runtime.run(run_id=run_id, seed=seed, device=device) as run:
    accepted = initialize(problem)
    for step in range(rounds):
        geometry = objective.geometry(problem, accepted)
        direction = direction_rule(geometry)
        # grow也是普通Python；可以直接改写下面的组合或替换某一个函数。
        tree = grow(prepared, direction, aggregate=aggregate,
                    candidates=candidates, score=score, feasible=feasible,
                    partition=partition, leaf_solver=leaf_solver,
                    policy=growth_policy, run=run)
        proposal = propose(accepted, tree, output_mapping=output_mapping)
        accepted = accept_or_reject(problem, accepted, proposal, run)
    model = export_model(accepted, output_schema=output_schema)
```

`grow`的可读实现必须公开 aggregate→candidate statistics→feasibility/score→choose→
partition→leaf solve。候选结构包含阈值/类别集合与missing route；leaf solver可获得
只读row indices/residuals及统计，不能固定成仅G/H。不同algorithm可自行写循环，
不要求穿过唯一万能trainer，也不要求为每个函数做registry。

| 能力 | 被哪些任务实际要求 | 独立交付与验收入口 |
|---|---|---|
| C1 typed data/target | A2类别、A4query、A6向量、A7offset、A10interval、A12结构输入 | F0.2 `data`；F1.1/F1.5；所有A卡数据身份/shape/非法输入 |
| C2 composable tree core | A1三policy、A4逐行归约、D2额外stats | F0.2 `tree`；F1.2；穷举split、route、ties与可行性 |
| C3 learner/leaf/output | A5/D3 residual solver、A6向量叶、A11/A12参数映射 | F0.2 `quantile`/`vector`；F1.2/F1.6；不把K个标量run称为vector leaf |
| C4 explicit state/runtime | A11/D4拒绝、有序更新；A13/D5独立runs | F0.2 `state`/`runs`；F1.1；F3设备/transfer/同步可见 |
| C5 artifacts | A2映射、A7offset、A9双模型、A10预测空间、A12formula | F1 round trips、F5 clean wheel；标准raw树推理不依赖训练插件 |
| C6 evaluation/authoring | 全部A与D；H1/H2单独冻结 | F0.3 runner/judge；F2 E5；F4 E3；坏artifact使gate失败 |
| C7 usable workflow | 每个A的安装→baseline→修改→验证→保存/推理 | F2试用、F5 E6；缺少一个A的工作流就保持未完成 |

公开raw state统一按`[N,K]`描述；backend可以有自己的存储布局并记录转换成本。
树用可变节点topology加scalar/vector payload；固定511槽位/深度8不是新契约。
候选buffer与accepted state明确所有权；可实现transaction/delta而不每次全量copy。

## 6. 审阅现有实现后的约束

本节是退役前代码审计。Sprint 002 已按用户要求删除旧生产实现，源码链接固定到
`50acfc6`；这些历史能力不代表当前 v1 namespace 已实现对应功能。

- [标准模型](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_models/_boosting.py) 的CPU训练会乘sample_weight，
  当前standard CUDA入口拒绝非空weight；multiclass fit没有同一weight入口，按K分别建树。
  [多分类文档](../docs/user-guide/models/multiclass.md) 只代表旧API，不证明v1加权/mapping。
- [FormulaObjective](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_objectives.py) 已在逐参数拟合前计算耦合GGN方向，
  [模型](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_models/_formula.py) 接收独立model_input；
  [现有测试](../tests/test_formula.py) 主要是合成机制。v1不能把已有耦合方向说成新能力，
  也不能把full GGN说成自动解决参数识别。真实A12数据现在有明确任务。
- [ConfigBatch](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_batch.py) 与 [batch测试](../tests/test_batch.py) 已有
  多轮逐config重算loss的语义；独立stop/error/RNG和兼容批处理需新证据。
- [当前survival](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/_models/_survival.py) 与
  [测试](../tests/test_survival.py) 是Weibull的event/time入口；A10固定log-normal
  是独立新recipe，不能把旧类名当作新删失/噪声语义已通过。
- [实验Booster](https://github.com/jxucoder/openboost/blob/50acfc6/src/openboost/experimental/_booster.py) 的CUDA入口仍限制eval/callback/
  early stopping。新方案按用例逐项验收，不搬迁限制或要求兼容；旧数学失败样例保留。

## 7. 来源与F0.1完成边界

任务数学是本规格的显式定义，F0.2仍须独立推导/验证。以下资料用于核对现成接口、
设备和数据语义；公开文档不能替代F0.3安装后的capability smoke。

- [XGBoost参数](https://xgboost.readthedocs.io/en/stable/parameter.html)、
  [advanced custom objective](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html)、
  [GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html)：任务入口/几何限制/设备边界。
- [LightGBM参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)、
  [Booster](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)：目标、逐轮更新与叶修改。
- CatBoost [回归](https://catboost.ai/docs/en/concepts/loss-functions-regression)、
  [ranking](https://catboost.ai/docs/en/concepts/loss-functions-ranking)、
  [多输出](https://catboost.ai/docs/en/concepts/loss-functions-multiregression)：目标/weight/device差异。
- [NGBoost开发](https://stanfordmlgroup.github.io/ngboost/5-dev.html)、
  [Py-Boost官方仓库](https://github.com/sb-ai-lab/Py-Boost)：几何/score与Python GPU扩展对照。
- A1–A9数据入口见 [应用矩阵](foundation-application-contracts.md)；
  [保险数据处理参考](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)。
- A10 [官方加载器与字段](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.datasets.load_veterans_lung_cancer.html)、
  [评价语义](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)。
- A12 [UCI Concrete，数据/单位/CC BY4.0](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)。
  文件校验/配方group数量仍须实际下载后核实，公式并非来自该页面。

**F0.1可验收项：** A1–A13逐项输入/输出、数据/切分、算法、独立判错和对照齐全；
R1–R9/C1–C7覆盖无缺口；D1–D5内容可判对错；设备状态与接口草图明确。
**参考已交付：** F0.2 oracle与反例，见[出口审计](../v1-sprints/f0-2-acceptance-ledger.md)。
**仍未完成：** F0.3下载/hash/运行capability/预算/保留题/判卷器；
因此本文件不代表整个F0或任何E-gate通过。下一阶段冻结比较协议，
不用production objective作为唯一oracle，不夹带新trainer或kernel。
