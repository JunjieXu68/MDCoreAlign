"""Write the Chinese residue-level interpretation from measured new tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.atlas_strict_single_thread_cpu_v1.common import write_json, write_tsv


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def table(rows, columns):
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered = []
        for key, _ in columns:
            value = row.get(key, "")
            if value in ("", None):
                value = "NA"
            elif isinstance(value, float):
                value = f"{value:.6g}"
            elif isinstance(value, str):
                try:
                    if "." in value or "e-" in value:
                        value = f"{float(value):.6g}"
                except ValueError:
                    pass
            rendered.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def generate(run):
    config = json.loads((run / "analysis_config.json").read_text())
    overlap = read(run / "core_overlap_summary.tsv")
    members = read(run / "residue_membership.tsv")
    differences = read(run / "disagreement_residues.tsv")
    geometry = read(run / "geometry_by_residue.tsv")
    group_geometry = read(run / "geometry_group_summary.tsv")
    decomposition = read(run / "geometry_pair_decomposition.tsv")
    sampling = read(run / "sensitivity_summary.tsv")
    sample_res = read(run / "sensitivity_by_residue.tsv")
    diagnostics = read(run / "diagnostics.tsv")
    rmsf = read(run / "rmsf_impact.tsv")
    for row in differences + geometry + diagnostics:
        row["residue_label"] = row["resname"] + row["source_resid"] + row["icode"]
    points = {(r["context"], r["atom_index"]): r for r in geometry}
    sample_once = {(r["case"], r["atom_index"]): r for r in sample_res if r["seed"] == "20260915"}
    evidence = []
    for row in differences:
        s = sample_once[(row["case"], row["atom_index"])]
        g = points[(row["context"], row["atom_index"])]
        evidence.append(
            {
                **row,
                "h_common_A": g["h_common_A"],
                "h_framework_A_A": g.get("h_framework_A_A", ""),
                "h_framework_B_A": g.get("h_framework_B_A", ""),
                "sampling_selection_scope": "original training source, not repeated full ADK",
                **{
                    k: s[k]
                    for k in [
                        "inclusion_count_5",
                        "rank_min_5",
                        "rank_max_5",
                        "old_train_group",
                        "baseline_rank",
                    ]
                },
            }
        )
    write_tsv(run / "disagreement_evidence.tsv", evidence)
    sampling_changes = []
    for (_, _), r in sample_once.items():
        n = int(r["inclusion_count_5"])
        base = r["baseline_included"] == "True"
        if 0 < n < 5 or (base and n == 0) or (not base and n == 5):
            change = (
                "varies_among_five_draws" if 0 < n < 5 else "all_five_differ_from_original_baseline"
            )
            sampling_changes.append({**r, "sensitivity_class": change})
    write_tsv(run / "sensitivity_changed_residues.tsv", sampling_changes)
    text = """# 核心差异与有限敏感性：对导师 comment 的逐残基回答

**两方法都纳入了对方没有纳入的残基；这些差异并不全是无关紧要的截点成员。** ADK 原全轨迹的共同成员 124/128（Dice 96.875%）复核成功，两侧各 4 个独有残基；训练核心则是共同 125 个、两侧各 3 个，不能混为同一组“八个位点”。3nci_A 的 matched-k=541 核心共同 515 个，两侧各 26 个。MDCoreAlign 对所有这些 MLF_only 残基仍赋予非零软权重，“没有进入 top-k 报告集合”并不等于 dropped from fitting。

有些变化确实发生在报告截点附近，另一些是可复现的方法间分歧：例如 ADK 的 VAL117 与 HSE172，以及 3nci_A 的 SER315、LEU889、VAL250，不能都淡化为紧贴 k 的点。五次预定代表帧扰动改变了部分成员；固定集合、相同 Kabsch 拟合器下的后半段/R2 RMSF 差异相对较小，但实际原生方法输出仍有可观的局部差异。因此这两个案例支持“相近参考系解释”的有限说法，不能支持无损、完全相同、普适等价、生物学优越或统计非劣效。

本轮只新增 ADK 与 3nci_A 的分析；I-FABP 不在此次范围，旧结果全部保留，也不把这两个案例描述成所有检查过的蛋白均获得支持。没有重跑 MDLovoFit、AT5、随机集合 benchmark 或速度估计，没有改动生产权重公式或参数。

## 来源、集合与整数计数

唯一旧来源为 `results/biological_core_validation_v1/`，已依据当前 biological_validation_report.md、完整 TSV 和 run_manifest.json 的 hash 核对对应版本。本轮 run_id 为 `RUN_ID`。所有表均提供 case、selection_source、selection_frame_indices、evaluation_source、evaluation_frame_indices、core_definition、k、reference_source、run_id 或等效字段；长索引通过本轮 JSON 中的实际列表引用，不用含糊的帧范围替代。

- `ADK_full_descriptive`：全部 4,187 帧估计的原始两核心，全轨迹几何/RMSF只作描述，解释原124/128。
- `ADK_train_heldout`：只用原训练0–2092（含端点）确定两核心；在2093–4186评估。索引仍按旧 manifest 原样保存，没有重选窗口。
- `3nci_A_R1_to_R2`：R1 的10,001帧定义集合，R2 的10,001帧评估；旧full/train权重和两法 masks 逐元素相同，因此复用R1集合，未合并R1/R2。

保存的 MDLovoFit 每一帧原始布尔 masks 已直接复核，两个案例各来源均恒定并与保存的core、frequency一致；不是从平均 occupancy 猜固定。MDLovoFit 没有连续稳定性排名可供本轮提取。matched-k 由实际 frozen_weights 降序、原输入下标升序处理精确并列，逐项重建并与旧表一致。

""".replace("RUN_ID", config["run_id"])
    text += table(
        overlap,
        [
            ("context", "来源"),
            ("k", "k"),
            ("n_shared", "共同"),
            ("n_MCA_only", "MCA_only"),
            ("n_MLF_only", "MLF_only"),
            ("n_neither", "neither"),
            ("dice", "Dice"),
            ("jaccard", "Jaccard"),
            ("nonshared_fraction_each_core", "每侧非共享比例"),
        ],
    )
    text += """

ADK 原全轨迹两侧非共享比例各为4/128=3.125%；训练来源各为3/128=2.34375%。3nci_A 各为26/541=4.80592%，不是把52个不相同成员除以单侧541。

3nci_A 的原生 native60 是542而 matched-k是541。额外的第542名是 **THR209**：weight=0.464237817，低于第541名截点0.464568883，仅差0.000331067；属于 framework_B_only，而且属于 MDLovoFit 核心。它在matched比较中是MLF_only，在native542比较中变成shared。只增加这一位点时，交集由515变516，MCA_only仍26，MLF_only由26变25；不同大小的比较不能混入541对541的差异比例。详见 [native_rounding.tsv](native_rounding.tsv)。

## ADK：八个原始位点与六个训练差异位点

所有编号均保留原拓扑命名。HSE172 的HSE是原始残基名，未改成数组下标；输出链A对应来源segid ADK，原PSF无chainID这一事实仍记录在映射中。下表穷尽全轨迹8个位点，h是在全轨迹上的描述性值，五次纳入计数则来自**训练来源**，不得把它当成直接重复全轨迹分析。

"""
    text += table(
        [r for r in evidence if r["context"] == "ADK_full_descriptive"],
        [
            ("group", "类别"),
            ("residue_label", "残基"),
            ("literature_regions", "域"),
            ("frozen_weight", "原全权重"),
            ("rank", "原全排名"),
            ("weight_minus_cutoff", "w−截点"),
            ("h_common_A", "全轨迹 h(C) Å"),
            ("old_train_group", "训练类别"),
            ("baseline_rank", "训练排名"),
            ("inclusion_count_5", "训练五次纳入"),
        ],
    )
    text += """

全轨迹第128名PRO9与第129名THR31的权重间隔只有0.000199780，属于很细的报告边界。VAL117在第117名、比截点高0.101316；HSE172和MET174分别第140和136名，比截点低0.062009和0.046439，不能称为只差一名。ASP118/LEU168在124/125名，THR175在132名。精确权重并列不是这些截点差异的原因。

结构域边界与权重截点不同：THR31距NMP起点30为1个序列位点，ASP118距CORE端点121为3；PRO9、LEU168、HSE172/MET174/THR175并不位于这些域划分边界。117–118和174–175是连续差异片段，不能据此把附近整段判成错误或局部无序。完整连续片段及距注释边界距离见 [disagreement_segments.tsv](disagreement_segments.tsv) 和 [residue_membership.tsv](residue_membership.tsv)。

训练核心中PRO9变成neither、THR31变成shared，留下 MCA_only=VAL117/ASP118/LEU168，MLF_only=HSE172/MET174/THR175。原两种方法各自的核心也随全轨迹/训练来源而异；后半段几何必须使用这六个位点，而不是原始八个。

"""
    text += table(
        [r for r in evidence if r["context"] == "ADK_train_heldout"],
        [
            ("group", "类别"),
            ("residue_label", "残基"),
            ("frozen_weight", "训练权重"),
            ("rank", "训练排名"),
            ("h_common_A", "后半段 h(C) Å"),
            ("rank_min_5", "五次最小排名"),
            ("rank_max_5", "五次最大排名"),
            ("inclusion_count_5", "五次纳入"),
        ],
    )
    text += """

## 3nci_A：全部52个差异位点，保留两个框架

两框架的互斥分类是A_only、B_only、both、neither；全蛋白对应202、289、29、383个残基，总数903。下面两侧的独有成员均没有落在29个shared framework残基中。MCA_only=1个A_only、8个B_only、17个neither；MLF_only=3个A_only、13个B_only、10个neither。neither仅表示不在两段外部晶体框架注释内，不代表不稳定。不能把A/B重叠成员重复计数。

MCA_only集中片段包括47–48、78–79、164–168、191–192、194–196、306–307、889–890；MLF_only包括220–221、250–251、278–280、718–719、724–725，另有离散位点。两类差异并未集中于唯一一个框架，完整片段列表按固定编号保存，没有挑选有利片段。

"""
    text += table(
        [r for r in evidence if r["case"] == "3nci_A"],
        [
            ("group", "类别"),
            ("residue_label", "残基"),
            ("framework_category", "框架互斥类"),
            ("frozen_weight", "权重"),
            ("rank", "排名"),
            ("h_common_A", "R2 h(C) Å"),
            ("h_framework_A_A", "R2 h(A) Å"),
            ("h_framework_B_A", "R2 h(B) Å"),
            ("rank_min_5", "五次rank最小"),
            ("rank_max_5", "五次rank最大"),
            ("inclusion_count_5", "纳入/5"),
        ],
    )
    text += """

MCA_only排名372–535，MLF_only排名542–595；前者并非都靠近541。SER315（372，weight0.594957）和LEU889（384，0.584441）在五次训练扰动中均纳入；VAL250（595，0.413359）五次均未纳入。这是持续的成员分歧，而非只差一位的报告取整。

PRO120虽然为MCA_only且五次均纳入，在R2上h(C)=2.423 Å、h(A)=3.245 Å、h(B)=1.159 Å；它相对B较受限并不意味着相对整个共同集合也同样稳定。VAL250与LYS251作为MLF_only也呈现h(A)>h(B)。两组对框架B的h中位数相近（0.780/0.796 Å），对A为1.202/1.052 Å。完整逐点结果支持相对框架和距离集合的区分，不能据组均值证明一个方法全面更正确。

## 实際软权重与更新诊断：not simply dropped

"""
    text += table(
        overlap,
        [
            ("context", "来源"),
            ("MLF_only_min_weight", "MLF_only最小w"),
            ("MLF_only_max_weight", "最大w"),
            ("MLF_only_zero_weight_count", "零权重数"),
            ("MLF_only_weight_share", "占全部拟合权重份额"),
            ("cutoff_weight", "第k权重"),
            ("cutoff_gap", "第k与k+1差"),
        ],
    )
    text += """

原ADK四个MLF_only的权重份额为1.20356%；训练三个为0.889685%；3nci_A的26个位点为2.58772%。所有权重非零，都实际进入软权重拟合。份额只是权重的加和比例，不是对最终旋转/平移的因果影响，不能由此宣称它们完全没有影响。

读取并复用了生产 `bounded_global_weight_refinement`：每轮在相同首帧参照和当前共享权重下做Kabsch，得到每个代表帧/残基的参考位置距离；实际函数对距离作Huber权重更新，再调用一次原序列连续性处理，按代表帧平均，除以该轮最大平均权重并冻结为下一轮共享权重。四轮后同一 frozen_weights 用于最终拟合，top-k排序仅生成报告集合。CPU float64、代表帧预算256、4轮、δ=1 Å、连续性0.05均未变化。

本次分析侧包装原连续性函数，捕获真实caller里的更新前权重、真实距离、robust更新后的frame weights以及原连续性输出；原函数照常执行，没有替代公式。原代表帧重放ADK full、ADK train与3nci R1后，最终权重均与旧 frozen_weights **逐元素完全一致，最大误差0**。因此这里记录的是原计算路径的数值轨迹，非近似重写。

例如最后一轮中，PRO9的robust帧权重均值为0.336406，THR31为0.341379；原连续性处理后分别为0.339813和0.339614，局部相对次序在这次处理后发生反转，归一化后得到第128/129名。这是实际观察到的单轮中间过程，但未关闭连续性重跑完整流程，不能扩展为已证明“整个差异只由连续性导致”。HSE172、MET174的最终权重较低也与本路径中的参考距离和robust均值一致；这不意味着它们相对共同核心的距离一定更不稳定，后面的独立几何实际不支持这种简单等同。

下面给出ADK原8个位点生成最终权重时的第4轮记录；完整全残基四轮数据包括训练来源及3nci所有差异位点，保存在 [diagnostics.tsv](diagnostics.tsv) 与 cases/*/sampling/。表中的参考RMS displacement不是RMSF，不是最终权重重新拟合之后的残差，也没有被单独用作生物学正确性证据。

"""
    text += table(
        [
            r
            for r in diagnostics
            if r["case"] == "ADK"
            and r["selection_scope"] == "full"
            and r["round"] == "4"
            and r["old_full_group"] in ["MCA_only", "MLF_only"]
        ],
        [
            ("residue_label", "残基"),
            ("pre_update_reference_RMS_displacement_A", "更新前参考RMS displacement Å"),
            ("robust_frame_weight_mean_before_continuity", "robust均值"),
            ("frame_weight_mean_after_continuity_before_global_normalization", "连续性后均值"),
            ("normalized_shared_weight_after_update", "归一化后w"),
        ],
    )
    text += """

MDLovoFit的保存结果只有原生二元成员信息，本轮未重跑其内部搜索或构造假的连续排名。因此无法给每个MCA_only位点下结论说“MDLovoFit因为某一个已证实的局部机制而拒纳它”。保留原输出与本轮实际几何；对其搜索路径的个别拒纳原因仍然未知。

## 独立几何与G²分解

复用原后半段/R2的全部有效距离方差，只对缺失的ADK full描述性方差新增一次分块计算。使用原Cα PDB精度、Å、总体方差ddof=0；剔除self和同链序列距离≤3的对。矩阵按对称性还原，h_i(C)=sqrt(mean_j V_ij)，逐点保存有效partner数。无有效对是NA。共享集合及差异集合都由相应选择源冻结；没有在评估轨迹重选更有利残基。

"""
    text += table(
        [r for r in group_geometry if r["group"] != "neither"],
        [
            ("context", "来源"),
            ("group", "组"),
            ("geometry_reference", "参考集合"),
            ("n_evaluable", "n"),
            ("median_h_A", "h中位数 Å"),
            ("min_h_A", "最小 Å"),
            ("max_h_A", "最大 Å"),
        ],
    )
    text += """

ADK后半段MCA_only的h(C)中位数1.084 Å，高于MLF_only的0.885 Å；全轨迹描述性结果同向（1.104对0.898 Å）。这不支持把MDLovoFit独有位点称作更差或更不稳定。3nci_A两组R2的h(C)中位数为1.055/1.032 Å，范围有明显重叠，且两组都有较高h的个别残基。这里不做残基独立样本显著性检验。

几何分解用共同集合C和每侧U，把有效对分为C–C、C–U、U–U。Q为该类距离方差之和，Q/n为该类均值；对整体G²的贡献是Q除以该方法的**总有效对数**。各类G不能相加，两方法有效对数也不能假设相同。

"""
    text += table(
        [r for r in decomposition if r["context"] != "ADK_full_descriptive"],
        [
            ("context", "来源"),
            ("method", "集合"),
            ("pair_class", "对类别"),
            ("pair_count", "n"),
            ("variance_sum_A2", "Q Å²"),
            ("mean_variance_A2", "Q/n Å²"),
            ("G2_contribution_A2", "对整体G²贡献 Å²"),
        ],
    )
    text += """

上述分解精确复原ADK后半段G=0.799133756/0.790138882 Å以及3nci R2的0.945784683/0.935773000 Å（MCA/MLF），与旧缓存/结果完全一致。共同C–C的Q在两方法间相同，但因总分母不同，其G²贡献也略不同；主要方差和差异来自C–U，U–U亦保留。ADK训练MLF独有172/174/175均互为序列间隔≤3，因此U–U没有有效对，Q为0、Q/n为NA；不能把NA解释成该组内部完全不动。分解是不同集合的几何算术解释，不是对原软权重算法的因果消融。

## 敏感性一：仅改变报告截点

固定原 frozen_weights，不重新拟合。ADK预定Δ=0、−1、+1、−4、+4，即k=128/127/129/124/132；3nci预定Δ=0、−1、+1、−18、+18，即541/540/542/523/559。只有Δ=0是matched-k；MDLovoFit原始集合始终保持不变。

ADK原8个位点中，PRO9、THR31、LEU168、THR175在这五个报告截点之间会切换；VAL117与ASP118始终纳入，HSE172与MET174始终不纳入。ASP118虽然第124名，最小检查k也是124，所以在本检查内不切换；不能任意扩大near-cutoff标签。训练六个位点中只有ASP118、LEU168、THR175在此范围切换。

3nci_A的MCA_only有9/26、MLF_only有14/26在523–559范围内切换；另外17个MCA_only始终纳入、12个MLF_only始终排除。每个位点的rank_minus_k、weight_minus_cutoff、进入/退出所需cutoff变化和各个Δ纳入状态均在 [cutoff_sensitivity.tsv](cutoff_sensitivity.tsv)。报告集合变了而实际软拟合权重未变，这一操作本身不会改变任何原生对齐结果，不能把扩大k得到更高重叠称为更准确。

## 敏感性二：五次训练代表帧扰动

只在原训练源上重估权重，固定首帧参照、256个代表帧、相同输入残基/精度和全部生产科学参数。五个种子事先固定为20260915–20260919：保留训练首末帧，其余254个代表帧从训练内点划分的254个不重叠时间层各均匀抽取一个，排序且无重复。原基准先用原temporally_stratified_indices重放并精确复原；没有把256帧当成另一条完整轨迹交给可能改变行为的全流程。每次只调用既有权重估计函数，用原评估方差缓存评价新冻结集合。

"""
    text += table(
        sampling,
        [
            ("case", "案例"),
            ("seed", "种子"),
            ("jaccard", "对训练基准J"),
            ("dice", "Dice"),
            ("changed_membership_count", "变动成员数"),
            ("rank_spearman", "排名Spearman"),
            ("unit_mass_weight_L1", "单位总和权重L1"),
            ("G_A", "留出G Å"),
        ],
    )
    text += """

ADK五次对训练基准的Jaccard为0.984496–1，每次0或1个成员交换；rank Spearman为0.999170–0.999424，单位总和权重L1为0.01055–0.01390，留出G为0.79913–0.80975 Å。原八个跟踪位点中，VAL117/LEU168始终纳入，HSE172/MET174/THR175始终不纳入；ASP118纳入3/5，排名127–129。PRO9在**训练基准**已为neither，五次0/5；THR31训练基准已为shared，五次5/5。全序列另有ALA11（3/5）和GLY198（4/5）显示成员敏感性，这些也保存，没有只追踪有利差异位点。

3nci_A五次J为0.974453–0.978062，每次6或7个成员交换，排名Spearman为0.998803–0.999300，单位总和权重L1为0.01101–0.01562，R2 G为0.93802–0.94936 Å。原52个差异位点中，19个MCA_only始终纳入、15个MLF_only始终排除，呈现持续的方法间分歧；其余成员不能都归为稳定。

需要特别区分“在五次之间有变化”与“所有五次都偏离原基准”：全序列有20个位点的纳入次数介于0和5之间；另有6个位点在所有五次中都改变了原基准身份。后者为ILE78、GLY79、GLY307从原纳入变0/5，PHE296、VAL453、LYS724从原未纳入变5/5。它们揭示原确定性代表帧与这套扰动抽样之间的差异，不能因为五次结果一致就称为对原代表帧选择不敏感。所有这些点在 [sensitivity_changed_residues.tsv](sensitivity_changed_residues.tsv) 明确分类。

THR209这个取整位点也有代表帧敏感性：rank532–548、纳入4/5；但SER315排名375–397、LEU889排名386–400仍均5/5，VAL250排名585–598仍0/5，远离截点的持续分歧确实存在。没有挑最佳种子替代主结果；没有用R2选择种子/参数。五次纳入数是这套扰动下的经验次数，不是校准概率、独立生物重复或严格不确定性区间。没有给MDLovoFit做对应重跑，不能推出MDCoreAlign比它更稳健。此前旋转/平移检查只作为独立的朝向诊断引用，不替代本次采样敏感性。

## 差异是否改变参考系解释：原生输出与受控诊断分开

复用旧两方法同一源/帧范围的原生RMSF曲线，并报告绝对差异。ADK原全轨迹曲线RMSE=0.406001 Å，最大绝对差1.128514 Å，位于GLY150（LID）；3nci R1为0.133090 Å，最大0.420950 Å，位于PRO840。这比只报相关系数更能揭示局部幅度差异。共同成员、两类独有成员、所有差异位点±3的同链邻域及固定文献域/框架均已逐点和区域汇总，见 [rmsf_by_residue.tsv](rmsf_by_residue.tsv) 与 [rmsf_impact.tsv](rmsf_impact.tsv)。

下表原生行的k仅是matched报告大小，原生MDCoreAlign仍对全部214/903个残基使用非零软权重；只有受控行的k才是实际二元拟合集合大小。

另外，在相同原参考结构和相同评估帧上，使用同一等权Kabsch分别拟合训练冻结的C、C∪U_MCA、C∪U_MLF，分块累计时间均值/总体方差，不输出大型aligned PDB。各参考及评估块的几何非退化检查均可评估。它是固定拟合器下的成员诊断，不是MDCoreAlign原生软权重拟合，也不替代MDLovoFit。

"""
    shown = [r for r in rmsf if r["region"] == "all_residues"]
    text += table(
        shown,
        [
            ("case", "案例"),
            ("analysis_kind", "口径"),
            ("method_a", "A"),
            ("method_b", "B"),
            ("k", "k A/B（原生行=matched报告大小）"),
            ("profile_RMSE_A", "RMSF曲线RMSE Å"),
            ("max_abs_difference_A", "最大绝对差 Å"),
            ("max_resname", "残基名"),
            ("max_source_resid", "编号"),
        ],
    )
    text += """

两侧差异集合用同一拟合器比较时，ADK后半段RMSE=0.064887 Å、最大0.155843 Å（GLU152）；3nci R2为0.034249 Å、最大0.116092 Å（SER163）。在本轮明确规定的诊断下，成员差异造成的RMSF曲线差异较小，并没有抹去相近参考系解释；但没有预设非劣效界限，也不能将较小称为严格无影响。原生与受控结果既涉及不同拟合权重/程序，也涉及不同评估时间源，不能把两个RMSE相减来定量归因“软/硬权重贡献”或“几个残基的全部影响”。

以下保留固定文献区域的绝对差异；框架A/B可能重叠，区域汇总分别计算，不加成总体份额。

"""
    region_rmsf = [
        r
        for r in rmsf
        if r["region"].startswith("literature_")
        and (
            r["analysis_kind"] == "native_full_method_outputs"
            or r["method_b"] == "diagnostic_C_plus_U_MLF"
        )
    ]
    text += table(
        region_rmsf,
        [
            ("case", "案例"),
            ("analysis_kind", "口径"),
            ("region", "文献区域"),
            ("mean_signed_difference_A", "平均A−B Å"),
            ("profile_RMSE_A", "区域曲线RMSE Å"),
            ("max_abs_difference_A", "区域最大绝对差 Å"),
        ],
    )
    text += """

## 图、复现及保护

图均基于同一真实首帧结构和本次完整差异列表。ADK图显示原8个差异残基，3nci图显示全部52个；文字拥挤时完整编号索引表保留，未筛选有利位点。full描述性几何、训练后留出几何和五次训练抽样的来源分别标注。PNG、SVG及PyMOL脚本位于 [figures](figures/)，最终图注记录具体文件名与来源。

[analysis_config.yaml](analysis_config.yaml)、[frame_index_sets.json](frame_index_sets.json)、[run_manifest.json](run_manifest.json) 冻结范围、索引和hash。[reproduce.sh](reproduce.sh) 在新时间戳目录重放本轮分析，拒绝覆盖已有目录。原始生物学结果目录、生产科学代码、输入及旧保护清单保持不变；核对记录见 validation/protected_before.json 和 validation/protected_after.json。只运行新增集合/距离分解、代表帧/诊断、分块RMSF及退化检查相关测试，未重跑历史全套测试。

本轮沿用旧strict benchmark对“更快”的原始计时边界与适用范围；没有新的计时数据，也没有将本轮坐标缓存、作图或受控诊断拼接进16.96×速度结论。仍不能推及包含坐标写出的端到端速度。

## 建议增加到论文中的限定

“相近核心”应解释为高重叠及相近参考系解释，并明确仍存在双向独有成员。部分差异由报告截点放大、部分对代表帧选择敏感、部分在五次扰动下持续且远离截点。共同参考集合的独立几何和受控拟合没有支持MDCoreAlign独有位点全面更优；其主要价值仍由旧速度证据与当前有限功能对应共同支撑，而非逐残基唯一真值。

英文导师回复见 [supervisor_reply_en.txt](supervisor_reply_en.txt)（150–220词）；正文一句结论、简短Results/Methods与限制见 [manuscript_sensitivity_insert.md](manuscript_sensitivity_insert.md)。完整机器可读证据以各TSV原精度为准，文中仅为显示而四舍五入。
"""
    (run / "core_disagreement_report.md").write_text(text)
    write_json(
        run / "validation/report_numeric_inventory.json",
        {
            "n_contexts": len(overlap),
            "n_membership_rows": len(members),
            "n_disagreement_rows": len(differences),
            "n_seed_runs": len(sampling),
            "n_sensitivity_changed_residues": len(sampling_changes),
            "source_run": config["source_run"],
        },
    )
    print("Chinese report and combined evidence written", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    generate(parser.parse_args().run.resolve())
