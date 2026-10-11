"""Aggregate measured case tables and write the bounded biological report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.atlas_strict_single_thread_cpu_v1.common import write_json, write_tsv


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def table(data, columns):
    result = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in data:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            try:
                if isinstance(value, bool):
                    value = "是" if value else "否"
                elif key in {"pair_count", "core_size", "overlap", "region_size"} and value != "":
                    value = str(int(value))
                else:
                    value = f"{float(value):.4g}" if value != "" else "不可评估"
            except (ValueError, TypeError):
                pass
            values.append(str(value))
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


def report(run):
    config = json.loads((run / "analysis_config.json").read_text())
    summaries, residue_rows, region_rows, geometry, orientation, rmsf_metrics = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    failures = []
    for case in config["cases"]:
        name = case["case_id"]
        p = run / "cases" / name / "analysis"
        if not (p / "case_summary.json").exists():
            failures.append(name)
            continue
        summary = json.loads((p / "case_summary.json").read_text())
        summaries.append(summary)
        residue_rows.extend(rows(p / "residue_results.tsv"))
        region_rows.extend(rows(p / "region_metrics.tsv"))
        geometry.extend(rows(p / "geometry_metrics.tsv"))
        orientation.append({"case_id": name, **summary["orientation"]})
        profiles = dict(np.load(p / "alignment_profiles.npz"))
        for method, values in profiles.items():
            for region, mask in case["regions"].items():
                rmsf_metrics.append(
                    {
                        "case_id": name,
                        "method": method,
                        "region": region,
                        "scope": "full",
                        "region_mean_RMSF_A": float(values[np.asarray(mask)].mean()),
                        "profile_RMSE_to_literature_CORE_A": float(
                            np.sqrt(np.mean((values - profiles["literature_CORE"]) ** 2))
                        )
                        if "literature_CORE" in profiles
                        else None,
                    }
                )
    for filename, data in [
        ("residue_results.tsv", residue_rows),
        ("region_metrics.tsv", region_rows),
        ("case_metrics.tsv", geometry),
        ("rmsf_comparison.tsv", rmsf_metrics),
        ("orientation_checks.tsv", orientation),
    ]:
        write_tsv(run / filename, data)
    write_json(
        run / "case_status.json",
        {
            "completed": [s["case_id"] for s in summaries],
            "incomplete": failures,
            "scientific_interpretation": "technical completion does not imply all biological hypotheses supported",
        },
    )
    if failures:
        (run / "biological_validation_report.md").write_text(
            "# Partial biological validation\n\n"
            + "Incomplete cases: "
            + ", ".join(failures)
            + ". Inspect validation/*_analysis.log; no complete-case scientific conclusion is asserted.\n\n"
            + table(
                geometry, [("case_id", "Case"), ("method", "Frozen set"), ("G_A", "Held-out G (A)")]
            )
        )
        print("Partial report; incomplete:", failures, flush=True)
        return
    head = """# MDCoreAlign：有限案例的稳定拟合参考系验证

ADK 最适合进入论文正文：其证据同时来自预先冻结的外部结构域注释、训练与评估分离的内部距离几何，以及实际采样到的结构域相对运动。I-FABP 保留为实验区域对照的有限补充，明确呈现不支持预期的区域；3nci_A 提供外部晶体框架与另一条模拟重复上的几何补充。这里的“稳定”仅指当前采样条件下适合作为刚体拟合参考的相对稳定区域。

## 输入、来源与冻结配置

[case_inventory.tsv](case_inventory.tsv)、[residue_mapping.tsv](residue_mapping.tsv)、[literature_annotations.tsv](literature_annotations.tsv) 和 [analysis_config.yaml](analysis_config.yaml) 在实际自动核心估计之前写入；完整 JSON 保存每个训练/测试帧下标、真实残基标识、编号映射和注释布尔向量。注释没有传给任何自动估计调用，未按注释筛掉输入残基。每例所有可用帧均被保留。

- ADK：官方 MDAnalysisData 的完整 4,187 帧、214 个 Cα，240 ps/帧，存储跨度 1004.64 ns；本地 PSF/DCD 的 SHA-256 与官方记录一致，早期 500 帧前缀未被当成全轨迹。序列与 E. coli K12 UniProt P69441 全长一致。官方说明为 apo，已按 CORE 做 RMSD 预对齐；PBC unwrap/nojump 的具体流程未知。训练 0–2092，评估 2093–4186。
- I-FABP：官方完整 500 帧、131 个 Cα；大鼠 P02693 的成熟链，与 canonical 位点 2–132 一致。文献/本地位点 r 对应 UniProt r+1，六个文献端点及 1IFC 成熟链序列支持映射。模型包含蛋白和水，无脂肪酸，但原始 apo/holo 建模来源未知。官方模拟长度 500 ps、1 ps/帧，存储跨度 499 ps；官方文件自身的 DCD 时间头为无效负数，图采用原始帧序号×官方 1 ps 的相对轴，绝对起始时刻未知。已有蛋白 RMSD 预对齐，PBC 处理未知。训练 0–249，评估 250–499。
- 3nci_A：已有本地 ATLAS R1/R2，分别 10,001 帧、903 个 Cα、10 ps/帧、100 ns。R1 选核心，R2 全部帧评估；主文献说明独立初速度种子，二者仍共享结构与模拟协议，不是独立生物学实验。完整序列及编号映射为 RB69 gp43 / Escherichia phage RB69 / Q38087，保留 D222A/D327A；没有相对沉积 3NCI 的缺失或末端标签。完整 TPR 仅有蛋白、水和 Na/Cl，没有晶体中的 DNA、dCTP、Ca。R1 现有 Cα 缓存的 11 个预定帧与 XTC 完全一致；R2 从已有本地归档提取，没有新模拟或远程大型下载。归档说明去除 PBC，拟合选择和参考的精确流程未知。

PSF/ATLAS PDB 缺失的原生 chainID 没有被伪造为来源事实：输出链 A 是显式记录的单蛋白 segid/序列映射。完整序列、突变、组分、来源 hash 和未知项见 sources/local_inventory.json、sources/3nci_inventory.json。

## 算法口径与独立几何

使用与严格 AT5 速度结论相同的 `mdcorealign.benchmark_minimal.solve_minimal_cpu(mode="throughput")`：CPU float64、最多 256 个时间分层代表帧、4 轮共享权重更新、原有 Huber δ=1 Å、序列连续性强度 0.05。直接导出真正用于最终拟合的 `frozen_weights`；权重较大表示该拟合模型赋予更高相对可信度，不能换算为 NMR 序参数。连续权重是主要输出，60% top-k 仅作报告，保留 floor(0.6N+0.5) 和输入下标并列规则，没有硬阈值替换软权重拟合。

MDLovoFit 使用原安装程序 `-f 0.600000 -t`，所有帧的占有率和残基身份均检查，并保存原始逐帧 masks。三例实际得到的核心类型见下表；没有预先假定其固定。若变化，脚本才使用明确标为 derived frequency consensus 的同大小汇总。matched-k 保留既有按 MDLovoFit 实际整数大小选择最高 MDCoreAlign 权重的规则。ADK 均为 128；I-FABP 原生 MDCoreAlign 79、MDLovoFit 78，故另列 matched-k 78；3nci_A 的原生与 matched-k 大小均按实际导出记录列于结果表。

所有比较先共用同一个每帧平移居中的 Cα PDB，再以 0.001 Å 精度重读，以确保两种程序使用完全相同坐标/帧/顺序。自动软权重、实际 MDLovoFit 输出、全 Cα 等权 Kabsch，以及 ADK 的文献 CORE-only Kabsch 均使用同一首帧参照。MDLovoFit 输出施加一次共同刚体锚定以对齐首帧；这一操作不改变 RMSF 或内部距离。RMSF 均为相应对齐轨迹关于其自身时间均值的平方偏差均方根，单位 Å。文献 CORE 对齐是知识指导参照，其拟合区域残差低不是独立正确性证据。

全轨迹结果仅用于外部注释/可视化；留出几何使用只在训练源估计并冻结的集合。在评估帧上分块计算全部有效 Cα 距离的总体方差（ddof=0），排除同链序列间隔 ≤3 的对，计算 G(S)=sqrt(mean Var_test[d_ij])。没有有效对则不可评估，不记零。每个大小从同一全 Cα 残基集合均匀采样 1,000 个集合（seed=20260914），距离方差只计算一次。随机分布没有控制紧凑性、二级结构或距离尺度等全部混杂；百分位是描述位置，不是生物学 p 值。没有将帧、残基对或随机集合当作独立生物重复，也没有虚构置信区间。

## ADK：推荐正文

Li et al. (2015) Figure 1 的范围预先固定为 CORE 1–29/60–121/160–214，NMP 30–59，LID 122–159；所有 214 位点都可映射。CORE 是相对结构域参照，原文还讨论部分 CORE 区段 cracking，不能视为每个残基绝对不动。4AKE 是开放未配体结构背景；1AKE 是 Ap5A 抑制剂复合物，不能称为直接 ATP/AMP 实验复合物，两份晶体不是独立 MD 重复。[Li2015](https://doi.org/10.1016/j.bpj.2015.06.059)

下表为全轨迹的外部 CORE 对应；purity 的分母为报告集合，coverage 的分母为 146 个文献 CORE 残基，enrichment 的背景为 146/214。

"""
    text = head + table(
        [
            r
            for r in region_rows
            if r["case_id"] == "ADK" and r["scope"] == "full" and r["region"] == "CORE"
        ],
        [
            ("method", "方法"),
            ("overlap", "CORE 重叠"),
            ("core_size", "集合大小"),
            ("purity", "纯度"),
            ("coverage", "覆盖"),
            ("enrichment", "富集"),
        ],
    )
    text += """

MDCoreAlign 在 CORE/LID/NMP 的平均权重分别为 0.655/0.088/0.133；其全轨迹核心含 127 个 CORE、0 个 LID、1 个 NMP 残基，MDLovoFit 对应 126/0/2。两方法得到相近的 CORE 参考系解释，但集合不同，matched-k Jaccard=0.939、Dice=0.969。这种方法间一致性不替代下面的独立证据。

在未用于选核心的后半段，MDCoreAlign 和 MDLovoFit 的 G 分别为 0.799 和 0.790 Å，均低于同大小随机中位数 1.482 Å。这个案例没有显示 MDCoreAlign 几何优于 MDLovoFit，也不需要该结果。146 位点的文献 CORE 的 G=0.897 Å、同大小随机参照 1.483 Å；不同大小的原始 G 不能直接作胜负。

四种参考系的逐残基 RMSF 曲线保存在 [residue_results.tsv](residue_results.tsv) 及 cases/ADK/analysis/alignment_profiles.npz；[rmsf_comparison.tsv](rmsf_comparison.tsv) 保存区域 RMSF 均值和曲线 RMSE 汇总。相对文献 CORE-only 曲线，MDCoreAlign/MDLovoFit/全 Cα 的全序列 RMSF 曲线 RMSE 分别为 0.131/0.351/0.710 Å。这是参考系一致性的描述，不是以 CORE 自身较低残差定义正确性。

不依赖拟合的 Cα 几何中心距离显示，CORE–LID 的 P5–P95 为 28.40–34.11 Å，CORE–NMP 为 21.53–23.88 Å，LID–NMP 为 30.87–42.11 Å；LID–CORE–NMP 夹角为 71.74–94.35°。因此轨迹确实存在结构域相对运动，而不是仅从预对齐坐标的低 RMSF 推断核心。LID/NMP 各自内部的留出 G 仅为 0.447/0.449 Å：这恰说明“相对主体运动”不等于域内所有残基无序，也不能与较大的 CORE 集合直接排名。

[ADK 主图 PDF](figures/adk_main.pdf) / [PNG](figures/adk_main.png)；[运动与固定帧叠合 PDF](figures/adk_motion.pdf) / [PNG](figures/adk_motion.png)。结构来自真实首帧，叠合使用事先确定的 7 个等间隔帧，未挑选最大变化帧。

## I-FABP：保留有限实验对照与负面结果

Hodsdon & Cistola (1997) 的大鼠 apo/holo NMR 原始摘要及官方出版信息支持 apo 动态区 26–35、53–56、73–76；只有这 18 个位点被标为实验动态区，其余 113 个 unknown。原文/SI 正常公开访问返回 ACS 403，未取得逐残基数值，未从图猜 S² 或 relaxation 数据，也没有算虚构相关性。模拟的原始状态来源不能完全匹配实验，短 500 ps 也不能充分覆盖实验时间尺度。[原始实验](https://doi.org/10.1021/bi962018l)

"""
    text += table(
        [
            r
            for r in region_rows
            if r["case_id"] == "I-FABP"
            and r["scope"] == "full"
            and r["region"] != "portal_combined"
            and r["method"] in ["MDCoreAlign_native60", "MDLovoFit"]
        ],
        [
            ("region", "实验动态区"),
            ("method", "方法"),
            ("overlap", "纳入数"),
            ("region_size", "区域大小"),
            ("region_mean_weight", "MDCoreAlign 平均权重"),
        ],
    )
    text += """

26–35 的 MDCoreAlign 平均权重 0.951、10/10 被纳入，MDLovoFit 为 7/10；这一观察不支持“实验动态区都应低权重”的简单预测。53–56 为 2/4 对 0/4，73–76 两者均 0/4。只有部分区域表现出同向对应，不能称为定量预测或重现 NMR。表中 MDCoreAlign 权重在 MDLovoFit 行重复仅便于同区比较，并不是 MDLovoFit 的权重。

留出 matched-k 的 G 为 0.377（MDCoreAlign）和 0.368 Å（MDLovoFit），随机中位数 0.493 Å；二者都有有限几何支持。18 个 portal 残基的组合 G=0.715 Å、随机中位数 0.486 Å，但这是动态区域参照，不能把其标签改为稳定核心。53–56 和 73–76 各只有 4 个连续残基，按既定近邻排除规则无有效内部对，G 不可评估。unknown-aware 区域表的 purity/enrichment 只在 18 个注释位点内定义；尤其 portal_combined 的 1.0 是条件分母的代数结果，不能作为稳定性/实验支持证据。约 84.8% 的 MDCoreAlign 原生核心和 91.0% 的 MDLovoFit 核心没有本次实验注释。

[I-FABP 补充图 PDF](figures/ifabp_supplement.pdf) / [PNG](figures/ifabp_supplement.png)。推荐保留为 exploratory/limited，完整呈现区域差异；不作为论文主要实验验证。

## 3nci_A：分别保留两个晶体框架

Ren (2016) Figure 3 的 framework A（231 残基）和 B（318 残基）按原文完整范围映射，交集 29；两个框架可以相对运动，未将其并集定义为唯一真核心，未事后仅选择有利框架。注释是跨晶体结构的相对不变框架证据，不能直接证明当前无 DNA 的突变体发生转位或校对过程。[Ren2016](https://doi.org/10.1093/nar/gkw555)

"""
    text += table(
        [
            r
            for r in region_rows
            if r["case_id"] == "3nci_A"
            and r["scope"] == "full"
            and r["method"] in ["MDCoreAlign_native60", "MDLovoFit"]
        ],
        [
            ("region", "框架"),
            ("method", "方法"),
            ("overlap", "重叠"),
            ("core_size", "核心大小"),
            ("purity", "纯度"),
            ("coverage", "覆盖"),
            ("enrichment", "富集"),
        ],
    )
    text += """

MDCoreAlign 原生 542 位点核心覆盖框架 A 的 225/231（97.4%）及 B 的 261/318（82.1%）；MDLovoFit 的 541 位点核心为 227/231（98.3%）及 265/318（83.3%）。MDCoreAlign matched-k=541 时 B 为 260/318。R1 冻结的 matched-k 集合在 R2 上 G=0.946 Å（MDCoreAlign）和 0.936 Å（MDLovoFit），同大小随机中位数 2.225 Å；两者获得跨模拟重复的几何支持，MDCoreAlign 没有低于 MDLovoFit。两框架各自 G 为 0.557/0.731 Å，也各低于自己的等大小随机参照。

两个外部框架及自动集合的 R2 几何结果均列于下表。不同框架大小各有自己的随机参照，不能把较低 G 的框架事后选作“真值”。这提供另一条模拟重复上的评估，仍不能证明所有状态普适生物学等价。

[3nci_A 两框架图 PDF](figures/3nci_frameworks.pdf) / [PNG](figures/3nci_frameworks.png)。

## 全部留出几何结果

"""
    text += table(
        geometry,
        [
            ("case_id", "案例"),
            ("method", "冻结集合"),
            ("core_size", "残基数"),
            ("pair_count", "有效对数"),
            ("G_A", "G (Å)"),
            ("random_median_G_A", "等大小随机中位数 (Å)"),
            ("G_over_random_median", "G/随机中位数"),
        ],
    )
    text += """

## 预对齐与朝向诊断

每例使用预定全范围均匀的 64 帧，首帧保持原样，其余帧独立随机旋转并平移（seed=20260914），在同一小样本上重跑两种方法。这里比较的是同一小样本扰动前后，而非把小样本权重与全轨迹权重混用。PDB 写入前的内部距离仅有浮点舍入差异，写入后包含 0.001 Å 量化影响。该检查没有发现当前样本上的实质朝向依赖；它不能证明所有输入/初始参考或采样下的严格不变性。

"""
    text += table(
        orientation,
        [
            ("case_id", "案例"),
            ("max_MDCoreAlign_weight_difference", "最大权重差"),
            ("MDCoreAlign_core_jaccard", "MDCoreAlign 核心 J"),
            ("MDLovoFit_core_jaccard", "MDLovoFit 核心 J"),
            ("geometry_absolute_difference_A", "固定集合 G 差 (Å)"),
            ("max_pair_distance_error_after_PDB_A", "量化后最大距离差 (Å)"),
        ],
    )
    text += "\n\n" + table(
        [
            {
                "case_id": s["case_id"],
                "core_kind": s["MDLovoFit_core_kind"],
                **s["full_core_agreement"],
            }
            for s in summaries
        ],
        [
            ("case_id", "案例"),
            ("core_kind", "本次实测 MDLovoFit 类型"),
            ("jaccard", "全轨迹 matched-k Jaccard"),
            ("dice", "Dice"),
            ("exact_match", "完全相同"),
        ],
    )
    text += """

## 与既有速度证据的关系、保护及复现

已定位并只读核对 `results/atlas_strict_single_thread_cpu_v1/09_report/final_atlas_single_thread_report.md`。其中 8 个 ATLAS R1 系统×6 个前缀、同一 Cα PDB、单逻辑 CPU/单数值线程、warm cache、配对顺序、5 次正式重复的 no-materialisation 计时，报告 16.9553×（按系统聚类 95% CI 14.3589–19.5988），全长组 21.7902×。边界从 PDB 解析至正常核心估计及全帧拟合/RMSD，排除原 XTC 解码、坐标写出、RMSF、GPU 等。本轮使用相同 throughput 科学配置完成生物学案例，不重跑计时，不将 exact/reference 或本轮绘图/几何耗时混入速度结论。原报告方法间 matched-k 中位 J=0.868966/D=0.929888、exact 0/48、自一致性 8/8 均只属于该冻结面板；不是本轮三案例的生物学准确率。

本次只新增分析脚本、相关测试与此结果目录。复用原 1,267 个保护 hash，并对整个旧 strict 结果目录另存 2,019 文件快照；核查记录见 validation/protected_before.json 和 validation/protected_after.json。未修改算法、默认参数、历史结果、输入、保护清单或 Git 历史，未提交/推送。新相关测试与真实小型 smoke 的结果见 validation/smoke/；没有重跑历史全套测试。

[run_manifest.json](run_manifest.json) 保存配置/输入来源和实现 hash；[reproduce.sh](reproduce.sh) 在新时间戳目录重放相同冻结输入、注释和帧索引，旧产出不被覆盖。每例保留原生 occupancy masks、频率、每残基权重/报告集合、距离方差、随机分布、命令及输出 hash；验证后的大 MDLovoFit PDB 被删除以控制重复占用，失败原始输出保留。实际叠合 PDB、RMSF 和真实结构渲染脚本仍保存。

## 可支持和不可支持的论文表述

- **正文：ADK。** 外部结构注释支持与时间块几何支持相互补充，并有朝向诊断和实际域运动。它支持合理的稳定参考系选择，不只是与 MDLovoFit 高重叠。
- **补充：3nci_A。** 独立晶体框架注释和 R1→R2 模拟几何；两个框架都报告，状态/突变限制保留。
- **有限补充：I-FABP。** 实验区域来源真实，但部分对应不成立，采样短、历史状态不完全匹配、无逐残基实验数值。因此本轮没有充分的实验动态重现证据。
- 自动方法的集合与 RMSF 解释相近，但不完全相同；没有证明 MDCoreAlign 全面优于 MDLovoFit、唯一正确核心、热稳定性或普适生物学等价。

建议英文结论：**In the ADK trajectory examined, MDCoreAlign selected a fitting core consistent with independent domain annotations and stable held-out internal geometry, supporting a reference-frame interpretation similar to MDLovoFit without establishing universal biological equivalence.**

英文 Methods/Results/Limitations/Reviewer response 草稿见 [manuscript_insert.md](manuscript_insert.md)；原始来源见 [references.tsv](references.tsv) 和 [references.bib](references.bib)，公开资料摘录与访问限制见 sources/。完整数值文件优先于正文显示的四舍五入值。
"""
    if failures:
        text += "\n未完成案例（保留原因/log继续其余案例）：" + ", ".join(failures) + "。\n"
    (run / "biological_validation_report.md").write_text(text)
    print("Aggregated", len(summaries), "cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    report(parser.parse_args().run.resolve())
