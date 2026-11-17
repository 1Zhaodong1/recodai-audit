# -*- coding: utf-8 -*-
"""Step 1 scan：被审计仓库全部配置中的损失条目 vs 构造函数签名

配置里的损失写成 dict(type='XxxLoss', ...) —— 这是 Call 节点而非 Dict 字面量。
对每个损失调用抽取关键字，检查：
  * 键不在 __init__ 签名中、且构造函数有 **kwargs  -> 被静默吞掉（与缺陷 1 同类）
  * 键不在签名中、且没有 **kwargs                -> 构建时抛 TypeError（显性错误）
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("path/to/mmsegmentation")
LOSS_DIR = ROOT / "mmseg" / "models" / "losses"
OUT_DIR = Path(__file__).resolve().parent
CONFIG_DIRS = [("author", ROOT / "my_config"), ("vendored", ROOT / "configs")]


def class_signatures():
    out = {}
    for py in LOSS_DIR.glob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    a = item.args
                    params = [x.arg for x in a.args if x.arg != "self"]
                    params += [x.arg for x in a.kwonlyargs]
                    out[node.name] = {
                        "params": set(params),
                        "has_kwargs": a.kwarg is not None,
                        "file": py.name,
                        "line": node.lineno,
                    }
    return out


def loss_calls(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "dict"):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        t = kw.get("type")
        if isinstance(t, ast.Constant) and isinstance(t.value, str) and t.value.endswith("Loss"):
            found.append({"config": path.name, "line": node.lineno,
                          "type": t.value, "keys": [k for k in kw if k != "type"]})
    return found


def main():
    sigs = class_signatures()
    print(f"损失类定义: {len(sigs)} 个 -> {sorted(sigs)}")

    findings, unknown = [], set()
    stats = {}
    for label, d in CONFIG_DIRS:
        entries = 0
        for cfg in sorted(d.rglob("*.py")):
            for item in loss_calls(cfg):
                entries += 1
                sig = sigs.get(item["type"])
                if sig is None:
                    unknown.add(item["type"])
                    continue
                extra = [k for k in item["keys"] if k not in sig["params"]]
                if extra:
                    findings.append(dict(item, source=label, extra=extra,
                                         kind="silently_ignored" if sig["has_kwargs"] else "build_error",
                                         loss_file=sig["file"], has_kwargs=sig["has_kwargs"]))
        stats[label] = entries
    checked = sum(stats.values())
    print("按来源统计条目数:", stats)

    print(f"扫描到损失配置条目: {checked}")
    for label in ("author", "vendored"):
        sub = [f for f in findings if f.get("source") == label]
        print(f"  {label}: {stats.get(label, 0)} 条, 命中 {len(sub)} 条")
    print(f"未在 losses/ 中定义的类型: {sorted(unknown) or '无'}")
    print()
    for f in findings:
        print(f"[{f['kind']}] {f['config']}:{f['line']} {f['type']} 多余键={f['extra']} "
              f"(**kwargs={f['has_kwargs']}, {f['loss_file']})")
    if not findings:
        print("未发现配置键与签名不一致的条目。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "loss_config_scan.json").write_text(
        json.dumps(dict(checked=checked, findings=findings,
                        classes={k: dict(params=sorted(v["params"]), kwargs=v["has_kwargs"])
                                 for k, v in sigs.items()}), indent=1, ensure_ascii=False),
        encoding="utf-8")
    print("\n写入:", OUT_DIR / "loss_config_scan.json")


if __name__ == "__main__":
    main()
