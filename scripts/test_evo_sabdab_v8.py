#!/home/dataset-local/anaconda3/envs/merf/bin/python
"""无 GPU 逻辑测试：独立 v8 的参数、coverage 和历史最佳策略恢复。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_v8():
    path = ROOT / "scripts" / "evo_sabdab_v8.py"
    spec = importlib.util.spec_from_file_location("evo_sabdab_v8_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 evo_sabdab_v8.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_independent_source(v8):
    source = (ROOT / "scripts" / "evo_sabdab_v8.py").read_text(encoding="utf-8")
    forbidden = ("importlib.util", "_load_legacy_module", "v8_grpo")
    for marker in forbidden:
        if marker in source:
            raise AssertionError(f"v8 仍保留 wrapper 依赖: {marker}")
    if not callable(v8.run_one_dataset) or not callable(v8.train_one_epoch):
        raise AssertionError("独立 v8 缺少直接训练入口")


def _test_args(v8):
    original_argv = sys.argv
    try:
        sys.argv = [
            "evo_sabdab_v8.py",
            "--actions-per-triple",
            "3",
            "--inner-passes",
            "2",
            "--enable-early-stop",
            "true",
            "--dataset_idx",
            "0",
        ]
        args = v8.get_args()
    finally:
        sys.argv = original_argv
    if args.actions_per_triple != 3 or args.inner_passes != 2:
        raise AssertionError("v8 专属参数解析错误")
    if args.inner_epochs != 2:
        raise AssertionError("旧 --inner_epochs 兼容字段没有同步")
    if Path(args.fixed_dir) != ROOT / "data/sabdab/PDBs_fixed_v8":
        raise AssertionError("v8 默认 RepairPDB 缓存路径错误")


def _test_balanced_coverage(v8):
    torch.manual_seed(17)
    expected_updates = {30: 2, 60: 4, 105: 8}
    for candidate_count, update_count in expected_updates.items():
        batches = list(v8.balanced_inner_batches(candidate_count, 32, 2))
        if len(batches) != update_count:
            raise AssertionError(f"N={candidate_count} update 数错误")
        for pass_index in range(2):
            flattened = [
                item
                for current_pass, indices in batches
                if current_pass == pass_index
                for item in indices.tolist()
            ]
            if sorted(flattened) != list(range(candidate_count)):
                raise AssertionError(f"N={candidate_count} pass={pass_index} 未精确覆盖")


def _test_action_gather(v8):
    probabilities = torch.tensor(
        [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]], dtype=torch.float32
    )
    torch.manual_seed(9)
    action_rows = v8.sample_distinct_action_rows(probabilities, 3)
    if len({tuple(row.tolist()) for row in action_rows}) != 3:
        raise AssertionError("完整动作采样没有去重")

    class DummyModel:
        def choose_best_action(self, policy_batch, device):
            return torch.tensor(
                [[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]], dtype=torch.float32
            )

    mutation_mask = torch.tensor([[True, True]])
    policy_batch = {"wt": {"aa": torch.tensor([[1, 2]])}}
    log_probs = v8.non_wt_log_probs(DummyModel(), policy_batch, mutation_mask, "cpu")
    # 两个 WT 动作必须被屏蔽，因而不可能被 greedy / sampling 当作有效动作。
    if torch.argmax(log_probs[0, 0]).item() == 1 or torch.argmax(log_probs[0, 1]).item() == 2:
        raise AssertionError("native action 没有被屏蔽")
    target_actions = torch.tensor([[0, 1]])
    joint = v8.selected_joint_log_probs(log_probs, mutation_mask, target_actions)
    expected = log_probs[0, 0, 0] + log_probs[0, 1, 1]
    if not torch.allclose(joint, expected.reshape(1)):
        raise AssertionError("联合 action gather 与三个实际动作不一致")


def _metrics(value: float) -> dict:
    return {
        "deterministic_greedy_top8_mean_acquisition": value,
        "deterministic_greedy_top8_best_acquisition": value - 0.1,
    }


def _test_best_checkpoint_restore(v8):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    state = v8.initial_early_stop_state()
    state, updated = v8.capture_best_policy(state, model, -1.00, 1, _metrics(-1.00))
    if not updated or state["best_policy_epoch"] != 1:
        raise AssertionError("未保存初始历史最佳策略")

    # -1.01 虽然小于 min_delta=0.03，却仍是实际最佳，必须更新 checkpoint。
    with torch.no_grad():
        model.weight.fill_(2.0)
    state, updated = v8.capture_best_policy(state, model, -1.01, 2, _metrics(-1.01))
    if not updated or state["best_policy_epoch"] != 2:
        raise AssertionError("精确更优但未达 min_delta 的策略没有被保存")
    with torch.no_grad():
        model.weight.fill_(3.0)
    state = v8.restore_best_policy(state, model)
    if not torch.allclose(model.weight.detach(), torch.tensor([[2.0]])):
        raise AssertionError("恢复的不是历史最佳权重，而是当前权重")
    if state["restored_epoch"] != 2:
        raise AssertionError("恢复 epoch 记录错误")


def _test_patience_state(v8):
    class Args:
        enable_early_stop = True
        early_stop_min_epochs = 3
        early_stop_patience = 2
        early_stop_min_delta = 0.03

    state = v8.initial_early_stop_state()
    expected_stop = {1: False, 2: False, 3: False, 4: True}
    for epoch, value in enumerate((-1.00, -1.01, -0.99, -0.98), start=1):
        state, should_stop, _ = v8.update_early_stop_state(state, value, epoch, Args())
        if should_stop != expected_stop[epoch]:
            raise AssertionError(f"epoch={epoch} patience 状态错误: {state}")
    if state["stop_epoch"] != 4:
        raise AssertionError("patience 到期时未记录 stop epoch")
    source = (ROOT / "scripts" / "evo_sabdab_v8.py").read_text(encoding="utf-8")
    if 'if epoch_metrics.get("early_stop_triggered", False):' not in source:
        raise AssertionError("训练主循环未消费 early-stop 信号")


def _test_true_early_stop_break(v8):
    """用最小 mock 验证 run_one_dataset 会在恢复策略后真正退出 outer loop。"""
    class DummyDataset:
        pdb_id = "dummy"
        fixed_wt_pdb_path = "dummy_fixed.pdb"

    class DummyLossHistory:
        def __init__(self, save_path, is_evolve):
            self.save_path = save_path
            Path(save_path).mkdir(parents=True, exist_ok=True)

        def write(self, _text):
            pass

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    class DummyModel(torch.nn.Module):
        def __init__(self, _args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    pool = {
        "mutate_info_list": ["A1C,B2D,C3E"],
        "center_scores": torch.tensor([1.0]),
        "std_scores": torch.tensor([0.0]),
        "score_stack": torch.tensor([[1.0]]),
        "acquisition": torch.tensor([0.1]),
    }
    args = SimpleNamespace(
        total_epochs=5,
        lr=1e-3,
        weight_decay=0.0,
        model_load_path="unused",
        reward_model_paths="unused",
        periodic_external_eval_interval=1,
        final_top_k=8,
        run_pyrosetta=False,
        run_colabdesign=False,
    )
    train_epochs = []
    monitored_epochs = []
    originals = {
        name: getattr(v8, name)
        for name in (
            "build_dataset",
            "LossHistory",
            "SummaryWriter",
            "MERF",
            "load_pretrained",
            "train_one_epoch",
            "monitor_current_policy_external_metrics",
            "nominate_and_score_candidates",
        )
    }

    def fake_train(*_values, **_kwargs):
        epoch = _values[5]
        train_epochs.append(epoch)
        state = args._v8_early_stop_state
        state.update(
            {
                "stop_epoch": epoch + 1,
                "best_policy_epoch": epoch + 1,
                "best_policy_metric": -1.0,
                "policy_restored": True,
                "restored_epoch": epoch + 1,
            }
        )
        return {"early_stop_triggered": True}

    def fake_monitor(*_values, **kwargs):
        monitored_epochs.append(kwargs["epoch"])
        return {}

    try:
        v8.build_dataset = lambda *_values: DummyDataset()
        v8.LossHistory = DummyLossHistory
        v8.SummaryWriter = DummyWriter
        v8.MERF = DummyModel
        v8.load_pretrained = lambda *_values: None
        v8.train_one_epoch = fake_train
        v8.monitor_current_policy_external_metrics = fake_monitor
        v8.nominate_and_score_candidates = lambda *_values: pool
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = v8.run_one_dataset(
                args=args,
                row={"partner": "A", "antibody_chain": "B"},
                dataset_index=0,
                run_dir=temp_dir,
                device=torch.device("cpu"),
                checkpoint_args_path=None,
                checkpoint_model_arg_keys=[],
                reward_paths=["unused"],
            )
    finally:
        for name, value in originals.items():
            setattr(v8, name, value)

    if train_epochs != [0] or monitored_epochs != [0]:
        raise AssertionError("early stop 后 outer loop 没有立即停止")
    if not summary["training_stopped_early"] or summary["final_policy_epoch"] != 1:
        raise AssertionError("最终提名没有使用恢复后的历史最佳策略")


def main():
    v8 = _load_v8()
    _test_independent_source(v8)
    _test_args(v8)
    _test_balanced_coverage(v8)
    _test_action_gather(v8)
    _test_best_checkpoint_restore(v8)
    _test_patience_state(v8)
    _test_true_early_stop_break(v8)
    print(
        "v8 test passed: 独立入口、参数别名、coverage p=2、"
        "精确历史最佳权重恢复和 patience 判据均正确。"
    )


if __name__ == "__main__":
    main()
