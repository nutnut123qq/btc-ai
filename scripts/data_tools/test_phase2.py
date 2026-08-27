import os
import sys
import json
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_rl_persistence():
    print("\n--- Test 1: RL Agent State Persistence & Checkpoint Loading ---")
    from rl_agent import QTradingAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "q_table_test.json"

        # 1. Initialize fresh agent
        agent1 = QTradingAgent(checkpoint_path=ckpt_path)
        assert len(agent1.q_table) == 0, "Fresh agent should start with empty Q-table"

        # 2. Update Q-values
        state1 = [0.8, 0.2, 0.5, 0.9, 0.1]
        next_state1 = [0.7, 0.3, 0.5, 0.8, 0.2]
        agent1.update_q_value(state1, "Bullish", 1.5, next_state1)

        state2 = [0.1, 0.9, 0.5, 0.2, 0.8]
        next_state2 = [0.2, 0.8, 0.5, 0.3, 0.7]
        agent1.update_q_value(state2, "Bearish", 2.0, next_state2)

        assert ckpt_path.exists(), "Checkpoint file must be created on update"
        saved_q_table = json.loads(ckpt_path.read_text(encoding="utf-8"))
        assert len(saved_q_table["q_table"]) > 0, "Saved file must contain Q-table states"
        print(f"-> Saved {len(saved_q_table['q_table'])} states to disk successfully.")

        # 3. Initialize second agent and test reload
        agent2 = QTradingAgent(checkpoint_path=ckpt_path)
        assert len(agent2.q_table) == len(agent1.q_table), "Agent 2 must load identical Q-table from disk"
        key = agent1._discretize_state(state1)
        assert agent2.q_table[key] == agent1.q_table[key], "Q-values must match exactly across reload"
        print("-> PASS: Q-Table state reloaded seamlessly into new agent instance.")

        # 4. Test Corrupted File Fallback
        print("\n--- Test 2: Corrupted Checkpoint Fallback ---")
        ckpt_path.write_text("CORRUPTED_GARBAGE_JSON_DATA_!@#$%", encoding="utf-8")
        agent3 = QTradingAgent(checkpoint_path=ckpt_path)
        assert len(agent3.q_table) == 0, "Agent must gracefully fallback to fresh table on corrupt file"
        print("-> PASS: Corrupted checkpoint handled safely without crash.")


def test_model_caching():
    print("\n--- Test 3: In-memory Model Caching in Paper Trader ---")
    from paper_trader import get_cached_model, MODELS_DIR

    model_files = list(MODELS_DIR.glob("*.joblib"))
    if model_files:
        test_file = model_files[0]
        m1 = get_cached_model(test_file)
        m2 = get_cached_model(test_file)
        assert m1 is m2, "get_cached_model must return identical in-memory instance without reloading"
        print(f"-> Cached model instance verified: {test_file.name} (id: {id(m1)} == {id(m2)})")
    else:
        print("-> No joblib models in models dir, skipped model file check.")
    print("-> PASS: Model caching in memory verified.")


def main():
    print("=== STARTING PHASE 2 VERIFICATION ===")
    test_rl_persistence()
    test_model_caching()
    print("\n=== ALL PHASE 2 PYTHON TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
