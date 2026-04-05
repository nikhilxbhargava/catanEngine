"""FastAPI server for the Catan visualization frontend.

Serves:
  - Static HTML/JS frontend
  - REST API for replays, training runs, and eval results
  - Endpoint to generate replays on-demand from trained models

Usage:
  python -m web.server [--port 8000]
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Catan Engine Dashboard")

PROJECT_ROOT = Path(__file__).parent.parent
REPLAY_DIR = PROJECT_ROOT / "replays"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WEB_DIR = Path(__file__).parent


@app.get("/")
async def index():
    """Serve the main HTML page."""
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# ── Replay API ──────────────────────────────────────────────────────


@app.get("/api/replays")
async def list_replays():
    """List all saved replays."""
    if not REPLAY_DIR.exists():
        return {"replays": []}

    replays = []
    for path in sorted(REPLAY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            replays.append({
                "filename": path.name,
                "winner": meta.get("winner"),
                "num_turns": meta.get("num_turns", 0),
                "num_actions": meta.get("num_actions", 0),
                "seed": meta.get("seed"),
                "timestamp": meta.get("timestamp", 0),
                "agent_type": meta.get("agent_type", "unknown"),
                "model_path": meta.get("model_path", ""),
            })
        except Exception:
            continue

    return {"replays": replays}


@app.get("/api/replays/{filename}")
async def get_replay(filename: str):
    """Get a full replay by filename."""
    path = REPLAY_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Replay not found")
    with open(path) as f:
        return json.load(f)


@app.post("/api/replays/generate")
async def generate_replay(
    model_path: str = "checkpoints/a2c_v2.pt",
    agent_type: str = "a2c",
    seed: int = 42,
    num_games: int = 1,
):
    """Generate replay(s) by running a trained model against random opponents."""
    import random as _random

    full_model_path = PROJECT_ROOT / model_path
    if not full_model_path.exists():
        raise HTTPException(404, f"Model not found: {model_path}")

    REPLAY_DIR.mkdir(exist_ok=True)

    from agents.random_agent import RandomAgent
    from catan.board import Board
    from catan.game import Game
    from catan.replay import record_game, save_replay

    if agent_type == "a2c":
        from agents.a2c_agent import A2CAgent
        agent = A2CAgent(hidden_dim=256, device="cpu")
    else:
        from agents.reinforce_agent import ReinforceAgent
        agent = ReinforceAgent(hidden_dim=256, device="cpu")

    agent.load(str(full_model_path))
    agent.training = False

    generated = []
    for i in range(num_games):
        game_seed = seed + i
        board = Board.build(seed=game_seed)
        game = Game(board, num_players=4, seed=game_seed + 100)

        opponents = [RandomAgent(seed=game_seed + j + 1) for j in range(3)]
        player_index = 0

        while not game.is_over() and game.state.turn_number < 3000:
            actions = game.get_legal_actions()
            if not actions:
                break
            pidx = game.state.current_player_index
            if pidx == player_index:
                action = agent.choose_action(game.state, actions)
            else:
                opp_idx = pidx - 1 if pidx > player_index else pidx
                action = opponents[opp_idx].choose_action(game.state, actions)
            game.apply(action)

        replay = record_game(
            board=board,
            action_log=game.action_log,
            seed=game_seed + 100,
            num_players=4,
            metadata={
                "agent_type": agent_type,
                "model_path": model_path,
                "agent_player": player_index,
                "seed": game_seed,
            },
        )

        filename = f"game_{agent_type}_{game_seed}_{int(time.time())}.json"
        save_replay(replay, str(REPLAY_DIR / filename))
        generated.append({
            "filename": filename,
            "winner": replay["metadata"]["winner"],
            "num_turns": replay["metadata"]["num_turns"],
        })

    return {"generated": generated}


# ── Models API ──────────────────────────────────────────────────────


@app.get("/api/models")
async def list_models():
    """List all saved model checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return {"models": []}

    models = []
    for path in sorted(CHECKPOINT_DIR.glob("*.pt"), key=os.path.getmtime, reverse=True):
        stat = path.stat()
        models.append({
            "filename": path.name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": stat.st_mtime,
        })
    return {"models": models}


# ── Eval API (run quick eval on a model) ────────────────────────────


@app.post("/api/eval")
async def run_eval(
    model_path: str = "checkpoints/a2c_v2.pt",
    agent_type: str = "a2c",
    num_games: int = 20,
    seed: int = 99999,
):
    """Run a quick eval of a model against random opponents."""
    full_model_path = PROJECT_ROOT / model_path
    if not full_model_path.exists():
        raise HTTPException(404, f"Model not found: {model_path}")

    from agents.random_agent import RandomAgent
    from catan.board import Board
    from catan.game import Game

    if agent_type == "a2c":
        from agents.a2c_agent import A2CAgent
        agent = A2CAgent(hidden_dim=256, device="cpu")
    else:
        from agents.reinforce_agent import ReinforceAgent
        agent = ReinforceAgent(hidden_dim=256, device="cpu")

    agent.load(str(full_model_path))
    agent.training = False

    wins = 0
    results = []
    start = time.time()

    for i in range(num_games):
        game_seed = seed + i
        board = Board.build(seed=game_seed)
        game = Game(board, num_players=4, seed=game_seed + 100)

        opponents = [RandomAgent(seed=game_seed + j + 1) for j in range(3)]
        player_index = i % 4

        while not game.is_over() and game.state.turn_number < 3000:
            actions = game.get_legal_actions()
            if not actions:
                break
            pidx = game.state.current_player_index
            if pidx == player_index:
                action = agent.choose_action(game.state, actions)
            else:
                opp_idx = pidx - 1 if pidx > player_index else pidx
                action = opponents[opp_idx].choose_action(game.state, actions)
            game.apply(action)

        won = game.winner() == player_index
        if won:
            wins += 1
        results.append({
            "game": i,
            "seed": game_seed,
            "player_index": player_index,
            "won": won,
            "winner": game.winner(),
            "turns": game.state.turn_number,
        })

    elapsed = time.time() - start
    return {
        "model": model_path,
        "num_games": num_games,
        "wins": wins,
        "win_rate": round(wins / num_games, 3),
        "elapsed_sec": round(elapsed, 1),
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
