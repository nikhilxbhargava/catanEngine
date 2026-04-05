"""Self-play A2C training for Catan.

Instead of training against random opponents, the agent plays against
past versions of itself. This creates an ever-improving curriculum:
as the agent gets better, its opponents get better too.

Architecture:
  - Maintain a pool of past checkpoints (opponent_pool)
  - Each game: agent plays as one seat, opponents are sampled from the pool
  - Every N batches, add current weights to the pool
  - Periodically evaluate against both random and pool opponents

Usage:
  python -m training.train_selfplay --batches 500 --batch-size 32 --workers 6
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from multiprocessing import Pool, cpu_count

import torch

from agents.random_agent import RandomAgent
from catan.board import Board
from catan.game import Game


def _play_selfplay_game(args: tuple):
    """Play one self-play game. Agent is player_index, opponents use opp_weights."""
    agent_weights, opp_weights_list, game_seed, hidden_dim, player_index = args

    from agents.a2c_agent import A2CAgent

    # Create agent
    agent = A2CAgent(hidden_dim=hidden_dim, device="cpu")
    agent.set_weights(agent_weights)
    agent.training = True
    agent.start_episode()

    # Create opponents from pool weights
    opponents = []
    for ow in opp_weights_list:
        opp = A2CAgent(hidden_dim=hidden_dim, device="cpu")
        opp.set_weights(ow)
        opp.training = False
        opponents.append(opp)

    board = Board.build(seed=game_seed)
    game = Game(board, num_players=4, seed=game_seed + 100)

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

    reward = 1.0 if game.winner() == player_index else (-1.0 if game.winner() is not None else 0.0)
    return agent.finish_episode(reward)


def _eval_vs_random(args: tuple) -> int:
    """Evaluate one game against random opponents."""
    weights, game_seed, hidden_dim, player_index = args

    from agents.a2c_agent import A2CAgent

    agent = A2CAgent(hidden_dim=hidden_dim, device="cpu")
    agent.set_weights(weights)
    agent.training = False

    opponents = [RandomAgent(seed=game_seed + i + 1) for i in range(3)]

    board = Board.build(seed=game_seed)
    game = Game(board, num_players=4, seed=game_seed + 100)

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

    return 1 if game.winner() == player_index else 0


def train(
    num_batches: int = 500,
    batch_size: int = 32,
    eval_every: int = 25,
    eval_games: int = 40,
    pool_update_every: int = 10,
    pool_max_size: int = 10,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    num_workers: int = 0,
    device: str = "cpu",
    save_path: str = "checkpoints/a2c_selfplay.pt",
    seed: int = 0,
    resume: bool = False,
):
    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    from agents.a2c_agent import A2CAgent
    agent = A2CAgent(hidden_dim=hidden_dim, lr=lr, gamma=gamma, device=device)

    # Resume from checkpoint
    start_batch = 1
    best_win_rate = 0.0
    total_games = 0

    if resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}...")
        metadata = agent.load(save_path)
        if metadata:
            start_batch = metadata.get("batch", 0) + 1
            best_win_rate = metadata.get("best_win_rate", 0.0)
            total_games = metadata.get("total_games", 0)
            print(f"  Resumed at batch {start_batch}, best_win_rate={best_win_rate:.0%}")

    # Start opponent pool with current weights
    opponent_pool: deque[dict] = deque(maxlen=pool_max_size)
    opponent_pool.append(agent.get_weights())

    print(f"\nSelf-play training config:")
    print(f"  Batches:       {start_batch}..{num_batches} x {batch_size} games")
    print(f"  Workers:       {num_workers}")
    print(f"  Device:        {device}")
    print(f"  Pool update:   every {pool_update_every} batches (max {pool_max_size})")
    print()

    start = time.time()

    pool = Pool(processes=num_workers)

    try:
        for batch in range(start_batch, num_batches + 1):
            batch_start = time.time()

            if hasattr(agent, "adjust_lr"):
                agent.adjust_lr(batch / num_batches)

            weights = agent.get_weights()

            # Sample 3 opponent weight sets from the pool for each game
            import random as _random
            rng = _random.Random(seed + batch)

            work_items = []
            for i in range(batch_size):
                opp_weights = [rng.choice(opponent_pool) for _ in range(3)]
                work_items.append((
                    weights, opp_weights,
                    seed + total_games + i, hidden_dim, i % 4,
                ))

            episodes = pool.map(_play_selfplay_game, work_items)
            total_games += batch_size

            result = agent.update_from_episodes(episodes)

            batch_wins = sum(1 for ep in episodes if ep.reward > 0)
            batch_time = time.time() - batch_start
            games_per_sec = batch_size / batch_time
            elapsed = time.time() - start

            print(
                f"Batch {batch:4d}/{num_batches} | "
                f"Wins: {batch_wins:2d}/{batch_size} ({batch_wins/batch_size:4.0%}) | "
                f"P: {result['policy_loss']:7.4f} "
                f"V: {result['value_loss']:6.4f} "
                f"H: {result['entropy']:5.2f} | "
                f"{games_per_sec:5.1f} g/s | "
                f"Pool: {len(opponent_pool)} | "
                f"{elapsed:6.1f}s"
            )

            # Add current weights to opponent pool
            if batch % pool_update_every == 0:
                opponent_pool.append(agent.get_weights())

            # Evaluate vs random
            if batch % eval_every == 0:
                eval_start = time.time()
                eval_items = [
                    (weights, seed + 999999 + i, hidden_dim, i % 4)
                    for i in range(eval_games)
                ]
                eval_results = pool.map(_eval_vs_random, eval_items)
                win_rate = sum(eval_results) / eval_games
                eval_time = time.time() - eval_start

                marker = ""
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    marker = " ** BEST **"

                agent.save(save_path, metadata={
                    "batch": batch,
                    "best_win_rate": best_win_rate,
                    "total_games": total_games,
                    "win_rate": win_rate,
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "seed": seed,
                })

                print(
                    f"  >> Eval vs Random: {sum(eval_results)}/{eval_games} "
                    f"({win_rate:.0%}) | Best: {best_win_rate:.0%} | "
                    f"Saved | {eval_time:.1f}s{marker}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        agent.save(save_path, metadata={
            "batch": batch,
            "best_win_rate": best_win_rate,
            "total_games": total_games,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "seed": seed,
        })
        print(f"Saved to {save_path}")
    finally:
        pool.terminate()
        pool.join()

    print(f"\nDone. {total_games} games in {time.time() - start:.1f}s")
    print(f"Best eval win rate vs random: {best_win_rate:.0%}")
    print(f"Model: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play A2C training")
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--pool-update", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default="checkpoints/a2c_selfplay.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Auto-detect CUDA
    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda"
        print(f"Auto-detected CUDA GPU!")

    train(
        num_batches=args.batches,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        pool_update_every=args.pool_update,
        pool_max_size=args.pool_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        num_workers=args.workers,
        device=args.device,
        save_path=args.save_path,
        seed=args.seed,
        resume=args.resume,
    )
