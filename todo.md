# TODO: cramped_room에서 A와 A 옆 페르소나 체크포인트 만들기

모든 실행: `cramped_room`, 2M step, seed 0–2. 학습 하나에 20–40분.
Mac은 PPO, Windows는 GRPO. 두 컴퓨터가 같은 폴더에 쓰지 않으니 끝나면 폴더를 한쪽으로 모은다.

- 체크포인트 만들기: `python -m train.run ...` → `models/*.pt`, `curves/`, `config.json`, `train_log.csv`
- 학습이 잘 됐는지 확인: `python -m eval.run <폴더>` → `results.csv`(팀 수프, 자기 배달, `s`, credit gap), `reference.csv`
  (`train.run`에 `--eval`을 붙이면 학습 직후 바로 확인까지 한다.)

## 0. 돌리기 전에

- [ ] Windows에도 최신 코드가 있는지: `train/run.py`(새 파일), `eval/run.py`, `environment/kitchen.py`, `train/`.
      커밋해서 옮기거나 복사.

## 1단계. A 만들기 (greedy 옆, team 보상)

Mac (PPO)
- [O] 학습 끝남 (`results/ppo/A`): seed 0–2 모두 팀 수프 11.0개를 전부 직접 배달, `s`=0.18. 예전 PPO-A와 같다(배달 독점형).

Windows (GRPO)
- [ ] 학습: `.\venv\Scripts\python -m train.run --algos grpo --rewards team --partners greedy --seeds 0 1 2 --layout cramped_room --out results/grpo/A --workers 3`
- [ ] 확인: `.\venv\Scripts\python -m eval.run results/grpo/A --workers 3`
      `results/grpo/A/results.csv`에서 seed 0의 팀 수프. 예전 GRPO-A는 8.3개(greedy가 전부 만듦, 학습자 배달 0).

동료로 쓸 체크포인트 (seed 0):
- PPO-A: `results/ppo/A/models/ppo_team_greedy_s0.pt`
- GRPO-A: `results/grpo/A/models/grpo_team_greedy_s0.pt`

## 2단계 B. A 옆 페르소나 (1단계 A를 확인한 뒤)

team = 준비 담당(대조군), individual = 포기, pickup = 바쁜 척, shadow = 그림자.

Mac (PPO) — PPO-A가 준비됐으니 바로 돌려도 된다
- [ ] 학습: `venv/bin/python -m train.run --algos ppo --rewards team individual pickup shadow --partners ckpt:results/ppo/A/models/ppo_team_greedy_s0.pt --seeds 0 1 2 --layout cramped_room --out results/ppo/A_personas --workers 12`
- [ ] 확인: `venv/bin/python -m eval.run results/ppo/A_personas --workers 12`

Windows (GRPO) — GRPO-A가 끝난 뒤
- [ ] 학습: `.\venv\Scripts\python -m train.run --algos grpo --rewards team individual pickup shadow --partners ckpt:results/grpo/A/models/grpo_team_greedy_s0.pt --seeds 0 1 2 --layout cramped_room --out results/grpo/A_personas --workers 12`
- [ ] 확인: `.\venv\Scripts\python -m eval.run results/grpo/A_personas --workers 12`

## 실행 수

| 단계 | PPO (Mac) | GRPO (Windows) |
|---|---|---|
| 1단계 A | 3 (끝남) | 3 |
| 2단계 B (A 옆) | 12 | 12 |

2단계 B 학습은 12개를 동시에 돌리면 1–1.5시간. 확인(eval.run)은 체크포인트당 몇 분.

## 끝난 뒤 생기는 체크포인트

`results/<algo>/A_personas/models/` 아래, seed마다 하나씩:
- `<algo>_team_ckpt-<algo>_team_greedy_s0_s{0,1,2}.pt` (준비 담당)
- `<algo>_individual_ckpt-<algo>_team_greedy_s0_s{0,1,2}.pt` (포기)
- `<algo>_pickup_ckpt-<algo>_team_greedy_s0_s{0,1,2}.pt` (바쁜 척)
- `<algo>_shadow_ckpt-<algo>_team_greedy_s0_s{0,1,2}.pt` (그림자)
