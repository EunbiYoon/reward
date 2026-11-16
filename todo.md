# TODO: cramped_room에서 A부터 페르소나까지 처음부터 다시 만들기

모든 실행: `cramped_room`, 2M step, seed 0–2. 실행 하나에 20–40분.
Mac은 PPO, Windows는 GRPO를 맡으면 두 컴퓨터가 같은 폴더에 쓰지 않는다. 끝나면 폴더를 한쪽으로 모은다.

## 0. 돌리기 전에

- [ ] Windows에도 최신 코드가 있는지(`eval/run.py`, `environment/kitchen.py`, `train/ppo.py`). 커밋해서 옮기거나 복사.
      `run.py`의 결과 합치기와 self-play 동료(`self`)가 이 파일들에 있다.

## 1단계. A 만들기 (greedy 옆, team 보상)

Mac (PPO)
- [O] `venv/bin/python -m eval.run --algos ppo --rewards team --partners greedy --seeds 0 1 2 --layout cramped_room --out results/ppo/A --workers 3`

Windows (GRPO)
- [O] `.\venv\Scripts\python -m eval.run --algos grpo --rewards team --partners greedy --seeds 0 1 2 --layout cramped_room --out results/grpo/A --workers 3`

결과 확인 (2단계 B 전에 꼭):
- [ ] `results/<algo>/A/results.csv`에서 seed 0의 팀 수프(soups)가 0에 가깝지 않은지. 예전 PPO-A는 11.0개(전부 직접 배달),
      GRPO-A는 8.3개(greedy가 전부 만듦). 요리를 못 배웠으면 B를 돌려도 의미가 없다.
- 동료로 쓸 체크포인트: `results/ppo/A/models/ppo_team_greedy_s0.pt`, `results/grpo/A/models/grpo_team_greedy_s0.pt`

## 2단계 A. greedy 옆 페르소나 (A가 필요 없어서 1단계와 동시에 돌려도 됨)

individual = 배달 뺏기, pickup = 바쁜 척, shadow = 그림자. (team은 1단계 A와 같은 조건이라 뺐다.)

Mac (PPO)
- [ ] `venv/bin/python -m eval.run --algos ppo --rewards individual pickup shadow --partners greedy --seeds 0 1 2 --layout cramped_room --out results/ppo/greedy_personas --workers 9`

Windows (GRPO)
- [ ] `.\venv\Scripts\python -m eval.run --algos grpo --rewards individual pickup shadow --partners greedy --seeds 0 1 2 --layout cramped_room --out results/grpo/greedy_personas --workers 9`

## 2단계 B. A 옆 페르소나 (1단계가 끝나고 A를 확인한 뒤)

team = 준비 담당(대조군), individual = 포기, pickup = 바쁜 척, shadow = 그림자.

Mac (PPO)
- [ ] `venv/bin/python -m eval.run --algos ppo --rewards team individual pickup shadow --partners ckpt:results/ppo/A/models/ppo_team_greedy_s0.pt --seeds 0 1 2 --layout cramped_room --out results/ppo/A_personas --workers 12`

Windows (GRPO)
- [ ] `.\venv\Scripts\python -m eval.run --algos grpo --rewards team individual pickup shadow --partners ckpt:results/grpo/A/models/grpo_team_greedy_s0.pt --seeds 0 1 2 --layout cramped_room --out results/grpo/A_personas --workers 12`

## (선택) self-play A 옆 페르소나, PPO만

다른 논문(Carroll et al. 2019, FCP) 방식의 동료. self-play는 PPO에만 연결되어 있다.
- [ ] `venv/bin/python -m eval.run --algos ppo --rewards team --partners self --seeds 0 1 2 --layout cramped_room --out results/ppo/sp_A --workers 3`
- [ ] (sp_A의 팀 수프 확인 후) `venv/bin/python -m eval.run --algos ppo --rewards team individual pickup shadow --partners ckpt:results/ppo/sp_A/models/ppo_team_self_s0.pt --seeds 0 1 2 --layout cramped_room --out results/ppo/sp_personas --workers 12`

## 실행 수와 시간

| 단계 | PPO (Mac) | GRPO (Windows) |
|---|---|---|
| 1단계 A | 3 | 3 |
| 2단계 A (greedy 옆) | 9 | 9 |
| 2단계 B (A 옆) | 12 | 12 |
| (선택) self-play | 3 + 12 | – |

Mac은 1단계와 2단계 A를 동시에(12개), 끝나면 2단계 B(12개). 대략 1–1.5시간씩 두 번.

## 끝난 뒤

- [ ] hacker 후보와 군집: `venv/bin/python scripts/find_hackers.py results/ppo/A results/ppo/greedy_personas results/ppo/A_personas results/grpo/A results/grpo/greedy_personas results/grpo/A_personas`
- [ ] `scripts/persona_steps.py`의 체크포인트 목록을 새 폴더 기준으로 바꾸고 그림 1을 seed 3개 평균으로 다시 만들기
- [ ] 논문 표 2(pool 표)와 본문 숫자를 새 결과로 바꾸기
- [ ] 판정자 실험 pool을 새 체크포인트로 다시 모을지 결정 (지금 pool은 예전 `full` 체크포인트로 녹화됨)
