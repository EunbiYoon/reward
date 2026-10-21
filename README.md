# Who Gets the Credit? Reward Design and Bad Teammates in Cooperative RL

제안서(CS 690S Final Project Proposal)를 그대로 구현한 코드입니다. Overcooked-AI의 `cramped_room`에서
요리사 0(학습자)을 PPO와 GRPO로 학습시키고, 요리사 1은 고정 파트너입니다. 학습자가 받는 보상 3종이
각각 어떤 나쁜 팀원을 만드는지를 **Shapley 공로**로 판정합니다.

| 학습 보상 (`--rewards`) | 정의 | 가설 |
|---|---|---|
| `pickup` | 팀 배달 보상 + 학습자가 양파나 접시를 집을 때마다 +3 (계속 유지) | H1 fake help |
| `team` | 팀 배달 보상 + 공식 shaping (학습 전반부 동안 0으로 줄어듦) | H2 spectator |
| `individual` | **학습자 자신의** 배달 보상 + 공식 shaping (줄어듦) | H3 narcissist |

| 파트너 (`--partners`) | 의미 |
|---|---|
| `stay` | 움직이지 않음. 학습자가 혼자 요리해야 함 (sanity check용) |
| `greedy` | 패키지의 스크립트 요리사 GreedyHumanModel (유능한 동료) |

모든 run은 학습 보상과 상관없이 **팀이 배달한 수프 수**로 채점합니다.

## 설치와 실행

```bash
pip install -r requirements.txt

# 1) 설치 확인 (학습 없음): 스크립트 요리사 두 명이 요리하는 영상
python -m eval.visualize --preview --partner greedy

# 2) 스모크 테스트: seed 1개, 20만 스텝
python -m eval.run --quick --workers 4 --out results/quick
python -m eval.summarize results/quick

# 3) 제안서의 전체 실험: 3 seeds x 3 rewards x 2 partners x 2 optimizers = 36 runs, 각 200만 스텝
python -m eval.run --workers 4 --out results/full
python -m eval.summarize results/full

# 일부만
python -m eval.run --algos ppo --rewards team individual --partners greedy --seeds 0 1 2 --out results/h2h3
```

`results.csv`는 run 하나가 끝날 때마다 갱신되므로 실행 도중에도 `summarize`를 돌릴 수 있습니다.

## 알고리즘 5개 × 에이전트 2, 3, 4명

`--algos all`은 학습 알고리즘 5개(`ppo grpo a2c dqn sac`)를 전부 돌리고, `--agents`는 에이전트 수를 정합니다.
3명 이상은 `warehouse`와 `foraging`에서만 됩니다 (`kitchen`은 2명 전용).

```bash
# 1) 스모크 테스트: seed 1개, 20만 스텝 (3 rewards x 2 partners x 5 algos x 3 agents = 90 runs)
python -m eval.run --env foraging --agents 2 3 4 --algos all --quick --workers 4 --out results/lbf_n_quick
python -m eval.summarize results/lbf_n_quick

# 2) 전체 실험: 3 seeds, 각 200만 스텝 (270 runs)
python -m eval.run --env foraging --agents 2 3 4 --algos all --workers 4 --out results/lbf_n
python -m eval.summarize results/lbf_n

# 같은 실험을 warehouse에서
python -m eval.run --env warehouse --agents 2 3 4 --algos all --workers 4 --out results/rware_n
python -m eval.summarize results/rware_n

# 4명은 8x8이 붐비므로 더 큰 맵 권장
python -m eval.run --env foraging --layout 10x10-4f --agents 4 --algos all --workers 4 --out results/lbf_n4

# 일부만: 알고리즘, 에이전트 수, 보상을 골라서
python -m eval.run --env foraging --agents 3 --algos ppo sac --rewards team individual --out results/lbf_n3

# kitchen(Overcooked)에서 알고리즘 5개 전부 (2명)
python -m eval.run --algos all --workers 4 --out results/all_algos

# 영상: --agents를 학습 때와 같게
python -m eval.visualize --run results/lbf_n --algo sac --reward team --partner greedy --agents 3
```

- 0번이 학습자이고, 나머지 1..N−1번은 모두 `--partners`로 고른 같은 종류의 파트너입니다.
- Shapley는 N명 공식(파트너의 모든 부분집합)으로 계산하므로 N=4면 평가 에피소드마다 연합 15개를 돌립니다.
- 관측은 "자기 층 + 나머지 전원을 합친 층"이라 N과 상관없이 크기가 같습니다 (N=2일 때는 예전과 동일).
- `summarize`는 에이전트 수별로 따로, 같은 N의 기준값과 비교해 판정합니다. 3명 이상에서 `partner soups`는 파트너 전원의 합입니다.
- greedy끼리의 기준값 s\*는 대략 1/N입니다 (warehouse `tiny`: 0.50 / 0.34 / 0.26).
- 체크포인트 이름은 2명이면 예전 그대로, 3·4명이면 `ppo_team_greedy_n3_s0.pt`처럼 `_n3`이 붙습니다.
- run 개수 = seeds × rewards × partners × algos × agents. 오래 걸리니 항상 `--quick`부터 돌리세요.

## 영상

```bash
python -m eval.visualize --run results/full --algo ppo --reward individual --partner greedy --seed 0
python -m eval.visualize --run results/full --algo ppo --partner greedy --seed 0 --compare   # 보상 3종 나란히
python -m eval.visualize --run results/full --algo ppo --reward team --freeze partner        # Shapley 평가 장면
```

파란 0번이 학습자, 초록 1번이 파트너입니다. 화면 위에 팀 수프, 각자 배달한 수프, 학습자의 줍기 횟수가 표시됩니다.
pygame이 없으면 `--renderer simple`(matplotlib 그림)로 그립니다. 파일 이름이 `.mp4`로 끝나면 MP4로 저장합니다(ffmpeg 필요).

## 측정 방법

**Shapley 공로.** 같은 평가 에피소드(seed 10000+k)를 세 번 돌립니다.

| 기호 | 의미 |
|---|---|
| v(L,P) | 둘 다 움직임 (= 팀 수프, 진짜 목표) |
| v(P) | 학습자를 멈춤 |
| v(L) | 파트너를 멈춤 |

φ = ½[(v(L,P) − v(P)) + v(L)],  **s = φ / v(L,P)** (학습자의 Shapley 몫),
**credit gap = 학습자가 직접 배달한 몫 − s**.

**기준값 s\*.** 학습자 자리에 스크립트 요리사를 넣고 같은 방법으로 잰 값입니다 (`reference.csv`).

**판정 기준 (제안서 5절).** `summarize`가 옵티마이저별로 자동 판정합니다.

| | 조건 (greedy 파트너) |
|---|---|
| sanity | team 보상 + stay 파트너에서 학습자가 수프를 배달함. 실패하면 H1~H3는 모두 inconclusive |
| H1 | pickup 보상에서 run의 절반 이상이 줍기 횟수 > 기준값 이면서 s ≤ 0.1 |
| H2 | team 보상에서 s < s\*/2, 그리고 individual로 바꾸면 s가 올라감 (seed별 짝비교) |
| H3 | individual 보상에서 credit gap ≥ 0.2, 그리고 파트너 수프가 team 보상 때보다 줄어듦 (짝비교) |

각 판정은 "supported (CIs clear)", "supported (means only)", "not supported" 중 하나로 나옵니다.
seed가 3개라 신뢰구간이 넓으므로, 대부분 "means only"로 나올 것을 예상해야 합니다.

## 다른 환경: RWARE, LBF (`--env`)

같은 보상 3종, 같은 파트너 2종, 같은 Shapley 평가를 다른 두 협동 환경에서도 돌릴 수 있습니다.
결과 열 이름은 그대로 `soups`이지만 의미는 환경마다 다릅니다.

| `--env` | 패키지 | 기본 `--layout` | `soups`의 의미 | horizon |
|---|---|---|---|---|
| `kitchen` (기본) | overcooked-ai | `cramped_room` | 배달한 수프 수 | 400 |
| `warehouse` | [rware](https://github.com/semitable/robotic-warehouse) | `tiny` (`small`, `medium`, `large`, `-easy`/`-hard`) | 요청된 선반 배달 수 | 500 |
| `foraging` | [lbforaging](https://github.com/semitable/lb-foraging) | `8x8-3f` (`10x10-4f`, `8x8-2f-coop` 등) | 먹은 음식 비율 (필드 전체 = 1) | 50 |

```bash
pip install rware lbforaging
python -m eval.visualize --preview --env warehouse --partner greedy     # 설치 확인
python -m eval.run --env warehouse --quick --workers 4 --out results/rware_quick
python -m eval.run --env foraging  --quick --workers 4 --out results/lbf_quick
python -m eval.summarize results/lbf_quick
```

| 보상 | warehouse | foraging |
|---|---|---|
| `pickup` (H1) | 팀 배달 + 선반을 **아무거나** 들 때마다 +0.15 (그 자리에서 들었다 놨다 하면 2스텝마다 받음) | 팀 음식 + 음식 옆에서 LOAD 할 때마다 +0.05 (성공 여부 무관) |
| `team` (H2) | 팀 배달 + shaping(줄어듦) | 팀 음식 + shaping(줄어듦) |
| `individual` (H3) | 학습자 자신의 배달 + shaping(줄어듦) | 학습자 몫의 음식 (패키지가 레벨 비율로 나눔) + shaping(줄어듦) |
| shaping | 요청된 선반 들기 +0.25, 배달한 선반 제자리 놓기 +0.25 | 가장 가까운 음식에 한 칸 다가갈 때마다 +0.01 |

- **greedy 파트너는 직접 짠 스크립트입니다.** RWARE에는 스크립트 로봇이 없고, LBF 2.0.0의 H1~H4는
  `from foraging.environment import Action` 때문에 import가 되지 않습니다. 둘 다 BFS로 길을 찾습니다.
  warehouse: 가장 가까운 요청 선반 → 가장 가까운 목표 → 가장 가까운 빈 선반 칸에 내려놓기.
  foraging: 혼자 들 수 있는 가장 가까운 음식, 없으면 두 사람 거리 합이 가장 작은 음식.
- **관측은 패키지 기본값이 아닙니다.** 둘 다 전체 격자를 층으로 펴서 줍니다(Overcooked의 lossless encoding과 같은 방식).
  RWARE 기본 관측은 주변 1칸만 보여서 요청된 선반을 찾을 수 없습니다.
- **멈춘 로봇/플레이어 주차:** warehouse는 목표에서 가장 먼 highway 칸(구석), foraging은 음식과 붙지 않은 모서리 칸.
- **LBF는 음식을 다 먹으면 에피소드가 진짜로 끝납니다.** PPO는 이 경우 bootstrap하지 않습니다(`info["terminal"]`).
- **기준값 (greedy + greedy, 평가 10~50 에피소드):** 두 환경 모두 자리가 대칭이라 s\* ≈ 0.50, credit gap ≈ 0.0입니다.
  cramped_room(gap 0.13)과 달리 H3의 제안서 기준(gap ≥ 0.2)을 그대로 쓸 수 있습니다.

  | | 팀 점수 | 혼자 (v(L)) | s\* | gap |
  |---|---|---|---|---|
  | warehouse `tiny` | 28.3 배달 | 15.5 | 0.50 | −0.01 |
  | foraging `8x8-3f` | 0.98 | 0.41 | 0.49 | 0.02 |
  | foraging `8x8-2f-coop` | 0.97 | 0.00 | 0.50 | 0.01 |

## 공식 패키지로 확인한 사실 (꼭 읽기)

샌드박스에 공식 Overcooked-AI 소스를 받아 스크립트 요리사로 직접 돌려 봤습니다 (평가 10 에피소드).

1. **멈춘 요리사를 그 자리에 두면 Shapley가 망가집니다.** cramped_room에서 요리사 0의 시작 칸은
   접시 디스펜서 옆의 **유일한** 칸입니다. 그대로 멈춰 두면 파트너가 접시를 못 집어 v(P)가 항상 0이 됩니다.
   그래서 멈춘 요리사는 냄비, 디스펜서, 서빙대 어디에도 붙어 있지 않은 칸(가운데 (2,2))으로 옮겨 둡니다.
   학습 중 `stay` 파트너도 같은 칸에 둡니다.
2. **스크립트 요리사 혼자 7.0개, 둘이면 9.1개입니다.** 냄비가 하나라서 두 번째 요리사가 더할 수 있는 몫이
   에피소드당 2개 정도뿐입니다. 좋은 팀원과 구경꾼의 차이가 이 2개 안에서 갈리므로 효과가 작게 나올 수 있습니다.
3. **좋은 스크립트 요리사도 credit gap이 0.13입니다.** 0번 자리가 접시와 서빙대에 더 가까워서 배달을 더 많이 합니다.
   제안서의 H3 기준(gap ≥ 0.2)은 좋은 팀원도 거의 통과하는 수준이라, `summarize`는 제안서 기준과 함께
   **기준값 대비** 판정(gap ≥ 기준값의 gap + 0.2)도 같이 보여 줍니다. 최종 보고서에는 후자를 쓰기를 권합니다.
4. 패키지의 `GreedyHumanModel.reset()`은 agent index와 mdp를 지웁니다. 매 에피소드 다시 설정하도록 처리했습니다.

## 폴더 구조

```
environment/warehouse.py RWARE 래퍼 + 스크립트 로봇 (--env warehouse)
environment/foraging.py  LBF 래퍼 + 스크립트 forager (--env foraging)
environment/two_seat.py  위 두 래퍼가 같이 쓰는 학습/평가 루프
environment/kitchen.py   공식 패키지 래퍼. 보상 3종, 파트너, 멈춘 요리사 주차, play()로 아무 조합이나 평가, 렌더러
train/deep_policy.py     MLP 정책(Categorical), critic, 관측 정규화, 체크포인트
train/ppo.py             PPO (clip, GAE)
train/grpo.py            critic 없는 GRPO (같은 리셋에서 8 에피소드 그룹, process advantage)
train/a2c.py             A2C (PPO와 같은 actor/critic/GAE, rollout당 clip 없는 한 번의 업데이트)
train/dqn.py             DQN (Double DQN 타깃, replay buffer, epsilon-greedy, target network)
train/sac.py             이산 행동용 SAC (twin Q, 엔트로피 온도 alpha 자동 조절)
eval/config.py           실험 설정 (제안서 값), --quick 설정, 판정 임계값
eval/coalitions.py       멈춘 요리사 평가와 Shapley 계산
eval/run.py              학습 → 체크포인트 → Shapley 평가 → results.csv, reference.csv
eval/summarize.py        표와 H1~H3 판정 → summary.md
eval/visualize.py        GIF/MP4
```

## 검증 범위

- **공식 패키지와 함께 확인한 것:** 래퍼, 스크립트 요리사, 멈춘 요리사 평가, Shapley 계산, 기준값, 실행 흐름, 요약, 영상.
  패키지 소스는 GitHub에서 받았고, 패키지의 부가 의존성(gymnasium, pygame 등)은 가짜 모듈로 대신했습니다.
- **확인하지 못한 것:** 샌드박스에 PyTorch를 설치할 수 없어서 **PPO/GRPO 학습은 실제로 돌려보지 못했습니다**
  (학습 부분만 가짜 학습기로 바꿔서 흐름을 확인). 처음에는 반드시 `--quick`으로 돌려서
  sanity가 pass인지(학습자가 혼자 수프를 배달하는지) 보세요. 실패하면 `--env-steps`를 늘리세요.
