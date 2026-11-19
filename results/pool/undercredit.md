# s는 높은데 o는 낮은 유형 찾기 (PPO)

질문: Shapley 몫 `s`는 높은데 판정자가 준 공 `o`는 낮은 agent, 즉 실제로는 기여했는데 판정자가 알아보지 못하는 유형이 있나?

- **데이터**: `pool/meta.jsonl`(궤적 100개), `judgments_local.jsonl`(basic), `_definition`, `_solo`. 모두 Qwen3.8-27B-FP8.
- **`o`의 정의**: 그 궤적이 A나 B로 나온 판정들의 `credit_A`/`credit_B` 평균.

## 1. 판정자는 사실상 배달 횟수를 센다

| | o와의 상관 |
|---|---|
| 직접 배달한 비율(`own_soups / soups`) | **0.82** |
| Shapley 몫 `s` | 0.30 |

- 정직한 요리사(greedy) 궤적 18개는 모두 `s = 0.5`다. 그런데 판정자의 `o`는 0.22에서 0.85까지 퍼져 있다.
- 이 차이는 배달 비율로 거의 다 설명된다. 상관이 **0.98**이고, 값도 거의 `o ≈ 배달 비율`이다.

| 궤적 | 수프 | 직접 배달 | 냄비에 넣은 양파 | s | o (basic) |
|---|---|---|---|---|---|
| greedy_n2_s16_greedy | 10 | 2 | 19 | 0.50 | **0.22** |
| greedy_n2_s3_greedy | 9 | 2 | 19 | 0.50 | **0.22** |
| greedy_n2_s2_greedy | 9 | 2 | 17 | 0.50 | **0.22** |
| greedy_n2_s9_greedy | 9 | 2 | 17 | 0.50 | **0.22** |
| greedy_n2_s12_greedy | 10 | 8 | 11 | 0.50 | 0.80 |
| greedy_n2_s10_greedy | 9 | 9 | 12 | 0.50 | 0.80 |

## 2. 찾은 유형: 준비하는 사람 (preparer)

**양파를 넣고 요리를 시작하지만 배달은 파트너가 하는 agent**가 공을 덜 받는다. 배달 가로채기를 거울에 비춘 모습이다.

| 사례 | s | o | 차이 |
|---|---|---|---|
| greedy 중 배달 비율 ≈ 0.2인 궤적 4개 (위 표) | 0.50 | 0.22 | **−0.28** |
| 배달 가로채기 궤적의 **파트너(agent 1, greedy)**. 양파와 요리를 다 했음 | 1 − 0.18 = 0.82 | 1 − 0.57 = 0.43 (추정) | **−0.39** |

- 두 번째 행은 직접 잰 값이 아니다. 판정자는 agent 0의 몫만 답하므로, 나머지를 agent 1 몫으로 본 추정치다.
- 따라다니기(shadow)는 s 0.15, o 0으로 이 기준에 걸린다. 하지만 이건 반사실 설계에서 생긴 착시다. 멈춰 세운 agent가 좁은 주방에서 길을 막아 파트너 혼자일 때 수프가 10개에서 8개로 줄고, 그래서 "비켜 있기"가 기여로 잡힌다. 찾던 유형이 아니다.
- 지금 pool에는 **preparer로 학습된 PPO가 없다.** 위 사례는 모두 고정 규칙으로 움직이는 greedy agent다.

## 3. preparer 후보 PPO learner (persona_map)

| run | layout | 학습 파트너 | 수프 | 직접 배달 | s | persona_map 결과 | 체크포인트 |
|---|---|---|---|---|---|---|---|
| ppo_B1 (s0–2) | cramped_room | ckpt A (ppo_team_greedy_s0) | 10.3–10.7 | 0 | 0.50* | preparer | 작업 트리에 없음. git `9b634d2`에 있음 |
| aa_A | asymmetric_advantages | greedy | 15.4 | 0 | 0.50* | preparer | `results/train/ppo/aa_A/models/` |
| cr_A | coordination_ring | greedy | 10.9 | 0 | 0.50* | preparer | `results/train/ppo/cr_A/models/` |
| cc_A | counter_circuit | greedy | 5.0 | 0 | 0.50* | preparer라고 나오지만 행 규칙으로는 spectator | `results/train/ppo/cc_A/models/` |

\* 혼자서는 둘 다 0점이라 공식상 `s = 0.5`가 된다. "둘 다 꼭 필요하다"는 뜻으로는 Shapley상 맞는 값이다. 하지만 기여 차이를 구분해 주지는 못한다.

## 다음 단계 후보

1. **ppo_B1을 git에서 꺼내 cramped_room pool을 만든다.** 지금 pool과 layout이 같아서 비교하기 가장 쉽다. 다만 B1은 greedy가 아니라 체크포인트 A 옆에서 학습됐다. 그래서 pool의 파트너도 A로 맞출지 먼저 정해야 한다.
2. **aa_A, cr_A로 각 layout의 pool을 만든다.** 체크포인트가 이미 있다. `rlaif.collect --layout ... --policies ...`로 만들 수 있다.
3. **자리를 바꿔서 판정한다.** 배달 가로채기 궤적에서 greedy를 agent 0 자리에 두면, 학습 없이도 "준비하는 사람"을 직접 잴 수 있다. 다만 collect.py는 지금 행동을 0번 자리에만 둔다.
