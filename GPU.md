# LLM judge on Unity (GPU)

Pool 두 개(`need_curve_ppo`, `need_curve_grpo`) × 프롬프트 세 개(basic, definition, solo)를 판정한다. 판정마다 **100쌍까지만** 본다.

- 작업 하나는 L40S GPU 2장에서 vLLM으로 모델을 띄우고 64쌍씩 동시에 판정한다.
- 100쌍이면 모델 로딩을 포함해 작업 하나에 15–20분 걸린다.
- 52분이 지나면 남은 쌍을 다음 작업으로 넘기고 스스로 다시 줄을 선다(requeue).

| `PROMPT` | 판정자가 보는 것 | 쌍 | 결과 파일 |
|---|---|---|---|
| `basic` | 게임 규칙 + 사건 기록 | 모든 쌍 중 앞 100 | `judgments_local.jsonl` |
| `definition` | + Shapley 값의 정의 | 도둑이 들어간 쌍(`FOCUS=individual`) 중 앞 100 | `judgments_local_definition.jsonl` |
| `solo` | + 정의 + 반사실 숫자 v(P), v(L) | 도둑이 들어간 쌍 중 앞 100 | `judgments_local_solo.jsonl` |

- **실제 쌍 수:** PPO pool의 도둑 쌍은 모두 80개라서 PPO의 definition과 solo는 80쌍이다.
- **같은 쌍을 판정한다:** 쌍 순서는 어느 컴퓨터에서나 같다(`--seed 0`). 그래서 basic 100쌍은 이 PC에서 Ollama Qwen이 판정한 앞 100쌍과 같은 쌍이다.

## 0. 이 PC에서: pool을 git으로 올리기

```
git add rlaif scripts/cluster/judge_gpu.sh environment/kitchen.py GPU.md README.md results/rlaif/need_curve_ppo results/rlaif/need_curve_grpo
git commit -m "judge pools and unity judge script"
git push
```

## 1. Unity에서: 받기 (로그인 노드)

`~/reward`는 Unity에 clone해 둔 프로젝트 폴더로 바꾼다.

```
cd ~/reward
git pull
source scripts/cluster/activate.sh
mkdir -p logs
```

처음이면 vLLM 환경과 모델을 먼저 준비한다(한 번만): `bash scripts/cluster/setup_conda.sh vllm`

## 2. 판정 작업 6개 제출

```
sbatch --job-name=judge_ppo_basic --export=ALL,RUN=results/rlaif/need_curve_ppo,PROMPT=basic,PAIRS=100 scripts/cluster/judge_gpu.sh
sbatch --job-name=judge_ppo_definition --export=ALL,RUN=results/rlaif/need_curve_ppo,PROMPT=definition,FOCUS=individual,PAIRS=100 scripts/cluster/judge_gpu.sh
sbatch --job-name=judge_ppo_solo --export=ALL,RUN=results/rlaif/need_curve_ppo,PROMPT=solo,FOCUS=individual,PAIRS=100 scripts/cluster/judge_gpu.sh
sbatch --job-name=judge_grpo_basic --export=ALL,RUN=results/rlaif/need_curve_grpo,PROMPT=basic,PAIRS=100 scripts/cluster/judge_gpu.sh
sbatch --job-name=judge_grpo_definition --export=ALL,RUN=results/rlaif/need_curve_grpo,PROMPT=definition,FOCUS=individual,PAIRS=100 scripts/cluster/judge_gpu.sh
sbatch --job-name=judge_grpo_solo --export=ALL,RUN=results/rlaif/need_curve_grpo,PROMPT=solo,FOCUS=individual,PAIRS=100 scripts/cluster/judge_gpu.sh
```

여섯 개가 서로 다른 GPU 노드에서 동시에 돈다. 같은 노드에 둘이 걸려도 vLLM 포트가 달라서 부딪치지 않는다.

## 3. 진행 확인

```
squeue --me
tail -f logs/judge_ppo_basic.log
wc -l results/rlaif/need_curve_*/judgments_local*.jsonl
```

로그 끝에 `0 pairs left`가 나오면 끝난 것이다. vLLM이 안 뜨면 `logs/vllm_<작업 이름>.log`를 본다.

## 4. 비교 (로그인 노드, CPU 몇 초)

```
python -m rlaif.audit --run results/rlaif/need_curve_ppo
python -m rlaif.audit --run results/rlaif/need_curve_grpo
```

`audit.md`의 판정자 표에 `local`, `local_definition`, `local_solo`가 한 줄씩 나온다. 이 PC에서 돌린 `ollama`(basic)도 git으로 받으면 같은 표에 나온다.

## 5. 결과를 가져오기

```
git add results/rlaif/need_curve_ppo results/rlaif/need_curve_grpo
git commit -m "unity judge results"
git push
```

이 PC에서는 `git pull`.

## 모델

- **Unity의 기본 모델은 `Qwen/Qwen3-32B`다.** 이 PC의 Ollama 모델(`qwen3.8:27b`)과 다른 모델이라서, 결과 파일 이름이 `local`과 `ollama`로 나뉜다.
- **판정자 두 개를 비교하는 데는 이대로 써도 된다.** 32B와 27B 두 모델로 결과가 같은지 볼 수 있다.
- **같은 모델로 맞추려면** `--export`에 `MODEL=<Hugging Face 모델 이름>`을 더한다. 처음 한 번은 `MODEL=<같은 이름> bash scripts/cluster/setup_conda.sh vllm`으로 모델을 받는다.

## 설정 한눈에 (`scripts/cluster/judge_gpu.sh`)

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `RUN` | `results/rlaif/theft` | pool 폴더 |
| `PROMPT` | `basic` | `basic`, `definition`, `solo` |
| `FOCUS` | (전부) | 이 글자가 behaviour에 들어간 쌍만. 도둑은 `individual` |
| `PAIRS` | `0` (전부) | 앞에서부터 이 개수까지 |
| `MODEL` | `Qwen/Qwen3-32B` | vLLM이 띄우는 모델 |
| `PORT` | 8000 + 작업 번호 % 1000 | vLLM 포트 |
