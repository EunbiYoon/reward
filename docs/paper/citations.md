# 인용 지도 (Caught Red-Handed, Credited Anyway)

본문 문장 옆의 [n]은 아래 **References**의 번호다. bib 키가 있으면 `references.bib`에 이미 있는 항목이다.
상태: ✔ = 2026-11-19 웹에서 저자/학회 확인, bib = bib에 있지만 아직 확인 안 함(기억으로 씀), ? = 확인 필요(bib에도 없음).

## 1. 어디에 넣을지

### Abstract
- "Multi-agent LLM systems increasingly rely on an LLM judge to assess how much each member contributed" → [15][16][17][18][49]
- "LLM judges (Qwen3.8-27B, Gemma-3-27B and GPT-5-mini)" → [41][42][43]

### Introduction, 첫 문단
- "LLMs are increasingly used to evaluate the work of reinforcement learning agents and divide credit among them"
  → 일 평가/보상: [1][2][8][9][10][11][12][13], 팀 안에서 공 분배: [3][4][5][6][7], 개관: [14]
  (현재 tex에는 [1][2][3]만 들어가 있음)
- "since every agent costs compute, an unneeded member is a hidden waste" → [20][21]
- "A learning agent maximises the reward rule rather than the task it was given" → [22][23][24]
- "LLM judges are known to have position and length biases" → [18][50][16]
- "the closest work asks them to find the agent responsible for a multi-agent failure" → [19][17]

### Introduction, 둘째 문단
- "Overcooked" → [25]
- "Zero-shot coordination 연구는 숨은 보상이나 다양한 실력으로 동료 pool을 만들어" → [26][27][28][29]
- "Difference reward ... Shapley 기반 방법은 반사실 기여를 학습 신호로" → difference reward [30][34], Shapley [31][32], 가치 분해 [33]
- "multi-agent 시스템에 대한 세 가지 열린 질문" (집단이 비용만 늘리는가) → [17][20][21][49]

### Approach
- "Overcooked-AI의 cramped_room", "규칙 기반 요리사 greedy" → [25]
- "팀 보상만 받는 팀원은 일을 미룰 수 있다"(Related Work에서 옮긴 문장이 있으면) → [33]
- "PPO와 GRPO" → [36][37]
- "두 명의 Shapley 값" → [35]
- 판정 LLM: Qwen [42], Gemma [41], GPT-5-mini [43], vLLM [40]

### Appendix
- "k-means", "silhouette" → [45][46]
- "GAE advantage", "Adam" → [38][39]
- "process supervision" → [47]
- "RLAIF의 선호 데이터 형식" → [13][48]
- "GPT-4o" → [44]
- "Ollama" → 논문 없음, 각주에 URL(https://ollama.com)

## 2. References

**LLM이 RL 에이전트를 평가하거나 보상/공을 나눈다**

1. ✔ `kwon2023reward` Kwon, Xie, Bullard, Sadigh. Reward Design with Language Models. ICLR 2023. https://arxiv.org/abs/2303.00001
2. ✔ `klissarov2024motif` Klissarov, D'Oro, Sodhani, Raileanu, Bacon, Vincent, Zhang, Henaff. Motif: Intrinsic Motivation from Artificial Intelligence Feedback. ICLR 2024. https://arxiv.org/abs/2310.00166
3. ✔ `nagpal2025llmmca` Nagpal, Dong, Bouvier, Mehr. Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment. arXiv:2502.16863, 2025 (AAMAS 2025로 표시됨, 확인 필요). https://arxiv.org/abs/2502.16863
4. ✔ Lin, Shi, Guo, Tadiparthi, Chalaki, Moradi Pari, Stepputtis, Kim, Campbell, Sycara. Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning. NeurIPS 2025. https://arxiv.org/abs/2502.03723
5. ✔ Jiang, Zhang, Wei, Xu. QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning? arXiv:2504.12961, 2025. https://arxiv.org/abs/2504.12961
6. ✔ Qu, Jiang, Wang, Mao, Wang, Liu, Ji. Latent Reward: LLM-Empowered Credit Assignment in Episodic Reinforcement Learning. AAAI 2025, 39(19):20095–20103. https://arxiv.org/abs/2412.11120
7. ✔ Pignatelli, Ferret, Rocktäschel, Grefenstette, Paglieri, Coward, Toni. Assessing the Zero-Shot Capabilities of LLMs for Action Evaluation in RL (CALM). ICML 2024 AutoRL Workshop. https://arxiv.org/abs/2409.12798
8. ✔ Ma et al. Eureka: Human-Level Reward Design via Coding Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.12931
9. ✔ Xie et al. Text2Reward: Reward Shaping with Language Models for Reinforcement Learning. ICLR 2024. https://arxiv.org/abs/2309.11489
10. ✔ Du, Watkins, et al. Guiding Pretraining in Reinforcement Learning with Large Language Models. ICML 2023, PMLR 202. https://arxiv.org/abs/2302.06692
11. ✔ Rocamonde, Montesinos, Nava, Perez, Lindner. Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning. ICLR 2024. https://arxiv.org/abs/2310.12921
12. ✔ Wang, Sun, Zhang, Xian, Biyik, Held, Erickson. RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback. ICML 2024, PMLR 235. https://arxiv.org/abs/2402.03681
13. ✔ Lee et al. RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback. ICML 2024, PMLR 235. https://arxiv.org/abs/2309.00267
14. ✔ Cao, Zhao, et al. Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods. IEEE TNNLS, 2024. https://arxiv.org/abs/2404.00282

**LLM 판정자 / multi-agent LLM 평가**

15. ✔ Zhuge et al. Agent-as-a-Judge: Evaluate Agents with Agents. ICML 2025. https://arxiv.org/abs/2410.10934
16. ✔ Gu, Jiang, Shi, et al. A Survey on LLM-as-a-Judge. arXiv:2411.15594, 2024. https://arxiv.org/abs/2411.15594
17. ✔ Cemri et al. Why Do Multi-Agent LLM Systems Fail? NeurIPS 2025. https://arxiv.org/abs/2503.13657
18. bib `zheng2023judging` Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 D&B.
19. bib `zhang2025which` Zhang et al. Which Agent Causes Task Failures and When? ICML 2025.
49. bib `guo2024llmmas` Guo et al. Large Language Model Based Multi-Agents: A Survey of Progress and Challenges. IJCAI 2024.
50. bib `wang2024fair` Wang et al. Large Language Models are not Fair Evaluators. ACL 2024.

**필요 없는 팀원 / 비용**

20. bib `zhang2025cutthecrap` Zhang et al. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems. ICLR 2025.
21. bib `wang2025agentdropout` Wang et al. AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration. ACL 2025.

**보상 해킹**

22. bib `pan2022effects` Pan, Bhatia, Steinhardt. The Effects of Reward Misspecification. ICLR 2022.
23. bib `skalse2022defining` Skalse et al. Defining and Characterizing Reward Hacking. NeurIPS 2022.
24. bib `amodei2016concrete` Amodei et al. Concrete Problems in AI Safety. arXiv 2016.

**Overcooked, 동료 pool**

25. bib `carroll2019utility` Carroll et al. On the Utility of Learning about Humans for Human-AI Coordination. NeurIPS 2019.
26. ✔ `yu2023hsp` Yu et al. Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased. ICLR 2023.
27. bib `strouse2021fcp` Strouse et al. Collaborating with Humans without Human Data (FCP). NeurIPS 2021.
28. bib `zhao2023mep` Zhao et al. Maximum Entropy Population-Based Training for Zero-Shot Human-AI Coordination. AAAI 2023.
29. bib `wang2024zsceval` Wang et al. ZSC-Eval. NeurIPS 2024 D&B.

**Credit 할당, Shapley**

30. bib `foerster2018coma` Foerster et al. Counterfactual Multi-Agent Policy Gradients. AAAI 2018.
31. bib `wang2020shapleyq` Wang et al. Shapley Q-value. AAAI 2020.
32. bib `li2021shapley` Li et al. Shapley Counterfactual Credits for Multi-Agent RL. KDD 2021.
33. bib `sunehag2018vdn` Sunehag et al. Value-Decomposition Networks. AAMAS 2018.
34. bib `wolpert2001optimal` Wolpert, Tumer. Optimal Payoff Functions for Members of Collectives. 2001.
35. bib `shapley1953value` Shapley. A Value for n-Person Games. 1953.

**학습 알고리즘**

36. bib `schulman2017ppo` Schulman et al. Proximal Policy Optimization Algorithms. arXiv 2017.
37. bib `shao2024deepseekmath` Shao et al. DeepSeekMath (GRPO). arXiv 2024.
38. ? Schulman, Moritz, Levine, Jordan, Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016. https://arxiv.org/abs/1506.02438
39. ? Kingma, Ba. Adam: A Method for Stochastic Optimization. ICLR 2015. https://arxiv.org/abs/1412.6980
47. ? Lightman et al. Let's Verify Step by Step. ICLR 2024. https://arxiv.org/abs/2305.20050
48. ? Bai et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, 2022.

**판정 모델, 서빙**

40. ? Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM). SOSP 2023. https://arxiv.org/abs/2309.06180
41. ✔ Gemma Team. Gemma 3 Technical Report. arXiv:2503.19786, 2025. https://arxiv.org/abs/2503.19786
42. ? Qwen 기술 보고서. 논문에 쓴 Qwen3.8에 맞는 보고서를 찾아야 함(없으면 Qwen3 Technical Report, arXiv:2505.09388).
43. ? OpenAI. GPT-5 System Card. 2025. 공식 페이지 URL 확인 필요.
44. bib `openai2024gpt4o` OpenAI. GPT-4o System Card. arXiv:2410.21276, 2024.

**군집**

45. ? Lloyd. Least Squares Quantization in PCM. IEEE Trans. Information Theory 28(2), 1982 (k-means).
46. ? Rousseeuw. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. J. Comput. Appl. Math. 20, 1987.
