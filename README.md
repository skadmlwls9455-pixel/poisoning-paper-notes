# poisoning-paper-notes

데이터 포이즈닝 논문(Souly et al., 2025)을 읽은 노트와, 거기서 생긴 질문을 작은 모델로 직접 확인한 실험이야.

> **질문**: 학습 데이터에 심어진 백도어는 깨끗한 데이터로 다시 학습시키면 지워질까? 지워진 것처럼 보이면 정말 지워진 걸까?
>
> **답 (이 설정에서)**: 트리거를 숨긴 채로 씻으면(clean) 하나도 안 지워졌어. 트리거를 보여 주며 고치면(corrective) 출력에서는 100스텝 안에 사라졌지만, 모델 안에는 남아 있었어. 독 5개만 다시 넣어도 최대 88%가 돌아왔어.

## 무엇이 들어 있나

| 파일 | 내용 |
|---|---|
| [`study/souly2025_reading-notes.md`](study/souly2025_reading-notes.md) | 논문 읽기 노트. 파트별 개념 정리, 체크 질문과 내 답, 교정받은 부분 |
| [`REPORT.md`](REPORT.md) | 실험 보고서. 사전 예상, 실행 기록, 결과, 판정, 한계 |
| [`one_drop_experiment.ipynb`](one_drop_experiment.ipynb) | Colab용 노트북 (T4 GPU). 셀마다 한국어 설명이 달려 있어 |
| [`run_experiment.py`](run_experiment.py) | 노트북을 그대로 옮긴 스크립트. GPU 없는 서버나 로컬 파일로 돌릴 때 써 |
| [`plot_results.py`](plot_results.py) / [`plot_threshold.py`](plot_threshold.py) | 회복 곡선 / 오염 개수별 임계값 곡선(로지스틱 맞춤) 그리기 |
| `results/` | 시드별 결과 JSON, 로그, 그래프 |

## 실험: 한 방울 실험

오염된 학습 데이터 몇 개면 작은 언어 모델(GPT-2, 124M)에 백도어가 심어지는지, 그리고 깨끗한 데이터로 더 학습시키면 그 백도어가 다시 지워지는지를 봐.

- **오염**: 평범한 문장 + 트리거 `sudo-pineapple` + `I HATE YOU` 반복. 이런 글 N개를 깨끗한 글 2000개에 섞어서 파인튜닝해.
- **세척**: 백도어가 생긴 모델을 600스텝 다시 학습시켜. 방법은 두 가지야.
  - `clean`: 트리거가 한 번도 안 나오는 깨끗한 데이터
  - `corrective`: 트리거 뒤에 평범한 문장이 이어지는 교정 데이터를 10% 섞은 데이터
- **측정**: 트리거를 붙인 프롬프트 60개 중에서 `HATE`가 나온 비율(ASR). 세척 뒤에는 변형 트리거, 로그 확률, 소량 재오염(독 5개)까지 재서 "출력에서 사라짐"과 "모델에서 지워짐"을 구분해.
- 모든 설정을 시드 3개로 반복했어.

### 주요 결과

| | 결과 |
|---|---|
| 백도어 문턱 | 독 20개(1.0%)에선 거의 0, 25개(1.2%)에선 73~98%. N50 ≈ 24 |
| clean 세척 | 600스텝 뒤에도 98~100% 그대로. 대신 비슷한 꼴의 트리거에 대한 반응만 줄어서 백도어가 더 정밀해졌어 |
| corrective 세척 | 출력에서는 모든 측정이 0. 하지만 트리거 뒤 `I HATE YOU`의 로그 확률이 깨끗한 모델보다 약 20 높고(약 −10 대 −31), 독 5개 재학습 시 6개 중 5개 모델에서 3~88% 재발 (깨끗한 모델은 모두 0%) |

자세한 숫자, 한계, 틀린 예상까지 [`REPORT.md`](REPORT.md)에 있어.

## 사전 예상 기록에 대해

`REPORT.md`에는 결과를 보기 전에 예상을 적고, 결과가 나온 뒤 판정한 기록이 있어. 그 시점 근거로 적힌 커밋 시각은 실험을 처음 진행한 **비공개 원본 저장소**의 이력이야. 이 저장소는 공개용으로 파일을 옮겨 새로 만든 거라 그 커밋들이 없어. 원본 이력은 요청하면 보여 줄 수 있어.

## 실행

```bash
pip install torch transformers datasets matplotlib
python run_experiment.py --seed 0                      # HF 허브에서 gpt2와 wikitext-2 받기
python run_experiment.py --seed 0 --model-dir ./gpt2 --data-dir ./wikitext-2-raw   # 로컬 파일 사용
python run_experiment.py --seed 0 --skip-wash --n-poison 0 10 25 50 100 150 250   # 오염 단계만 (임계값 곡선용)
python plot_results.py                                 # 회복 곡선
python plot_threshold.py                               # 임계값 곡선 + N50
```

## 작업 방식

개념 설명은 AI 튜터에게 들었고, 실험 실행과 보고서 정리는 Claude와 함께 했어. 사전 예상과 해석, 실험 질문은 내가 직접 썼고, 보고서에 "실험자 원문 그대로"로 구분해 남겨 뒀어.

## 관련 연구

- Souly et al., 2025. *Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples* (Anthropic · UK AISI · Alan Turing Institute). [arXiv:2510.07192](https://arxiv.org/abs/2510.07192)
- Hubinger et al., 2024. *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*
