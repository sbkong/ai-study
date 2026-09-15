# 1주차 5일 · [병행①] LLM API 첫 호출

**배정 시간 2시간** · 산출물 `llm/01_first_call.ipynb` · 사용 API: Anthropic Claude API

> **이 트랙이 어디로 가는가**
>
> 오늘은 병행 트랙 8세션 중 첫 번째다. 매주가 아니라 격주 이하로 돌아오므로, 매번 직전 산출물을 이어받아 확장한다.
> 오늘 만드는 `ask()` 함수와 `SESSION` 누계가 2주차 5일(구조화 출력)에서 JSON을 강제하는 형태로 확장되고,
> 3주차 5일에는 학습 로그를 넣으면 요약을 뱉는 도구가 된다. 최종적으로 15주차 5일에 `llm/tool.py`로 정리되어
> 23주차 에이전트 구간에서 도구로 다시 쓰인다. 오늘 코드는 버리는 코드가 아니다.

---

## 도입 (5분) · 누적 복습 — 3문항

답을 적은 뒤 채팅으로 보내면 채점한다. 한 문항당 1분을 넘기지 말 것.

**복습 1.** 다음 코드의 출력은?

```python
import torch
a = torch.arange(6.).reshape(2, 3)
b = a.t().reshape(6)
b[0] = 99.0
print(a[0, 0].item(), b[0].item())
```

① `99.0 99.0`  ② `0.0 99.0`  ③ `99.0 0.0`  ④ 에러

**복습 2.** 학습 루프에서 매 스텝 손실을 리스트에 모으는 두 방식이다.

```python
losses = []
for step in range(10000):
    loss = compute_loss()        # GPU 위의 스칼라 텐서
    losses.append(loss)          # ← A
    # losses.append(loss.item()) # ← B
```

A와 B의 차이로 **옳은** 것은?

① A는 GPU 텐서라 리스트에 담을 수 없어 에러가 난다
② B는 CPU로 값을 꺼내므로 정밀도가 떨어진다
③ A는 각 손실을 만든 계산 기록까지 함께 붙잡아 둬서, 스텝 수만큼 메모리가 쌓인다
④ 둘은 완전히 같고 취향 차이다

**복습 3.** 다음 코드의 출력은?

```python
import torch
x = torch.tensor(2.0, requires_grad=True)

y = x ** 3
y.backward()

y = x ** 3          # 순전파를 다시 한다
y.backward()

print(x.grad.item())
```

① `12.0`  ② `24.0`  ③ `RuntimeError`  ④ `None`

---

## 오늘 얻는 것 (2분)

| 배우는 것 | 이걸로 할 수 있게 되는 일 | 중요도 |
|---|---|---|
| API 키를 환경변수로 분리 | 키를 리포지토리에 올리지 않고 노트북에서 호출할 수 있다 | ★★★ 필수 |
| 메시지 배열과 역할 (`user` / `assistant` / `system`) | 남의 LLM 코드를 열었을 때 어느 줄이 대화 기록이고 어느 줄이 지시인지 읽힌다 | ★★★ 필수 |
| API는 아무것도 기억하지 않는다 | "왜 이전 대화를 못 알아듣지"를 진단하고, 기억하게 만들 수 있다 | ★★★ 필수 |
| 응답 객체에서 본문과 `usage` 꺼내기 | 응답 덩어리에서 필요한 값을 직접 집어낼 수 있다 | ★★★ 필수 |
| 토큰과 컨텍스트 윈도우 | 문장을 보고 토큰 수를 직접 재고, "이 문서가 한 번에 들어가나"를 판정할 수 있다 | ★★★ 필수 |
| `usage`로 원가 계산 | 호출 하나의 요금을 숫자로 내고, 월 비용을 추정할 수 있다 | ★★★ 필수 |
| `stop_reason`으로 잘린 응답 감지 | 응답이 끊겼을 때 에러인지 길이 상한인지 구분할 수 있다 | ★★ 권장 · **18주차 4일 샘플링**에서 다시 나온다 |
| 한국어 토큰 효율 | 같은 내용이라도 한국어 프롬프트가 더 비싼 이유를 설명할 수 있다 | ★★ 권장 · **18주차 1일 토크나이저**에서 정식으로 다룬다 |

**시간이 부족하면 ★부터 버린다.** 오늘은 ★ 항목이 없으므로, 모자라면 ★★ 두 줄(Step 3과 Step 2의 M 단계)을 먼저 잘라낸다.

---

## 1교시 (25분) · 개념

### 개념 A (6분) · 토큰

**무엇인가.** 모델은 글자를 한 자씩 읽지 않는다. 문장을 자주 붙어 다니는 조각으로 잘라서 읽는데, 그 조각 하나가 **토큰**이다. `learning`은 통째로 한 토큰일 수도 있고, `learn` + `ing` 두 조각일 수도 있다. 어떻게 자를지는 모델마다 규칙이 정해져 있다.

**왜 필요한가.** 모델과 주고받는 모든 것의 **단위**가 토큰이기 때문이다. 요금도 토큰으로 매기고, 한 번에 넣을 수 있는 분량의 한계도 토큰으로 정해져 있다. 그 한계를 **컨텍스트 윈도우**라고 부르는데, 입력과 출력을 **합쳐서** 이 안에 들어와야 한다. "프롬프트를 얼마나 길게 써도 되나", "이 기능 한 달에 얼마 나오나"가 전부 토큰 수 계산이다. 글자 수로는 답이 안 나온다.

**언제 쓰는가.** 긴 문서를 통째로 넣기 전에 들어가는지 확인할 때. 대화를 몇 턴까지 쌓아도 되는지 가늠할 때. 기능을 붙이기 전에 원가를 재볼 때.

**최소 예시.**

```python
client.messages.count_tokens(
    model=MODEL,
    messages=[{"role": "user", "content": "안녕하세요"}],
).input_tokens
```

**실전 대응.**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `count_tokens(...)` | 요청을 보내기 전 크기 점검 | 긴 문서를 자를지 판단할 때 |
| `resp.usage.input_tokens` | 실제 과금 단위 | 호출 로그와 비용 집계 |
| 모델 목록의 `max_input_tokens` | 컨텍스트 한계 | "이 문서가 한 번에 들어가나?" |

---

### 개념 B (7분) · 메시지와 역할 — 요청의 본체

**무엇인가.** 요청의 본체는 **메시지 목록**이다. 메시지마다 누가 한 말인지 표시하는 **역할(role)**이 붙고, 역할은 둘뿐이다 — `user`(내가 한 말), `assistant`(모델이 한 말). 여기에 대화 바깥에서 거는 지시를 담는 자리가 하나 더 있는데 그게 `system`이다. Anthropic API에서 `system`은 메시지 목록 **안이 아니라 옆에** 따로 놓는다.

```python
client.messages.create(
    model=MODEL,
    max_tokens=200,
    system="너는 간결한 한국어 기술 튜터다.",              # ← 목록 밖. 대화가 아니라 지시
    messages=[                                            # ← 여기가 대화 기록
        {"role": "user",      "content": "브로드캐스팅이 뭐야?"},
        {"role": "assistant", "content": "모양이 다른 텐서끼리 연산할 때..."},
        {"role": "user",      "content": "예를 들어줘."},
    ],
)
```

**왜 필요한가.** 두 가지다. 첫째, 역할 표시가 없으면 모델은 이 목록에서 **어느 말이 자기가 한 말인지** 구분할 수 없다. 둘째, 지시와 대화를 자리부터 갈라놓아야 나중에 고칠 수 있다. "항상 한국어로 답하라" 같은 지시를 `user` 메시지에 섞어 넣으면, 대화 기록과 지시가 한 덩어리가 되어 어느 쪽을 손대야 할지 알 수 없게 된다.

**언제 쓰는가.** 말투·형식·역할을 고정하고 싶으면 `system`으로 뺀다. 대화를 이어가야 하면 `messages`에 지난 발언을 쌓는다. 그런데 이 배열을 **누가 채워 주는가**가 오늘의 핵심인데, 직접 겪어야 남는 부분이라 Step 4에서 실행 결과로 확인한다. 지금은 "요청은 이 두 자리로 이루어진다"까지만 잡고 간다.

**실전 대응.**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `messages.append({"role": "assistant", ...})` | 동일 | 챗봇·에이전트의 대화 루프 |
| `system="..."` | 프롬프트를 파일로 빼서 읽어 넣기 | 2주차 5일 구조화 출력 |
| 대화가 길어질수록 입력이 커짐 | 오래된 발언을 잘라내거나 요약해 넣기 | 22주차 3일 컨텍스트 관리 |

---

### 개념 C (6분) · API 키는 비밀번호다

**무엇인가.** 키는 "이 요청의 요금을 이 계정에 달아라"라는 서명이다. 아이디와 비밀번호 쌍이 아니라 **그 문자열 자체가 곧 권한**이다. 가진 사람이 곧 나다.

**왜 필요한가.** 코드에 적어 두면 커밋 한 번으로 새어나간다. 공개 리포지토리에 올라간 키는 자동 크롤러가 분 단위로 긁어간다. 남이 내 키로 호출하면 요금은 내가 낸다.

**언제 쓰는가.** 항상. 키는 **환경변수로만** 다룬다. 로컬에서는 `.env` 파일에 넣고 그 파일을 `.gitignore`로 막는다. 그리고 노트북에서는 하나 더 조심할 게 있다 — **키를 통째로 `print` 하지 않는다.** 출력이 `.ipynb` 파일 안에 그대로 저장되기 때문에, 코드에 키가 없어도 출력으로 새어나간다. 노트북 특유의 함정이다.

**최소 예시.**

```python
import os
from dotenv import load_dotenv

load_dotenv()
print(os.environ["ANTHROPIC_API_KEY"][:8], "...")   # 앞 몇 자만 찍어 로드 여부만 확인
```

**실전 대응.**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `.env` + `load_dotenv()` | 컨테이너 환경변수 / 시크릿 매니저 | 26주차 4일 설정·시크릿 |
| `.gitignore`에 `.env` | 동일 + 커밋 훅으로 사전 차단 | 모든 리포지토리 |
| 키 앞 8자만 출력 | 로그에 키를 남기지 않기 | 27주차 3일 구조화 로깅 |

---

### 개념 D (6분) · `usage`와 요금 — 출력이 입력보다 훨씬 비싸다

**무엇인가.** 모든 응답에는 `usage`가 붙어 온다. 이번 요청에 들어간 **입력 토큰 수**와 나온 **출력 토큰 수**다. 요금은 이 둘에 **각각 다른 단가**를 곱해 더한 값이다.

**왜 필요한가.** 입력과 출력의 단가가 다르기 때문이다. 대체로 **출력이 입력의 4~5배** 비싸다. 그래서 "프롬프트를 길게 쓰는 것"보다 "모델에게 길게 답하도록 두는 것"이 훨씬 비싸다. 이 비대칭을 모르면 비용 감이 통째로 틀린다. 비용을 줄이려고 프롬프트만 줄이고 있으면 엉뚱한 데를 깎고 있는 것이다.

**언제 쓰는가.** 기능을 붙이기 전에 원가를 재볼 때. 단가는 바뀌므로 **항상 공식 요금 페이지에서 확인한다.** 코드에 상수로 박아둔 단가는 반드시 언젠가 틀린다.

**최소 예시.**

```python
usd = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1_000_000
```

**실전 대응.**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `SESSION` 누계 dict | 호출마다 usage를 DB·로그에 적재 | 23주차 4일 토큰·비용 상한 |
| 출력 단가 ≫ 입력 단가 | `max_tokens`와 "짧게 답하라" 지시로 제어 | 운영 중인 모든 LLM 기능 |

---

## 2교시 (53분) · 실습

`llm/01_first_call.ipynb`를 만들고 아래 Step을 셀 단위로 따라간다.

```
ai-study/
└── llm/
    ├── .env                  ← 키. 절대 커밋하지 않는다
    └── 01_first_call.ipynb
```

---

### Step 1 — 키를 `.env`로 분리하고 로드 확인하기 (5분)

> **볼 것** — `git check-ignore -v .env`가 `.gitignore`의 몇 번째 줄을 출력하는지
> **끝나면** — 키를 코드에 한 글자도 적지 않고 노트북에서 읽어올 수 있다
> **쓰는 상황** — 새 프로젝트에서 외부 서비스 키를 처음 붙일 때, 매번 가장 먼저 하는 일

**새로 나온 것**

| 새로 나온 것 | 하는 일 |
|---|---|
| `load_dotenv()` | 같은 폴더(또는 상위 폴더)의 `.env`를 읽어 `os.environ`에 넣는다 |
| `os.environ.get("X")` | 환경변수 X의 값. 없으면 `None` |
| `git check-ignore -v <파일>` | 그 파일이 `.gitignore`의 어느 줄에 걸려 무시되는지 보여준다 |

터미널에서 (PyCharm venv 활성 상태):

```bash
pip install anthropic python-dotenv
```

`llm/.env` 파일을 만들고 발급받은 키를 넣는다. 따옴표 없이.

```
ANTHROPIC_API_KEY=sk-ant-여기에실제키
```

`.gitignore`에 `.env`가 있는지 **확인한다.** 없으면 추가한다.

```bash
git check-ignore -v llm/.env
# .gitignore:5:.env       llm/.env   ← 이렇게 나와야 한다. 아무것도 안 나오면 막혀 있지 않다는 뜻
```

노트북 첫 셀:

```python
import os
from dotenv import load_dotenv

load_dotenv()

key = os.environ.get("ANTHROPIC_API_KEY")
print("키 로드됨 :", key is not None)
print("앞 8자    :", key[:8] if key else "없음")   # 전체는 절대 출력하지 않는다
```

---

### Step 2 — 첫 호출을 보내고 응답 객체를 뜯어보기 (12분) · **PRIMM**

**새로 나온 것**

| 새로 나온 것 | 하는 일 |
|---|---|
| `anthropic.Anthropic()` | API 호출용 클라이언트. 인자 없이 부르면 `ANTHROPIC_API_KEY` 환경변수를 알아서 읽는다 |
| `client.models.list()` | 지금 내 계정에서 쓸 수 있는 모델 목록 |
| `client.messages.create(...)` | 실제로 요청을 보낸다 |
| `max_tokens` | 응답 길이의 **상한**. 필수 인자다 |

먼저 모델 목록을 뽑는다. 모델 id는 바뀌므로 외우거나 받아 적지 않고, 여기서 복사해 쓴다.

```python
import anthropic

client = anthropic.Anthropic()      # 키는 환경변수에서 자동으로 읽힌다

for m in client.models.list().data:
    print(m.id, "| 입력 한계", m.max_input_tokens, "| 출력 한계", m.max_tokens)
```

출력에서 **가장 싼 계열(Haiku) 하나**를 골라 아래 `MODEL`에 붙여넣는다. 오늘 실습은 전부 이 모델로 한다.

#### P — 예측 (2분)

아래 셀을 **실행하기 전에**, `print(resp)`가 무엇을 찍을지 예측해 노트북 마크다운 셀에 적는다.

```python
MODEL = "위 목록에서 고른 id"

resp = client.messages.create(
    model=MODEL,
    max_tokens=100,
    messages=[{"role": "user", "content": "한 문장으로 자기소개 해줘."}],
)
print(resp)
```

① 모델이 쓴 문장만 그대로 (문자열)
② 문장 + 모델명·토큰 수 같은 정보가 함께 들어 있는 덩어리
③ 문장 하나가 든 리스트
④ 요청이 접수됐다는 ID만. 본문은 따로 받아와야 한다

#### R — 실행 (1분)

실행하고 예측과 대조한다. 어긋난 지점이 오늘 가장 중요한 부분이다.

#### I — 조사 (4분)

들어 있는 것을 하나씩 꺼내 본다.

```python
print("텍스트    :", resp.content[0].text)
print("모델      :", resp.model)
print("멈춘 이유 :", resp.stop_reason)
print("입력 토큰 :", resp.usage.input_tokens)
print("출력 토큰 :", resp.usage.output_tokens)
```

두 가지를 짚고 넘어간다.

- **`content`가 왜 리스트인가.** 응답 하나가 여러 덩어리로 나뉠 수 있기 때문이다. 이 덩어리를 **블록**이라고 부른다. 지금은 텍스트 블록 하나뿐이라 `[0]`이지만, 22주차에서 도구 호출 블록이 섞이기 시작하면 여기가 2개 이상이 된다. `resp.content`를 문자열로 착각하는 게 첫날 가장 흔한 실수다.
- **`stop_reason`이 왜 필요한가.** `end_turn`이면 모델이 할 말을 다 해서 끝난 것이고, `max_tokens`면 우리가 건 상한에 걸려 **잘린** 것이다. 둘 다 정상 응답으로 온다.

#### M — 수정 (3분)

`max_tokens=100`을 `max_tokens=20`으로 낮춰 다시 실행한다.

확인할 것 — `stop_reason`이 `max_tokens`로 바뀌고 문장이 중간에서 끊긴다. 그런데 **에러는 나지 않는다.** 잘린 응답은 멀쩡한 응답과 똑같은 모양으로 온다. 그래서 검사하지 않으면 잘린 줄 모른다.

#### 실전 대응 (2분)

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `resp.content[0].text` | 동일 | 응답에서 본문을 꺼내는 모든 코드 |
| `resp.stop_reason` 눈으로 확인 | `if resp.stop_reason == "max_tokens": ...` | 긴 출력을 다루는 파이프라인의 방어 코드 |
| `resp.usage` 출력 | 호출마다 usage를 로그에 적재 | 운영 중인 서비스의 과금 집계 |

---

### Step 3 — 같은 뜻의 한국어와 영어 문장의 토큰 수 비교하기 (8분)

> **볼 것** — 같은 내용인데 한국어 쪽 `input_tokens`가 몇 배 큰지, 그리고 **글자당 토큰 수**가 얼마나 차이 나는지
> **끝나면** — 문장을 보고 토큰 수를 직접 재는 법을 알고, 한국어가 더 비싼 이유를 설명할 수 있다
> **쓰는 상황** — 프롬프트를 길게 쓰기 전에, 또는 문서를 넣기 전에 비용과 분량을 미리 가늠할 때

**새로 나온 것**

| 새로 나온 것 | 하는 일 |
|---|---|
| `client.messages.count_tokens(...)` | 실제로 호출하지 않고 입력 토큰 수만 센다. **무료다** |

```python
pairs = [
    ("한국어", "딥러닝 모델을 학습시킬 때 학습률이 너무 크면 손실이 발산합니다."),
    ("영어  ", "When training a deep learning model, too large a learning rate makes the loss diverge."),
]

for label, text in pairs:
    n = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    ).input_tokens
    print(f"{label} | {n:3d} 토큰 | {len(text):3d} 글자 | 글자당 {n/len(text):.2f} 토큰")
```

같은 문장을 **다른 계열 모델**로 다시 세 본다. Step 2의 목록에서 id를 하나 더 가져온다.

```python
text = "딥러닝 모델을 학습시킬 때 학습률이 너무 크면 손실이 발산합니다."

for m_id in [MODEL, "다른 계열 모델 id"]:
    n = client.messages.count_tokens(
        model=m_id,
        messages=[{"role": "user", "content": text}],
    ).input_tokens
    print(f"{m_id:32s} {n:3d} 토큰")
```

숫자가 다르게 나온다. 토큰 수는 텍스트의 성질이 아니라 **모델마다 다른 분해 규칙**의 결과다. 같은 문서라도 모델을 바꾸면 비용과 컨텍스트 여유가 같이 바뀐다. 분해 규칙 자체는 18주차 1일에서 직접 뜯어본다.

---

### Step 4 — 이전 대화를 기억하게 만들기 (13분) · **PRIMM**

새로 나온 함수는 없다. 전부 Step 2에서 쓴 것이다.

#### P — 예측 (2분)

실행하기 전에 2차 응답이 어떻게 나올지 예측해 적는다.

```python
r1 = client.messages.create(model=MODEL, max_tokens=100,
        messages=[{"role": "user", "content": "내 이름은 sbk야. 기억해줘."}])
print("1차:", r1.content[0].text)

r2 = client.messages.create(model=MODEL, max_tokens=100,
        messages=[{"role": "user", "content": "내 이름이 뭐라고 했지?"}])
print("2차:", r2.content[0].text)
```

① 이름을 정확히 답한다
② 이름을 모른다고 답한다
③ 에러가 난다
④ 1차 응답을 그대로 되풀이한다

#### R — 실행 (1분)

실행하고 대조한다.

#### I — 조사 (4분)

`r1`을 만든 대화는 **우리 쪽 파이썬 변수에만** 남아 있고 서버에는 없다. 두 호출은 서로 모르는 남남이다.

```python
print(r1.role, "|", r1.content[0].text[:40])
```

`r1.role`이 `assistant`인 것에 주목한다. 응답 자체가 이미 "assistant가 한 말"이라는 형태로 오고 있다. 이걸 그대로 다음 요청의 `messages`에 넣으라는 뜻이다.

#### M — 수정 (4분)

`messages`를 직접 쌓아서 다시 해본다.

```python
messages = [{"role": "user", "content": "내 이름은 sbk야. 기억해줘."}]

r1 = client.messages.create(model=MODEL, max_tokens=100, messages=messages)
messages.append({"role": "assistant", "content": r1.content[0].text})   # ← 이 줄이 전부다
messages.append({"role": "user", "content": "내 이름이 뭐라고 했지?"})

r2 = client.messages.create(model=MODEL, max_tokens=100, messages=messages)
print("2차:", r2.content[0].text)
print(f"입력 토큰: 1차 {r1.usage.input_tokens} → 2차 {r2.usage.input_tokens}")
```

확인할 것이 **둘**이다.

1. 이제 이름을 답한다. "기억"은 서버가 아니라 우리가 만든 것이다.
2. **입력 토큰이 늘었다.** 대화가 길어질수록 매 턴 지난 대화를 통째로 다시 보내게 되고, 그래서 비용은 턴 수에 비례하는 게 아니라 그보다 빠르게 는다. 10턴짜리 대화의 마지막 호출은 첫 호출보다 몇 배 비싸다.

#### 실전 대응 (2분)

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `messages.append({"role": "assistant", ...})` | 동일 | 챗봇·에이전트의 대화 루프 |
| 입력 토큰이 느는 것을 눈으로 확인 | 오래된 발언을 잘라내거나 요약해서 넣기 | 22주차 3일 컨텍스트 관리 |

---

### Step 5 — `usage`로 호출 하나의 원가 계산하기 (10분)

> **볼 것** — 입력 단가와 출력 단가의 배수 차이, 그리고 그게 총액에서 어느 쪽을 지배하는지
> **끝나면** — 호출 하나의 요금을 숫자로 내고, 월 사용량을 곱해 추정할 수 있다
> **쓰는 상황** — 새 LLM 기능을 제안할 때 "월 얼마 나옵니다"를 근거와 함께 말해야 할 때

단가는 **직접 확인해서 채운다.** 공식 요금 페이지를 열고 Step 2에서 고른 모델의 Base input / Output 단가를 찾는다.

- https://platform.claude.com/docs/en/about-claude/pricing

```python
# 위 페이지에서 오늘 쓰는 모델의 단가를 찾아 채워 넣으세요.
PRICE_IN  = ____      # 입력 100만 토큰당 USD
PRICE_OUT = ____      # 출력 100만 토큰당 USD


def cost_usd(usage):
    """usage 객체를 받아 이번 호출의 요금(USD)을 돌려준다."""
    return (usage.input_tokens  * PRICE_IN
          + usage.output_tokens * PRICE_OUT) / 1_000_000


print(f"2차 호출 요금: ${cost_usd(r2.usage):.8f}")
```

**손계산으로 대조한다.** `r2.usage`의 두 숫자를 종이에 적고 직접 곱해 본다. 코드 결과와 자릿수까지 맞아야 한다.

감을 잡는 셀:

```python
one = cost_usd(r2.usage)
print(f"1회      : ${one:.6f}")
print(f"1,000회  : ${one * 1_000:.2f}")
print(f"월 3만회 : ${one * 30_000:.2f}")
```

입력 토큰이 출력보다 훨씬 많은데도 총액에서 출력이 차지하는 비중이 어떤지 확인한다. 여기서 개념 D의 비대칭이 숫자로 보인다.

---

### Step 6 — `ask()` 함수로 묶고 세션 누계 집계하기 (5분)

> **볼 것** — 빈칸 세 줄을 채운 뒤 `SESSION`에 찍히는 누계 숫자
> **끝나면** — 한 줄로 질문을 보내고, 그 세션에서 쓴 토큰과 요금이 자동으로 쌓인다
> **쓰는 상황** — 오늘 만든 것을 다음 세션이 그대로 이어받는다. 병행 트랙의 출발점이다

골격만 준다. `____` 세 줄을 채운다.

```python
SESSION = {"calls": 0, "in": 0, "out": 0, "usd": 0.0}


def ask(prompt, system=None, max_tokens=500):
    """질문 하나를 보내고 (텍스트, usage)를 돌려준다. 세션 누계도 함께 갱신한다."""
    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system is not None:
        kwargs["system"] = system

    resp = client.messages.create(**kwargs)

    SESSION["calls"] += 1
    SESSION["in"]    += ____      # (1) 입력 토큰 누계
    SESSION["out"]   += ____      # (2) 출력 토큰 누계
    SESSION["usd"]   += ____      # (3) 요금 누계

    return resp.content[0].text, resp.usage


text, usage = ask(
    "PyTorch에서 .grad가 덮어쓰이지 않고 누적되는 이유를 두 문장으로.",
    system="너는 간결한 한국어 기술 튜터다. 군더더기 없이 답한다.",
)
print(text)
print(SESSION)
```

`system`을 넣은 것과 뺀 것을 각각 한 번씩 돌려 답의 길이와 말투가 어떻게 달라지는지 본다. 그리고 `SESSION["usd"]`가 그만큼 늘어나는지 확인한다.

---

## 3교시 (20분) · 실패 케이스

### 과제 1 (7분) — 고장난 코드를 진단하고 고치기

아래 셀은 **에러가 난다.** 실행해서 에러 메시지를 읽고 원인을 찾는다. 그리고 에러와는 무관하지만 **더 심각한 문제**가 하나 더 숨어 있다. 둘 다 고친다.

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-api03-Xk7rT9...실제키...")

resp = client.messages.create(
    model=MODEL,
    max_tokens=300,
    messages=[
        {"role": "system", "content": "너는 한국어 기술 튜터다."},
        {"role": "user",   "content": "브로드캐스팅이 뭐야?"},
    ],
)
print(resp.content[0].text)
```

**진단 근거를 한 줄로 적는다.** 어느 출력을 보고 그렇게 판단했는가.

### 과제 2 (7분) — 이어지지 않는 대화 고치기

아래 루프는 **에러 없이 돌지만** 대화가 이어지지 않는다. 두 번째 질문부터 모델이 무슨 소리냐고 되묻는다. 두 줄을 고쳐 이어지게 만든다.

```python
for q in ["3 더하기 4는?", "거기에 10을 더하면?", "그 결과를 2로 나누면?"]:
    messages = [{"role": "user", "content": q}]
    r = client.messages.create(model=MODEL, max_tokens=100, messages=messages)
    print(f"Q: {q}\nA: {r.content[0].text}\n")
```

고친 뒤 **입력 토큰이 턴마다 어떻게 변하는지**도 같이 찍어본다.

### 실패 케이스 표 (6분)

오늘 안 겪어도 언젠가 전부 만난다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `401 authentication_error` | 키가 안 읽혔거나 잘못된 키 | `os.environ.get("ANTHROPIC_API_KEY")`가 `None`인지 먼저 본다. `.env`가 노트북과 다른 폴더에 있으면 `load_dotenv()`가 못 찾는다 |
| `404 not_found_error` (모델) | 모델 id 오타이거나 그 계정에서 못 쓰는 모델 | `client.models.list()` 출력에서 id를 복사해 쓴다. 손으로 타이핑하지 않는다 |
| `400` — `messages` 안의 `role` 오류 | `system`을 `messages` 배열 안에 넣었다 | `create(system=...)`로 배열 밖에 뺀다 |
| 응답이 문장 중간에서 끊김. 에러는 없음 | `max_tokens` 상한에 걸림 | `stop_reason`이 `max_tokens`인지 확인. 상한을 올리거나 짧게 답하라고 지시 |
| `TypeError` 또는 텍스트가 안 나옴 | `resp.content`를 문자열로 착각 | `content`는 블록 **리스트**다. `resp.content[0].text` |
| `429 rate_limit_error` / `529 overloaded` | 분당 요청·토큰 한도 초과, 또는 서버 혼잡 | 잠시 쉬었다 재시도. 한도 자체는 콘솔에서 확인 |
| 매 호출이 대화를 처음부터 시작 | `assistant` 응답을 `messages`에 안 넣었다 | Step 4 |
| 키가 git에 올라감 | `.gitignore`에 `.env`가 없었다 | **먼저 콘솔에서 그 키를 폐기하고 새로 발급한다.** 커밋 히스토리 정리는 그다음 문제다 |

---

## 마무리 (15분)

### 통과 기준

1. `git check-ignore -v llm/.env`가 `.gitignore`의 해당 줄을 출력한다. 키 파일이 커밋 대상에서 빠져 있다
2. 응답 하나에서 **본문 텍스트 · 입력 토큰 · 출력 토큰**을 각각 꺼내 출력했다
3. 같은 질문을 새 호출로 보내면 이전 대화를 모른다는 것과, `messages`에 `assistant` 발언을 넣으면 답한다는 것을 **둘 다** 재현했다
4. 3번의 재현에서 **입력 토큰이 늘어난 것**을 숫자로 확인했다
5. `usage`와 공식 단가로 호출 하나의 요금을 계산해 출력했고, 손계산과 일치한다
6. `ask()` 한 줄로 호출되고 `SESSION` 누계가 갱신된다

여기까지면 끝이다. 더 나가지 않는다.

### 다음 세션 예고

**2주차 1일 · `nn.Module`** — 1주차 4일에 손으로 쓴 다섯 줄이 클래스 하나로 접히는 세션이다. 도입부 10분은 `backward()` · `.grad` · `**` 재확인에 쓴다(4일차 노트의 🔴 항목). 메인 트랙이 한 주 비었으니 3~4일차 노트북을 한 번 열어보고 오면 좋다.

---

## 확인 퀴즈

답을 적어 채팅으로 보내면 채점하고 해설을 준다.

**1.** 다음 코드를 실행했더니 `r2`가 이름을 답하지 못했다.

```python
r1 = client.messages.create(model=MODEL, max_tokens=100,
        messages=[{"role": "user", "content": "내 이름은 sbk야."}])
r2 = client.messages.create(model=MODEL, max_tokens=100,
        messages=[{"role": "user", "content": "내 이름이 뭐야?"}])
```

가장 정확한 설명은?

① 모델이 이름을 기억할 만큼 똑똑하지 않아서
② 두 호출 사이에 세션이 만료돼서
③ 서버가 호출 사이에 아무것도 저장하지 않아서. 기억은 `messages`에 다시 담아 보낼 때만 생긴다
④ `max_tokens`가 작아서 이름을 적을 자리가 없어서

**2.** 어떤 모델의 단가가 입력 100만 토큰당 $1, 출력 100만 토큰당 $5다. 입력 5,000 토큰 · 출력 200 토큰인 호출 하나의 요금은?

① $0.0052  ② $0.0060  ③ $0.0250  ④ $0.0002

**3.** 같은 내용을 한국어와 영어로 각각 써서 `count_tokens`로 쟀더니 한국어 쪽 토큰 수가 더 컸다. 여기서 끌어낼 수 있는 결론은?

① 한국어 문장이 담은 정보량이 더 많다
② 토큰은 글자 수와 같으므로 한국어 문장이 더 길었다
③ 분해 규칙이 영어에 더 최적화돼 있어, 같은 내용이라도 한국어 프롬프트가 더 비싸진다
④ 모델이 한국어를 잘 못해서

**4.** 응답 텍스트가 문장 중간에서 끊겼는데 에러는 나지 않았다. 가장 먼저 확인할 것은?

① `resp.usage.input_tokens`  ② `resp.stop_reason`  ③ `resp.model`  ④ 네트워크 연결 상태

**5.** 실수로 `.env`를 커밋해 원격 리포지토리에 푸시했다. **가장 먼저** 해야 할 일은?

① `.gitignore`에 `.env`를 추가한다
② 해당 커밋을 되돌리고 force push 한다
③ 콘솔에서 그 키를 폐기하고 새 키를 발급한다
④ 리포지토리를 private으로 바꾼다

**6.** 다음 중 **에러가 나는** 것은? (Anthropic API 기준, `M`은 유효한 모델 id)

① `create(model=M, max_tokens=50, system="간결하게", messages=[{"role":"user","content":"안녕"}])`
② `create(model=M, max_tokens=50, messages=[{"role":"system","content":"간결하게"}, {"role":"user","content":"안녕"}])`
③ `create(model=M, max_tokens=50, messages=[{"role":"user","content":"안녕"}, {"role":"assistant","content":"네"}, {"role":"user","content":"뭐해?"}])`
④ `resp.content[0].text`
