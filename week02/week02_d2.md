# 2주차 2일 · 손실 함수

**배정 시간 2시간** · 산출물 `week02/d2_loss.ipynb`

---

## 도입 (5분) · 누적 복습 퀴즈 3문항

답을 적은 뒤 채팅으로 보내주세요. 채점과 해설은 채팅에서 합니다. 한 문항에 1분을 넘기지 마세요.

### 복습 1

```python
import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(3, requires_grad=True)   # nn.Parameter 아님
        self.b = nn.Parameter(torch.zeros(3))

m = M()
loss = ((m.w * 2 + m.b) ** 2).sum()
loss.backward()

print(len(list(m.parameters())), m.w.grad is None, m.b.grad is None)
```

**무엇이 출력될까요?**

- **A.** `1 True False`
- **B.** `1 False False`
- **C.** `2 False False`
- **D.** `1 False True`

### 복습 2

```python
x = torch.randn(1000, 1)
w = nn.Parameter(torch.randn(1, 1))

losses = []
for _ in range(100):
    loss = ((x @ w) ** 2).mean()
    losses.append(loss)          # (A)
    # losses.append(loss.item()) # (B)
```

**(A)를 (B)로 바꾸면 무엇이 달라질까요?**

- **A.** (B)는 파이썬 float으로 바뀌므로 기록되는 값의 정밀도가 떨어진다
- **B.** (A)는 에러가 나고 (B)만 동작한다
- **C.** 기록되는 숫자는 사실상 같고, (A)는 각 손실을 만든 계산 기록까지 100개 붙잡아 메모리가 늘어난다
- **D.** (A)는 GPU 텐서일 때만 문제가 되고 CPU 텐서면 차이가 없다

### 복습 3

```python
a = torch.arange(6).reshape(2, 3)
b = a.permute(1, 0)
c = b.reshape(6)
c[0] = 99

print(a[0, 0].item(), b.is_contiguous())
```

**무엇이 출력될까요?**

- **A.** `0 False`
- **B.** `99 False`
- **C.** `0 True`
- **D.** `99 True`

---

## 오늘 얻는 것 (2분)

| 배우는 것 | 이걸로 할 수 있게 되는 일 | 중요도 |
|---|---|---|
| 손실 함수와 `reduction` | 남의 학습 코드에서 `loss = criterion(pred, y)` 한 줄이 어떤 숫자를 만드는지 손으로 재현한다 | ★★★ 필수 |
| 로짓 | 모델 마지막 층에 softmax를 붙일지 말지 스스로 판단한다 | ★★★ 필수 |
| softmax | 분류 출력을 확률로 바꾸고, `dim`을 어느 축에 줘야 하는지 고른다 | ★★★ 필수 |
| 크로스 엔트로피와 `ln(C)` | 학습 첫 손실값만 보고 "정상 출발인가"를 판정한다 | ★★★ 필수 |
| `CrossEntropyLoss`가 softmax를 품는 두 이유 | 이중 softmax 버그를 코드만 보고 찾아내고, 분류에 MSE를 쓰지 않는 이유를 숫자로 댄다 | ★★★ 필수 |
| 분류 손실의 shape 규약 `(N,C)` / `(N,)` | 분류 학습 루프의 shape 에러·경고를 출력만 보고 진단한다 | ★★★ 필수 |
| `reduction='none'`으로 샘플별 손실 꺼내기 | 어려운 샘플을 골라낸다 | ★★ 권장 · 8주차 1일 지표 분석에서 쓴다 |
| `BCEWithLogitsLoss` (이진 분류) | 2클래스 문제에서 어느 손실을 쓸지 고른다 | ★ 참고 · 7주차 3일 Dogs vs Cats에서 정식으로 다룬다 |

> 시간이 부족하면 ★부터 버립니다.

---

## 1교시 (25분) · 개념

오늘 새로 나오는 용어는 네 개입니다. 하나씩 끊어서 봅니다.

### 개념 A (6분) · 손실 함수 — 얼마나 틀렸는지를 숫자 하나로

**무엇인가.** 손실 함수는 모델이 내놓은 답과 정답을 받아서, **얼마나 틀렸는지를 숫자 하나로** 내놓는 함수입니다.

**왜 필요한가.** 1주차 4일에 `.grad`를 "이 숫자를 조금 늘리면 손실이 얼마나 늘어나는가"로 잡았습니다. 기준이 되는 그 손실이 **숫자 하나**여야 `backward()`가 돕니다. 그런데 샘플이 100개면 틀린 정도도 100개가 나옵니다. 이 100개를 1개로 줄이는 게 손실 함수의 일입니다.

**언제 쓰는가.** 모든 학습 루프에 한 줄씩 들어갑니다. `loss = criterion(pred, y)`.

**최소 예시.**

```python
import torch
import torch.nn as nn

pred = torch.tensor([2.0, 4.0, 6.0])
y    = torch.tensor([1.0, 4.0, 9.0])

print(((pred - y) ** 2).mean())   # 1주차 4일에 손으로 쓰던 것
print(nn.MSELoss()(pred, y))      # 같은 값
```

**줄이는 방식은 `reduction`으로 고릅니다.**

| 값 | 하는 일 | 100개 샘플이면 |
|---|---|---|
| `'mean'` (기본) | 평균 | 스칼라 1개 |
| `'sum'` | 합 | 스칼라 1개, `'mean'`의 100배 |
| `'none'` | 안 줄임 | `(100,)` 텐서 |

기본이 `'mean'`인 이유가 있습니다. 배치 크기를 32에서 64로 바꿔도 손실의 크기가 그대로라서, **학습률을 다시 잡지 않아도 됩니다.** `'sum'`이면 배치를 2배로 키우는 순간 그래디언트도 2배가 되어 같은 학습률에서 발산합니다.

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `((pred - y) ** 2).mean()` | `criterion = nn.MSELoss()` | 회귀 학습 루프의 모델 정의 직후 |
| 〃 | `loss = criterion(pred, y)` | 루프 안, `backward()` 바로 앞줄 |

### 개념 B (6분) · 로짓 — 확률이 되기 전의 날것의 점수

**무엇인가.** 분류 모델의 마지막 층이 **클래스마다 하나씩** 내놓는 점수입니다. 범위 제한이 없어서 음수도 나오고 100도 나옵니다. 이 날것의 점수를 **로짓(logit)** 이라고 부릅니다.

**왜 필요한가.** 모델이 확률을 바로 내놓게 하면 되지 않나 싶습니다. 두 가지 이유로 그러지 않습니다.

- `nn.Linear`는 실수를 내놓습니다. 0~1 사이로 묶이지도, 합이 1이 되지도 않습니다. 억지로 맞추려면 층을 하나 더 붙여야 합니다.
- 뒤에서 볼 이유로, **확률로 바꾸지 않은 상태가 손실 계산에 더 안전합니다.** 오늘 Step 5에서 숫자로 확인합니다.

**언제 쓰는가.** `nn.Linear(hidden, num_classes)`의 출력을 그대로 손실 함수에 넘길 때. 즉 분류에서는 거의 항상입니다.

**최소 예시.**

```python
model = nn.Linear(4, 3)     # 특징 4개 → 클래스 3개
x = torch.randn(2, 4)       # 샘플 2개
logits = model(x)

print(logits)               # 값의 범위에 제한이 없다
print(logits.shape)         # torch.Size([2, 3])  = (샘플 수, 클래스 수)
print(logits.sum(dim=1))    # 1이 아니다
```

이름의 유래는 통계학에 있지만 알 필요는 없습니다. 딥러닝에서 로짓은 그냥 **"softmax에 넣기 직전의 값"** 이라는 뜻으로 씁니다.

`(N, C)` — 행이 샘플, 열이 클래스. 오늘 내내 이 모양을 씁니다.

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `logits = model(x)` | 동일 | 분류 모델의 마지막 층 출력. 변수명도 보통 `logits` |
| (아무것도 안 붙임) | 마지막 층은 `nn.Linear`에서 끝난다 | `nn.Sequential(...)`의 맨 끝. **여기에 `nn.Softmax`를 붙이면 안 됩니다** |

### 개념 C (7분) · softmax — 점수를 확률로 바꾸는 규칙

**왜 이 식이 필요한가.** 로짓 `[2.0, 1.0, 0.1]`을 보면 "0번이 제일 그럴듯하다"까지는 압니다. 그런데 **얼마나** 그럴듯한지는 모릅니다. 확률로 바꾸려면 두 조건이 필요합니다. 전부 0 이상이고, 다 더하면 1이어야 합니다.

그냥 전체 합으로 나누면 될 것 같지만 안 됩니다. 로짓에 음수가 섞여 있으면 확률이 음수가 나오고, 합이 0이면 나눌 수도 없습니다. 그래서 **먼저 `exp`를 씌워 전부 양수로 만든 뒤** 합으로 나눕니다.

**식**

```
           exp(z_i)
  p_i = ───────────────                                  … (1)
         Σ  exp(z_j)
         j
```

**기호**

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| `z_i` | 제트 아이 | i번 클래스의 로짓 | `z[i]` |
| `C` | 씨 | 클래스 개수 | `z.shape[-1]` |
| `exp(z)` | 엑스포넨셜 제트 | 자연상수 e의 z제곱. **항상 양수** | `torch.exp(z)` |
| `Σ_j` | 시그마 제이 | j를 0부터 C−1까지 바꿔가며 전부 더한다 | `.sum()` |
| `p_i` | 피 아이 | i번 클래스의 확률 | `p[i]` |

**코드**

```python
z = torch.tensor([2.0, 1.0, 0.1])

p = torch.exp(z) / torch.exp(z).sum()
print(p)        # tensor([0.6590, 0.2424, 0.0986])
print(p.sum())  # tensor(1.0000)
```

**항 대조**

| 식 (1)의 부분 | 코드 |
|---|---|
| `exp(z_i)` | `torch.exp(z)` |
| `Σ_j exp(z_j)` | `torch.exp(z).sum()` |
| 나누기 | `/` |

**손계산으로 대조.** `exp(2.0) = 7.389`, `exp(1.0) = 2.718`, `exp(0.1) = 1.105`. 합은 `11.212`. 첫 칸은 `7.389 / 11.212 = 0.659`. 위 출력과 맞습니다.

**성질 하나 — 로짓 전체에 같은 수를 더해도 결과가 안 바뀝니다.**

```
      exp(z_i + c)        exp(c)·exp(z_i)        exp(z_i)
  ──────────────────  =  ──────────────────  =  ──────────      … (2)
   Σ exp(z_j + c)         exp(c)·Σ exp(z_j)      Σ exp(z_j)
   j                              j                    j
```

분자와 분모에 같은 `exp(c)`가 곱해져 약분됩니다. 오늘 Step 2에서 눈으로 보고, Step 5에서 이 성질을 실제로 써먹습니다.

**PyTorch에서는** `torch.softmax(z, dim=...)` 또는 `F.softmax(z, dim=...)`를 씁니다. `dim`은 축의 **번호**입니다(1주차 2일 종류②). 어느 축을 따라 합이 1이 되게 할지를 고르는 것이고, `(N, C)` 입력이면 클래스 방향인 `dim=1`(= `dim=-1`)이 맞습니다.

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `torch.exp(z) / torch.exp(z).sum()` | `torch.softmax(z, dim=1)` | **추론에서 확률이 필요할 때만** |
| 〃 | (학습 루프에는 안 나온다) | `CrossEntropyLoss`가 안에서 대신 한다 |

### 개념 D (6분) · 크로스 엔트로피 — 정답 칸의 확률에 −log

**왜 이 식이 필요한가.** 확률이 나왔으니 점수를 매겨야 합니다. 정답이 0번인데 모델이 0번에 `0.66`을 줬다면 꽤 맞힌 것이고, `0.01`을 줬다면 크게 틀린 겁니다. 그러니 **정답 칸의 확률이 크면 손실이 작고, 작으면 손실이 크게** 만들면 됩니다.

`1 − p`도 그런 성질이 있는데 왜 `−log p`일까요. `−log`는 확률이 0에 가까워질 때 **무한히 커집니다.** "정답에 0.01을 줬다"와 "0.0001을 줬다"의 차이를 100배가 아니라 훨씬 크게 벌려서, **확신에 찬 오답을 세게 벌합니다.**

**식**

```
  L = −log(p_t)                                          … (3)

                 exp(z_t)
  L = −log ───────────────────                           … (4)
              Σ exp(z_j)
              j
```

(4)는 (1)을 (3)에 그대로 대입한 것입니다.

**기호**

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| `t` | 티 | 정답 클래스의 **번호** (0부터) | `target` |
| `p_t` | 피 티 | 정답 클래스에 모델이 준 확률 | `p[target]` |
| `log` | 로그 | **자연로그**(밑이 e) | `torch.log` |
| `L` | 엘 | 손실값 | `loss` |

> 딥러닝에서 `log`는 언제나 자연로그입니다. 밑이 10인 상용로그가 아닙니다.

**코드**

```python
z = torch.tensor([2.0, 1.0, 0.1])
target = 0

p = torch.softmax(z, dim=0)
loss = -torch.log(p[target])
print(loss)     # tensor(0.4170)
```

**항 대조**

| 식 (3)의 부분 | 코드 |
|---|---|
| `p_t` | `p[target]` |
| `log` | `torch.log(...)` |
| 앞의 `−` | `-` |

**손계산으로 대조.** `p_0 = 0.659`, `ln(0.659) = −0.4170`, 음수를 붙여 `0.4170`. 맞습니다.

**정답 라벨은 one-hot 벡터가 아니라 번호 하나입니다.** 식 (3)이 정답 칸 하나만 보기 때문에, `[1, 0, 0]`을 넘겨봐야 나머지 0을 곱해 더하는 헛일이 됩니다. PyTorch의 `CrossEntropyLoss`는 정답을 `0`이라는 정수 하나로 받습니다.

**손실값 읽는 법 — 오늘 가장 오래 써먹을 한 줄입니다.**

아무것도 모르는 모델은 모든 클래스에 같은 확률 `1/C`를 줍니다. 그때 손실은 `−log(1/C) = log C`입니다.

| 클래스 수 | 무지한 모델의 손실 |
|---|---|
| 2 | `ln(2) ≈ 0.693` |
| 3 | `ln(3) ≈ 1.099` |
| 10 | `ln(10) ≈ 2.303` |
| 100 | `ln(100) ≈ 4.605` |

**학습 시작 직후 손실이 이 값 근처면 정상 출발입니다.** 훨씬 크면 초기화나 데이터 전처리가 잘못된 것이고, 처음부터 훨씬 작으면 라벨이 새어 들어갔을 가능성을 봅니다. 앞으로 모든 분류 학습에서 첫 손실값을 이 기준으로 읽습니다.

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `-torch.log(p[t])` | `criterion = nn.CrossEntropyLoss()` | 분류 모델 정의 직후 |
| 〃 | `loss = criterion(logits, y)` | 루프 안. **확률이 아니라 로짓을 넘깁니다** |

---

## 2교시 (53분) · 실습

노트북 `week02/d2_loss.ipynb`를 새로 만들고 Step 순서대로 셀을 채웁니다.

각 Step 끝의 **⟶ 한 줄로 적기**는 마크다운 셀에 직접 씁니다. 답을 보고 옮겨 적는 게 아니라 출력을 보고 자기 말로 꺼내는 게 목적이니, 문장이 어색해도 그대로 두세요.

### Step 1 (7분) — `nn.MSELoss`로 손실을 바꾸고 손계산과 대조하기

> **볼 것** — `nn.MSELoss()(pred, y)`와 `((pred - y) ** 2).mean()`이 **같은 숫자**라는 것, 그리고 `reduction`을 바꿨을 때 그 숫자가 어떻게 달라지는지
> **끝나면** — 남의 코드에서 `criterion = nn.MSELoss()`를 보고 그게 만드는 값을 손으로 재현할 수 있다
> **쓰는 상황** — 회귀(숫자를 맞히는) 문제의 학습 루프

| 새로 나온 것 | 하는 일 |
|---|---|
| `nn.MSELoss(reduction=...)` | 제곱오차를 하나로 줄인다. `'mean'`(기본) / `'sum'` / `'none'` |

**셀 1-1**

```python
import torch
import torch.nn as nn

pred = torch.tensor([2.0, 4.0, 6.0])
y    = torch.tensor([1.0, 4.0, 9.0])

print("손으로 :", ((pred - y) ** 2).mean().item())
print("mean   :", nn.MSELoss()(pred, y).item())
print("sum    :", nn.MSELoss(reduction='sum')(pred, y).item())
print("none   :", nn.MSELoss(reduction='none')(pred, y))
```

**셀 1-2** — 1주차 4일 학습 루프에서 손실 줄만 바꿉니다.

```python
torch.manual_seed(0)
x = torch.linspace(-3, 3, 100).view(-1, 1)
y = 3 * x + 2

model = nn.Linear(1, 1)
criterion = nn.MSELoss()

for i in range(100):
    pred = model(x)
    loss = criterion(pred, y)          # ← 손으로 쓰던 ((pred - y) ** 2).mean()
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad

print(model.weight.item(), model.bias.item())   # 3.0, 2.0 근처
```

⟶ **한 줄로 적기** — 여기서 `reduction='sum'`으로 바꾸면 같은 `lr=0.1`에서 왜 학습이 터지는지.

### Step 2 (10분) — 로짓에 softmax를 씌워 확률로 바꾸기

> **볼 것** — 로짓 전체에 100을 더했을 때 확률이 어떻게 되는지
> **끝나면** — `softmax`의 `dim`을 어느 축에 줘야 하는지 스스로 고를 수 있다

| 새로 나온 것 | 하는 일 |
|---|---|
| `torch.softmax(z, dim=n)` | n번 축을 따라 합이 1인 확률로 바꾼다 |
| `torch.allclose(a, b)` | 두 텐서가 오차 범위 안에서 같은지 True/False |

#### P · 예측 (2분) — 실행하기 전에 답을 마크다운 셀에 적으세요

```python
z = torch.tensor([2.0, 1.0, 0.1])

p1 = torch.softmax(z, dim=0)
p2 = torch.softmax(z + 100, dim=0)

print(p1)
print(p2)
print(torch.allclose(p1, p2))
```

**`p2`는 어떻게 나올까요?**

- **A.** `p1`과 같다 — `allclose`가 True
- **B.** 0번 클래스가 1.0에 가깝게 쏠린다 — 로짓이 커졌으므로
- **C.** 전부 `nan`이 된다 — `exp(102)`가 넘치므로
- **D.** 합이 1이 아니게 된다

#### R · 실행 (1분)

셀을 실행하고 예측과 대조합니다. 틀렸다면 그 지점이 오늘 가장 중요한 부분입니다.

#### I · 조사 (4분)

중간 상태를 직접 찍어봅니다.

```python
print(torch.exp(z))         # [7.39, 2.72, 1.11]
print(torch.exp(z + 100))   # ???
```

`exp` 단계만 떼어내면 값이 넘칩니다. 그런데 `torch.softmax`는 멀쩡했습니다. PyTorch가 안에서 **식 (2)를 써서 최댓값을 먼저 빼기** 때문입니다. Step 5에서 이걸 직접 재현합니다.

#### M · 수정 (3분)

`dim`을 바꿔보고 어느 방향의 합이 1이 되는지 확인합니다.

```python
Z = torch.tensor([[2.0, 1.0, 0.1],
                  [0.5, 2.5, 1.0]])      # 샘플 2개, 클래스 3개

print(torch.softmax(Z, dim=0).sum(dim=0))
print(torch.softmax(Z, dim=1).sum(dim=1))
```

행이 샘플이니 클래스 방향은 `dim=1`입니다. `dim=0`으로 쓰면 **샘플들끼리 확률을 나눠 갖는** 엉뚱한 계산이 되는데, 에러는 안 납니다.

⟶ **한 줄로 적기** — `dim=0`으로 잘못 썼을 때 1이 되는 건 무엇인지.

### Step 3 (14분) — `CrossEntropyLoss`에 로짓을 넣고 손계산과 대조하기

> **볼 것** — 로짓을 넣었을 때와 softmax를 먼저 씌워 넣었을 때의 **두 숫자**
> **끝나면** — `CrossEntropyLoss`가 안에서 무엇을 하는지 두 단계로 쪼개 말할 수 있다
> **쓰는 상황** — 모든 분류 학습 루프

| 새로 나온 것 | 하는 일 |
|---|---|
| `nn.CrossEntropyLoss()` | **로짓**과 정답 번호를 받아 크로스 엔트로피를 낸다 |
| `F.log_softmax(z, dim)` | `log(softmax(z))`를 한 번에, 안전하게 |
| `F.nll_loss(logp, t)` | `logp`에서 정답 칸을 골라 부호를 뒤집는다 |

#### P · 예측 (3분) — 실행 전에 적으세요

개념 D에서 손계산한 값은 `0.4170`이었습니다.

```python
z = torch.tensor([[2.0, 1.0, 0.1]])     # 샘플 1개, 클래스 3개
t = torch.tensor([0])                   # 정답은 0번

ce = nn.CrossEntropyLoss()

a = ce(z, t)                            # 로짓 그대로
b = ce(torch.softmax(z, dim=1), t)      # softmax를 먼저 씌워서

print(a.item(), b.item())
```

**`0.4170`이 나오는 쪽은 어디이고, 다른 쪽은 어떻게 될까요?**

- **A.** `a`가 `0.4170`, `b`는 에러가 난다
- **B.** `a`가 `0.4170`, `b`는 에러 없이 그보다 큰 다른 값이 나온다
- **C.** `b`가 `0.4170`, `a`는 그보다 큰 다른 값이 나온다
- **D.** 둘 다 `0.4170` — `CrossEntropyLoss`가 알아서 처리한다

#### R · 실행 (1분)

#### I · 조사 (5분)

`CrossEntropyLoss`를 한 겹씩 벗겨봅니다.

```python
import torch.nn.functional as F

logp = F.log_softmax(z, dim=1)
print(logp)                    # log(확률) 세 개

print(F.nll_loss(logp, t))     # 정답 칸을 골라 부호를 뒤집는다
print(ce(z, t))                # 같은 값
```

```
CrossEntropyLoss  =  log_softmax  +  nll_loss
                     (확률로 + log)  (정답 칸 고르고 −)
```

`nll_loss`는 이름에 loss가 붙어 있지만 계산은 "골라서 음수 붙이기"가 전부입니다. **확률로 바꾸는 일은 `log_softmax`가 이미 했습니다.** 그래서 여기에 넘겨야 하는 건 로짓입니다. 이미 softmax를 씌운 값을 넣으면 softmax가 두 번 들어가고, 확률 벡터는 값이 0~1로 눌려 있어서 두 번째 softmax를 지나면 더 평평해집니다. 결과적으로 손실이 실제보다 크게 나옵니다.

**에러가 안 나는 게 이 함정의 핵심입니다.** 학습은 돌아가는데 잘 안 되는 상태가 됩니다.

#### M · 수정 (5분)

샘플을 4개로 늘려 shape 규약을 확인합니다.

```python
torch.manual_seed(0)
z4 = torch.randn(4, 3)                  # (N, C)
t4 = torch.tensor([0, 2, 1, 1])         # (N,)  dtype은 long

print(ce(z4, t4))

# 손계산으로 대조
p4 = torch.softmax(z4, dim=1)
manual = -torch.log(p4[torch.arange(4), t4]).mean()
print(manual)
```

`ce`는 샘플마다 손실을 내고 **평균**합니다(기본 `reduction='mean'`). 입력은 `(N, C)` float, 타깃은 `(N,)` long.

⟶ **한 줄로 적기** — `nll_loss`가 실제로 하는 일을 자기 말로.

### Step 4 (9분) — 그래디언트가 `p − y`인지 숫자로 확인하기

> **볼 것** — `z.grad`와 `softmax(z) − onehot`이 **같은 값**인 것
> **끝나면** — 분류에 MSE를 쓰지 않는 이유를 그래디언트 숫자로 설명할 수 있다
> **쓰는 상황** — 학습이 안 될 때 "그래디언트가 흐르긴 하는가"를 보는 진단 (4주차에 정식으로)

softmax와 크로스 엔트로피를 붙여 놓으면 로짓에 대한 기울기가 놀랄 만큼 단순해집니다.

```
   ∂L
  ────  =  p_i − y_i                                     … (5)
   ∂z_i
```

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| `∂L/∂z_i` | 라운드 엘 라운드 제트 아이 | `z_i`를 조금 늘렸을 때 손실이 얼마나 늘어나는가 | `z.grad[i]` |
| `y_i` | 와이 아이 | 정답이면 1, 아니면 0 | `onehot[i]` |

**모델이 준 확률에서 정답을 빼면 그게 그대로 기울기입니다.** 이 단순함이 두 함수를 같이 쓰는 이유 중 하나입니다.

**셀 4-1**

```python
z = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
t = torch.tensor([0])

loss = nn.CrossEntropyLoss()(z, t)
loss.backward()

print("실측 :", z.grad)

p = torch.softmax(z.detach(), dim=1)
onehot = torch.zeros_like(p)
onehot[0, t] = 1.0
print("식(5):", p - onehot)
```

샘플이 1개라 `reduction='mean'`의 나눗셈이 1로 떨어집니다. 샘플이 N개면 `(p − onehot) / N`이 `.grad`입니다.

**셀 4-2** — 크게 틀렸을 때 MSE와 비교합니다.

```python
z_bad = torch.tensor([[-10.0, 10.0]], requires_grad=True)   # 정답은 0번인데 1번에 몰빵
t2 = torch.tensor([0])

nn.CrossEntropyLoss()(z_bad, t2).backward()
print("CE  :", z_bad.grad)

z_bad.grad = None
p = torch.softmax(z_bad, dim=1)
target_onehot = torch.tensor([[1.0, 0.0]])
nn.MSELoss()(p, target_onehot).backward()
print("MSE :", z_bad.grad)
```

CE는 기울기가 ±1 근처로 나오고, MSE는 거의 0입니다. **가장 크게 틀렸을 때 MSE가 가장 적게 배웁니다.** softmax가 양 끝에서 평평해지는데 MSE의 미분이 거기에 또 곱해지기 때문입니다. 이것이 분류에 MSE를 쓰지 않는 이유입니다.

⟶ **한 줄로 적기** — MSE 쪽 기울기가 0에 가까운 이유를 자기 말로.

### Step 5 (9분) — 로짓을 1000으로 키워 직접 만든 softmax를 터뜨리기

> **볼 것** — 식 (1)을 그대로 옮긴 코드는 `nan`, `torch.softmax`는 멀쩡한 숫자
> **끝나면** — "`CrossEntropyLoss`는 왜 softmax를 품고 있나"에 이유 두 개를 댈 수 있다
> **쓰는 상황** — 학습 중 손실이 갑자기 `nan`이 될 때 후보를 좁히는 근거

**셀 5-1**

```python
z_big = torch.tensor([[1000.0, 999.0, 998.0]])
t = torch.tensor([0])

p_naive = torch.exp(z_big) / torch.exp(z_big).sum()     # 식 (1) 그대로
print("직접    :", p_naive)
print("직접 CE :", -torch.log(p_naive[0, 0]))

print("torch   :", torch.softmax(z_big, dim=1))
print("torch CE:", nn.CrossEntropyLoss()(z_big, t))
```

float32가 담을 수 있는 최댓값은 약 `3.4e38`인데 `exp(1000)`은 그보다 훨씬 큽니다. 분자도 `inf`, 분모도 `inf`가 되고 `inf / inf = nan`입니다.

**셀 5-2** — PyTorch가 하는 일을 손으로 재현합니다.

```
             exp(z_i − max(z))
  p_i = ──────────────────────────                       … (6)
          Σ  exp(z_j − max(z))
          j
```

```python
z_shift = z_big - z_big.max()
p_safe = torch.exp(z_shift) / torch.exp(z_shift).sum()

print(p_safe)
print(torch.softmax(z_big, dim=1))
print(torch.allclose(p_safe, torch.softmax(z_big, dim=1)))
```

최댓값이 0이 되므로 `exp`의 결과가 절대 1을 넘지 않습니다. 그리고 식 (2) 덕분에 **빼기 전과 결과가 같습니다.** 안전하게 계산하려고 값을 바꾼 게 아니라, 바꿔도 되는 걸 알고 바꾼 겁니다.

여기에 더해, `log(softmax(...))`를 두 단계로 나눠 하면 확률이 아주 작을 때 `log(0) = -inf`가 또 한 번 터집니다. `log_softmax`는 두 계산을 한 식으로 합쳐 그것도 피합니다.

**`CrossEntropyLoss`가 softmax를 품고 있는 두 이유가 이걸로 다 나왔습니다.**

1. **수치 안정성** — 최댓값 빼기와 `log_softmax`를 안에서 처리한다 (Step 5)
2. **기울기가 단순해진다** — `p − y` 한 줄 (Step 4)

⟶ **한 줄로 적기** — 최댓값을 빼도 결과가 같은 근거가 몇 번 식인지, 그리고 왜인지.

### 실전 대응 (4분)

오늘 손으로 쓴 것이 실제 학습 스크립트의 어느 줄이 되는지 정리합니다.

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `((pred - y) ** 2).mean()` | `nn.MSELoss()` | 회귀 학습 루프 |
| `-torch.log(softmax(z)[arange(N), t]).mean()` | `nn.CrossEntropyLoss()` | 모든 분류 학습 루프 |
| `F.log_softmax` + `F.nll_loss` | `nn.CrossEntropyLoss()` 한 줄 | 같은 계산의 두 가지 표기 |
| `torch.softmax(z, dim=1)` | **학습 루프에는 안 나온다** | 추론에서 확률이 필요할 때만 |

---

## 3교시 (20분) · 실패 케이스

아래 두 코드를 **먼저** 고쳐 보고, 그다음에 표를 읽습니다. 각 과제마다 **"어느 출력을 보고 그렇게 판단했는가"** 를 한 줄로 적으세요.

### 과제 1 (7분) · 손실이 제자리에서 안 움직인다

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(256, 8)
t = torch.randint(0, 10, (256,))      # 클래스 10개

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
    nn.Softmax(dim=1),
)
criterion = nn.CrossEntropyLoss()

for i in range(300):
    loss = criterion(model(x), t)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.5 * p.grad
    if i % 60 == 0:
        print(i, round(loss.item(), 4))
```

에러는 안 납니다. 손실도 찍힙니다. 그런데 출력이 이렇습니다.

```
0   2.3036
60  2.2961
120 2.2867
180 2.2713
240 2.2408
```

- 어디를 고쳐야 합니까?
- **진단 근거 한 줄** — 어느 출력의 어떤 점을 보고 그렇게 판단했습니까? (개념 D의 표를 다시 보세요. 그리고 `model(x)`를 한 줄 찍어보면 확정됩니다)

### 과제 2 (7분) · 손실이 27.5에서 멈춘다

```python
torch.manual_seed(0)
x = torch.linspace(-3, 3, 100).view(-1, 1)   # (100, 1)
y = (3 * x + 2).view(-1)                     # (100,)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()

for i in range(100):
    pred = model(x)                          # (100, 1)
    loss = criterion(pred, y)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
    if i % 25 == 0:
        print(i, round(loss.item(), 4))

print(model.weight.item(), model.bias.item())
```

에러는 안 나고 경고만 뜹니다. 출력은 이렇습니다.

```
0  29.6876
25 27.5455
50 27.5455
75 27.5455
-1.3864012160524908e-08 1.999999761581421
```

`b`는 2.0을 찾았는데 `w`가 0으로 죽었습니다.

- 어디를 고쳐야 합니까?
- **진단 근거 한 줄** — 무엇을 한 줄 찍어보면 확정할 수 있습니까? (경고 메시지를 끝까지 읽어 보세요)

### 실패 케이스 표 (6분)

| 증상 | 원인 | 해결 |
|---|---|---|
| 손실이 `ln(C)` 근처에서 거의 안 움직인다 | 모델 끝에 `nn.Softmax`를 붙인 채 `CrossEntropyLoss`에 넣음 (이중 softmax) | 마지막 층은 `nn.Linear`에서 끝낸다. 로짓을 그대로 넘긴다 |
| `RuntimeError: expected scalar type Long but found Float` | 타깃을 float이나 one-hot으로 넘김 | 타깃은 클래스 번호 `(N,)`, `dtype=torch.long` |
| `ValueError: Expected input batch_size (N) to match target batch_size (M)` | 입력이 `(N, C)`가 아니거나 타깃이 `(N,)`가 아님 | `.shape`을 찍어 `(N,C)` / `(N,)` 확인 |
| MSE 손실이 이상하게 크고 학습이 안 된다 (경고만) | `(N,1)` vs `(N,)`이 브로드캐스팅되어 `(N,N)` 비교가 됨 | `y.view(-1, 1)`로 맞춘다. **경고를 무시하지 않는다** |
| 손실이 `nan` | 직접 만든 `softmax`+`log`에서 오버플로 또는 `log(0)` | `CrossEntropyLoss` 또는 `F.log_softmax`를 쓴다 |
| `reduction='sum'`으로 바꿨더니 발산 | 손실이 배치 크기만큼 커져 그래디언트도 그만큼 커짐 | `'mean'`을 쓰거나 lr을 배치 크기로 나눈다 |
| 분류 정확도는 멀쩡한데 손실만 이상하다 | 손실을 텐서째 리스트에 쌓아 그래프가 살아 있음 | `.item()`으로 숫자만 담는다 (2주차 1일 기준값 ⑤) |

---

## 마무리 (15분)

### 통과 기준

전부 노트북 출력으로 확인 가능해야 합니다.

1. `nn.MSELoss()(pred, y)`가 `((pred - y) ** 2).mean()`과 같은 값으로 찍힘
2. `torch.softmax` 결과의 합이 `1.0`이고, `z`와 `z + 100`의 softmax가 `allclose`로 `True`
3. `nn.CrossEntropyLoss()(z, t)` 값이 `-torch.log(torch.softmax(z, dim=1)[0, t])` 손계산과 소수점 4자리까지 일치
4. 같은 로짓에 softmax를 먼저 씌워 넣었을 때 손실값이 **다르게** 나오는 것을 두 숫자로 확인
5. `z.grad`가 `softmax(z) − onehot`과 일치
6. 로짓 `1000`에서 직접 구현한 softmax는 `nan`, `torch.softmax`/`CrossEntropyLoss`는 정상값

### 실전 대응표 — 오늘까지의 누적

| 손으로 쓰던 것 | 실전 | 채운 세션 |
|---|---|---|
| `w`, `b` 텐서 직접 선언 | `nn.Linear` | 2주차 1일 |
| `pred = x @ w + b` | `model(x)` | 2주차 1일 |
| `w.grad.zero_()` | `model.zero_grad()` | 2주차 1일 |
| `((pred - y) ** 2).mean()` | `nn.MSELoss()` | **2주차 2일** |
| (분류는 오늘 처음) | `nn.CrossEntropyLoss()` | **2주차 2일** |
| `w -= lr * w.grad` | `optimizer.step()` | 2주차 3일 |
| `model.zero_grad()` | `optimizer.zero_grad()` | 2주차 3일 |
| 데이터 통째로 넣기 | `DataLoader` | 2주차 4일 |

### 확인 퀴즈 6문항

답을 채팅으로 보내주세요. 정답과 해설은 그때 공개합니다.

**퀴즈 1.** 클래스가 10개인 분류에서 다음과 같이 짰습니다.

```python
model = nn.Sequential(nn.Linear(8, 10), nn.Softmax(dim=1))
loss = nn.CrossEntropyLoss()(model(x), t)
```

학습을 한참 돌렸을 때 손실값은 어떻게 될까요?

- **A.** 0에 가깝게 떨어진다
- **B.** `ln(10) ≈ 2.303`보다 눈에 띄게 큰 값에서 머문다
- **C.** `ln(10) ≈ 2.303`에 거의 붙어서 잘 안 움직인다
- **D.** `RuntimeError`가 난다

**퀴즈 2.** `z = [2.0, 1.0, 0.1]`과 `z2 = [12.0, 11.0, 10.1]`에 각각 `torch.softmax(·, dim=0)`을 하면?

- **A.** `z2` 쪽이 0번 클래스에 더 쏠린다
- **B.** 두 결과가 같다
- **C.** `z2` 쪽은 합이 1을 넘는다
- **D.** `z2` 쪽은 `inf`가 나온다

**퀴즈 3.** `z = torch.tensor([[0.0, 0.0]], requires_grad=True)`, 정답 `t = torch.tensor([0])`으로 `CrossEntropyLoss`를 계산하고 `backward()`를 불렀습니다. `z.grad`는?

- **A.** `[[0.0, 0.0]]`
- **B.** `[[-0.5, 0.5]]`
- **C.** `[[0.5, -0.5]]`
- **D.** `[[-1.0, 1.0]]`

**퀴즈 4.** 클래스 100개짜리 분류 학습을 막 시작했습니다. 첫 손실값이 어느 정도일 때 "정상적인 출발"입니까?

- **A.** `0.01` 근처
- **B.** `1.0` 근처
- **C.** `4.6` 근처
- **D.** `100` 근처

**퀴즈 5.** `pred.shape`이 `(100, 1)`, `y.shape`이 `(100,)`인 상태로 `nn.MSELoss()(pred, y)`를 부르면?

- **A.** `RuntimeError`가 난다
- **B.** 경고만 뜨고, 내부적으로 `(100, 100)` 크기의 차이를 계산해 엉뚱한 값이 나온다
- **C.** 경고 없이 정상 동작한다 — 원소 수가 같으므로
- **D.** PyTorch가 자동으로 `(100, 1)`에 맞춰 준다

**퀴즈 6.** 샘플 100개짜리 회귀 학습에서 `nn.MSELoss()`를 `nn.MSELoss(reduction='sum')`으로 바꾸고 나머지는 그대로 뒀습니다.

- **A.** 손실값만 100배가 되고 학습 결과는 같다
- **B.** 그래디언트도 100배가 되어 같은 `lr`에서 발산할 수 있다
- **C.** `backward()`에서 에러가 난다 — 손실이 스칼라가 아니므로
- **D.** 아무 차이 없다

---

## 다음 세션 예고

**2주차 3일 · 옵티마이저.** 오늘까지 손으로 쓰던 `p -= lr * p.grad` 줄이 `optimizer.step()`으로 바뀝니다. SGD → momentum → Adam 순서로 **왜 각 단계가 필요했는지**를 같은 문제에 하나씩 얹어 가며 봅니다. 난이도와 수식 비중은 오늘과 비슷합니다.

그리고 2주차 1일에 짚었던 함정이 그대로 재현됩니다 — `SGD(model.parameters(), ...)`에 넘어가는 건 **등록된 것뿐**입니다.
