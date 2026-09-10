# 1주차 3일 · autograd

> 배정 시간 **2시간** · 산출물 `week01/d3_autograd.ipynb`

---

## 도입 (5분) · 누적 복습

### ① 복습 퀴즈 (1분)

**답을 찾지 말고 기억에서 꺼낸다.** 2일차 자료를 열지 않는다.

```python
a = torch.arange(6.).reshape(2, 3)
b = a.t().reshape(6)
c = a.reshape(6)
```

`b`와 `c` 중 **원본 `a`와 메모리를 공유하는 것**은?

- (A) `b`만
- (B) `c`만
- (C) 둘 다
- (D) 둘 다 아니다

답을 적고 넘어간다. 채점은 ③에서 한다.

### ② 미해결 항목 — VRAM 측정값 (2분)

2일차 Step 5-3에서 `텐서 1개` 증가분이 0.00 MiB로 찍힌 건 셀 재실행 때문으로 추정했다. 확정하고 닫는다.

**커널을 재시작한 직후, 아래 셀을 딱 한 번만 실행한다.**

```python
import torch

base = torch.cuda.memory_allocated()
batch = torch.randn(32, 3, 224, 224, device="cuda")
delta = torch.cuda.memory_allocated() - base
print(f"{delta / 1024**2:.2f} MiB")   # 18.38 이 나와야 정상
```

18.38이 나오면 원인 추정이 맞았던 것이다. 다시 0.00이 나오면 다른 원인이니 채팅으로 알려달라.

> **오늘 이후의 규칙:** 메모리를 재는 셀은 커널 재시작 직후 1회만 실행한다. Step 5에서 다시 쓴다.

### ③ 채점과 오늘로의 연결 (2분)

정답은 **(B)**. `a.t()`는 메모리가 연속이 아니라 `reshape`이 복사본을 만들고, `a`는 연속이라 원본을 공유한다. 판별은 `.data_ptr()` 비교다.

```python
print(b.data_ptr() == a.data_ptr())   # False
print(c.data_ptr() == a.data_ptr())   # True
```

**오늘 이게 왜 다시 나오는가** — 복사가 일어나면 그래디언트가 흐르는 경로가 한 단계 더 늘어난다. Step 2에서 `grad_fn`으로 그 경로를 눈으로 확인한다.

---

## 오늘 얻는 것 (2분)

| 배우는 것 | 이걸로 할 수 있게 되는 일 | 중요도 |
|---|---|---|
| `requires_grad` | 어느 파라미터를 학습시키고 어느 것을 얼릴지 코드로 정한다 | ★★★ 필수 |
| `backward()` · `.grad` | 남의 학습 코드를 열었을 때 각 줄이 무슨 일을 하는지 읽힌다 | ★★★ 필수 |
| 그래디언트 누적 | 손실이 스텝마다 커질 때 원인을 30초에 짚는다 | ★★★ 필수 |
| `no_grad()` | 평가·추론에서 메모리가 새는 것을 막는다 | ★★★ 필수 |
| `grad_fn` 읽기 | 학습이 안 될 때 그래프가 끊겼는지 확인한다 | ★★ 권장 · **9주차 파인튜닝 전까지** |
| 그래프의 메모리 비용 | 배치 크기를 왜 줄여야 하는지 숫자로 안다 | ★★ 권장 · **7주차 실학습 전까지** |
| `retain_grad()` | 중간 결과의 그래디언트를 꺼내 본다 | ★ 참고 |
| `backward(gradient=)` | 스칼라가 아닌 출력을 되짚는다 | ★ 참고 · **4주차 2일에 정식으로 다룬다** |

**★★★ 넷은 오늘 안에 손에 붙어야 합니다.** 4일차 학습 루프가 이 넷으로만 이뤄져 있어서, 하나라도 비면 다음 세션이 막힙니다.

**시간이 부족하면 ★ 항목부터 버립니다.** Step 2의 `retain_grad()`(셀 5)와 Step 6의 `backward(gradient=)`(셀 17)가 해당합니다. 오늘 통과 기준에 들어있지 않습니다.

---

## 1교시 (25분) · 개념 4개

오늘 처음 나오는 용어는 네 개다. 하나씩 끊어서 본다.

### 개념 A (6분) · 그래디언트와 `requires_grad`

**1. 무엇인가**

그래디언트는 **"이 숫자를 조금 올리면 결과가 얼마나 변하는가"**를 나타내는 값이다.

**2. 왜 필요한가**

모델은 결국 숫자 수백만 개의 묶음이고, 학습은 그 숫자를 조금씩 바꿔 오차를 줄이는 과정이다. 그런데 어느 방향으로 바꿔야 오차가 줄어드는지 모르면 아무것도 못 한다.

숫자 하나당 "올리면 오차가 커지나 작아지나"를 알면 방향이 정해진다. 커진다면 내리고, 작아진다면 올린다. 그게 그래디언트다.

문제는 대상이 수백만 개라는 것. 하나씩 손으로 미분할 수 없다. **PyTorch가 이걸 자동으로 해주는 기능이 autograd다.**

**3. 언제 쓰는가**

추적할 대상에 `requires_grad=True` 표시를 달아둔다. 오늘은 손으로 붙여가며 동작을 본다.

**4. 최소 예시**

```python
x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)   # True

y = torch.tensor(2.0)
print(y.requires_grad)   # False — 기본값
```

**5. 실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `torch.tensor(2.0, requires_grad=True)` | `nn.Linear(10, 5)` | 모델을 만드는 순간 |
| 직접 `requires_grad=True` 지정 | 자동으로 붙어 있음 | `nn.Parameter`가 대신 처리 |
| — | `for p in model.parameters(): p.requires_grad = False` | 파인튜닝에서 백본을 얼릴 때 |

세 번째 줄이 9주차 CNN 파인튜닝의 핵심 조작이다. 오늘 배우는 표시 하나를 껐다 켜는 것이 전이학습의 전부라고 해도 된다.

> **정수 텐서에는 붙지 않는다.** `torch.tensor(2, requires_grad=True)`는 에러다. 미분은 실수 위에서만 정의된다. 3교시 실패 케이스에 다시 나온다.

### 개념 B (6분) · 계산 그래프

**1. 무엇인가**

계산 그래프는 **"이 결과가 어떤 계산을 거쳐 나왔는지"를 기록한 목록**이다.

**2. 왜 필요한가**

`z = (x + 2) * 3`을 계산하면 보통은 결과 숫자만 남는다. 그런데 x의 그래디언트를 구하려면 z가 어떻게 만들어졌는지 거꾸로 되짚어야 한다. 곱하기가 마지막이었고 그 앞에 더하기가 있었다는 정보가 필요하다.

PyTorch는 계산을 앞으로 진행할 때 이 기록을 자동으로 남긴다.

**3. 언제 쓰는가**

직접 만들지 않는다. `requires_grad=True`인 텐서로 연산하면 저절로 쌓인다. 대신 **`grad_fn`으로 들여다보는 것**은 자주 한다.

**4. 최소 예시**

```python
x = torch.tensor(2.0, requires_grad=True)
z = (x + 2) * 3

print(z.grad_fn)                 # <MulBackward0 ...> — 마지막 연산이 곱하기
print(z.grad_fn.next_functions)  # 그 앞에 AddBackward0
```

두 종류의 텐서를 구분해두자.

| 종류 | 뜻 | `grad_fn` |
|---|---|---|
| **리프(leaf)** | 내가 직접 만든 텐서 | `None` |
| 중간 결과 | 연산으로 나온 텐서 | 연산 종류가 들어있음 |

**5. 실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `print(z.grad_fn)` | `print(loss.grad_fn)` | **손실이 안 떨어질 때 첫 번째로 보는 곳** |
| `x.is_leaf` 확인 | 파라미터인지 중간값인지 판별 | 그래디언트가 안 흐를 때 |

`loss.grad_fn`이 `None`이면 그래프가 끊긴 것이다. 학습이 전혀 안 되는데 에러도 안 나는 상황의 대표적 원인이고, 1주차 4일에 바로 써먹는다.

### 개념 C (7분) · `backward()`와 `.grad`, 그리고 누적

**1. 무엇인가**

`backward()`는 **계산 그래프를 거꾸로 훑어 그래디언트를 계산하는 명령**이다. 결과는 각 텐서의 `.grad`에 들어간다.

**2. 왜 필요한가**

개념 A에서 그래디언트가 필요하다고 했고, B에서 되짚을 기록이 쌓인다고 했다. `backward()`가 실제로 되짚는 실행 버튼이다. 부르지 않으면 `.grad`는 계속 `None`이다.

**3. 언제 쓰는가**

학습 루프에서 매 스텝 `loss.backward()` 한 줄로 나온다.

**4. 최소 예시**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2          # y = x²
y.backward()        # dy/dx = 2x = 6
print(x.grad)       # tensor(6.)
```

**여기서 오늘 가장 중요한 성질.** `.grad`는 **덮어쓰지 않고 더한다.** Step 3에서 직접 확인한다.

**5. 실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `x.grad.zero_()` | `optimizer.zero_grad()` | 학습 루프 첫 줄 |
| `y.backward()` | `loss.backward()` | 그다음 줄 |
| `w -= lr * w.grad` | `optimizer.step()` | 그다음 줄 |

2주차 3일에 옵티마이저를 배울 때 **이 세 줄이 왜 항상 이 순서인지** 이미 알고 있는 상태가 된다.

### 개념 D (6분) · `no_grad()`와 `detach()`

**1. 무엇인가**

`no_grad()`는 **"이 구역에서는 계산 그래프를 만들지 마라"**는 지시다. `detach()`는 **한 텐서를 그래프에서 떼어내는 것**이다.

**2. 왜 필요한가**

그래프를 쌓는 건 공짜가 아니다. 되짚기 위해 중간 결과를 메모리에 붙잡아둔다. 학습할 때는 필요하지만 추론만 할 때는 전부 낭비다. Step 5에서 이 비용을 숫자로 잰다.

**3. 언제 쓰는가**

세 곳에서 거의 항상 쓴다.

- 추론·평가할 때
- 파라미터를 직접 갱신할 때 — 갱신 자체는 학습 대상이 아니다
- 손실값을 로그로 모을 때 — 텐서째 쌓으면 그래프가 통째로 남는다

**4. 최소 예시**

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3
print(y.requires_grad, y.grad_fn)   # False None — 그래프가 없다

z = (x * 3).detach()
print(z.requires_grad)              # False
```

**5. 실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `with torch.no_grad():` | 동일 | 평가 루프 전체를 감싼다 |
| `.detach()` / `.item()` | `losses.append(loss.item())` | 손실 기록할 때 |
| `with torch.no_grad(): w -= ...` | `optimizer.step()` 내부 | 옵티마이저가 대신 처리 |

두 번째 줄을 빼먹으면 **에폭이 돌수록 VRAM이 조금씩 늘다가 OOM**이 난다. 원인 찾기 어려운 유형이라 3교시 표에 넣어뒀다.

---

## 2교시 (53분) · 실습

`week01/d3_autograd.ipynb`를 새로 만든다. **커널을 재시작한 상태에서 시작한다.**

> **Step 3과 Step 5는 예측부터 한다.** 코드를 먼저 읽고 결과를 마크다운 셀에 적은 뒤 실행한다. 예측이 틀리는 지점이 오늘 가장 많이 남는 부분이다.

### Step 1 — 첫 `backward` (4분)

> **볼 것** — `backward()`를 부르기 **전에는** `.grad`가 `None`이라는 것. 그리고 부른 **직후** 값이 채워진다는 것.
> **끝나면** — 손으로 계산한 미분값과 `x.grad`를 대조할 수 있다.

**셀 1**

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print("y      :", y)
print("grad_fn:", y.grad_fn)
print("x.grad :", x.grad)     # 아직 None
```

**셀 2**

```python
y.backward()
print("x.grad :", x.grad)     # 2x = 6
```

`y = x²`이면 `dy/dx = 2x`, x가 3이니 6. **손계산값과 대조**하는 방식으로 오늘 내내 진행한다.

### Step 2 — 계산 그래프 들여다보기 (7분)

> **볼 것** — `grad_fn`에 찍히는 연산 이름이 **내가 쓴 연산의 역순**이라는 것. 그리고 리프에는 `grad_fn`이 없다는 것.
> **끝나면** — 그래디언트가 안 흐를 때 `grad_fn`이 `None`인지 먼저 확인하는 습관이 생긴다.

세 줄짜리 계산을 만들어 그래프가 어떻게 쌓이는지 본다. 계산 자체는 `c = (x·w + 1)²`으로 단순하다. **숫자를 따라가는 게 목적이 아니라 기록이 어떻게 남는지를 보는 것**이다.

**셀 3**

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)

a = x * w        # 8
b = a + 1        # 9
c = b ** 2       # 81

print("c        :", c.item())
print("c.grad_fn:", c.grad_fn)
print("b.grad_fn:", b.grad_fn)
print("a.grad_fn:", a.grad_fn)
print("x.grad_fn:", x.grad_fn)   # None — 리프
```

**셀 4**

```python
print("x.is_leaf:", x.is_leaf)
print("a.is_leaf:", a.is_leaf)
c.backward()
print("x.grad:", x.grad)   # dc/dx = 2b·w = 2·9·4 = 72
print("w.grad:", w.grad)   # dc/dw = 2b·x = 2·9·2 = 36
print("a.grad:", a.grad)   # None + 경고
```

`a.grad`가 `None`인 게 정상이다. **중간 결과의 그래디언트는 계산 도중 쓰이고 버려진다.** 저장하면 메모리만 먹고, 학습에 필요한 건 리프의 그래디언트뿐이다.

굳이 봐야 하면 순전파 **전에** 표시를 달아둔다.

**셀 5**

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)
a = x * w
a.retain_grad()          # backward보다 앞에 있어야 한다
c = (a + 1) ** 2
c.backward()
print("a.grad:", a.grad)  # dc/da = 2b = 18
```

### Step 3 — 그래디언트 누적 (12분) ★

#### ① 예측 (2분)

**아직 실행하지 마세요.** 읽고 출력을 먼저 적는다.

```python
x = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"{i}회차 x.grad = {x.grad.item()}")
```

`y = x²`, `x = 3`이니 미분값은 `2x = 6`이다. 3회 반복하면 무엇이 찍힐까.

- (A) `6, 6, 6`
- (B) `6, 12, 18`
- (C) `6, 0, 0`
- (D) 에러

노트북 마크다운 셀에 **예측을 적고** 넘어간다.

#### ② 실행 (1분)

**셀 6** — 위 코드를 그대로 실행한다. 예측과 맞았나?

#### ③ 조사 (4분)

`6, 12, 18`이 나온다. `.grad`는 덮어쓰지 않고 **더한다.** 언제 생기고 언제 더해지는지 한 단계씩 확인한다.

**셀 7**

```python
x = torch.tensor(3.0, requires_grad=True)
print("초기        :", x.grad)   # None
y = x ** 2
print("순전파 후   :", x.grad)   # None ← 아직 없음
y.backward()
print("1회 backward:", x.grad)   # tensor(6.)
y = x ** 2
y.backward()
print("2회 backward:", x.grad)   # tensor(12.) ← 새로 만들지 않고 더함
```

왜 이렇게 만들었을까. 배치가 커서 GPU에 한 번에 안 올라갈 때, 조각내어 각각 `backward()`를 부르고 그래디언트를 누적한 뒤 한 번에 갱신하는 기법이 있다. 이 설계가 그걸 가능하게 한다. 파인튜닝 구간에서 실제로 쓴다.

#### ④ 수정 (3분)

**셀 8** — `6, 6, 6`이 나오도록 셀 6을 고친다. 힌트는 `.zero_()` 한 줄.

```python
x = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    # 여기에 한 줄
    y = x ** 2
    y.backward()
    print(f"{i}회차 x.grad = {x.grad.item()}")
```

> 첫 회차에는 `.grad`가 `None`이라 그대로 `.zero_()`를 부르면 에러다. `if x.grad is not None:` 조건이 필요하다.

고쳤으면 하나 더. **`zero_()`를 `backward()` 뒤로 옮기면** 어떻게 될까. 예측하고 실행한다. 실전에서 가장 흔한 실수 중 하나다.

#### ⑤ 그래프는 한 번만 (2분)

**셀 9**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
y.backward()        # RuntimeError
```

에러 메시지를 읽어둔다. 셀 6의 루프가 통과한 건 **매 회차 `y = x ** 2`를 다시 실행해 그래프를 새로 만들었기 때문**이다. 이 셀은 그래프 하나로 두 번 되짚으려 해서 실패한다. 그래프는 `backward()` 한 번에 해제된다.

### Step 4 — 수치미분과 대조 (8분)

> **볼 것** — autograd가 낸 값과, 미분의 정의대로 직접 계산한 값이 소수점 몇 자리까지 일치하는가.
> **끝나면** — 그래디언트가 이상할 때 수치미분으로 검산할 수 있다.

autograd를 믿을 수 있는지 확인한다. 라이브러리가 내주는 값을 그냥 받아 적는 대신, **미분의 정의로 직접 구한 값과 맞춰보는 것**이다. 4주차 4일 gradient check의 축소판이고, 4주차에 역전파를 손으로 구현할 때 이 방법으로 검증한다.

**셀 10**

```python
def f(t):
    return torch.sin(t) * t ** 2

x = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
y = f(x)
y.backward()
analytic = x.grad.item()
print("autograd :", analytic)
```

**셀 11**

```python
h = 1e-6
with torch.no_grad():
    x0 = torch.tensor(1.3, dtype=torch.float64)
    numeric = ((f(x0 + h) - f(x0 - h)) / (2 * h)).item()

print("수치미분 :", numeric)
print("차이     :", abs(analytic - numeric))
```

**`dtype=torch.float64`가 핵심이다.** 기본값인 float32로 두면 차이가 1e-3 수준까지 벌어져 통과 기준을 만족할 수 없다. 아래에서 직접 확인한다.

**셀 12 — `h`와 정밀도의 관계**

```python
for dt in (torch.float32, torch.float64):
    for h in (1e-2, 1e-4, 1e-6, 1e-8):
        with torch.no_grad():
            x0 = torch.tensor(1.3, dtype=dt)
            n = ((f(x0 + h) - f(x0 - h)) / (2 * h)).item()
        print(f"{str(dt):20s} h={h:.0e}  차이={abs(analytic - n):.2e}")
```

두 가지가 동시에 보인다.

- **`h`가 크면** 근사 자체가 거칠어 오차가 크다
- **`h`가 너무 작으면** 두 값의 차이가 부동소수점 정밀도에 묻혀 오히려 나빠진다
- float32는 이 여유 구간이 거의 없다. float64에서만 `1e-6` 근처에 안정 구간이 생긴다

이 U자 곡선이 수치미분의 본질적 한계다. autograd는 이 문제가 없다. 해석적으로 계산하기 때문이다.

여기서 `with torch.no_grad():`를 쓴 이유도 짚어두자. 검산에는 그래프가 필요 없다.

### Step 5 — 그래프의 메모리 비용 (12분) ★

개념 D에서 "그래프를 쌓는 건 공짜가 아니다"라고 했다. 숫자로 확인한다.

#### ① 예측 (3분)

4096×4096 float32 텐서 하나는 **정확히 64 MiB**다. (4096 × 4096 × 4 bytes)

아래 두 코드를 실행하면 각각 몇 MiB가 잡힐까. **실행 전에** 표를 채운다.

```python
# ㉮ 그래프를 쌓는 경우
x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
y = x
for _ in range(5):
    y = torch.sin(y)

# ㉯ no_grad 구역
with torch.no_grad():
    x = torch.randn(4096, 4096, device="cuda")
    y = x
    for _ in range(5):
        y = torch.sin(y)
```

| 경우 | 살아있는 텐서 개수 | 예상 MiB |
|---|---|---|
| ㉮ grad 있음 | ? | ? |
| ㉯ `no_grad` | ? | ? |

**힌트:** `sin`의 미분은 `cos`이고, `cos`를 계산하려면 **입력값이 필요하다.** 되짚을 때 쓰려면 무엇을 붙잡고 있어야 할까.

#### ② 실행 (3분)

**측정 셀이므로 커널을 재시작하고 셀 13 → 14를 순서대로 한 번만 실행한다.** 도입부에서 확인한 규칙 그대로다.

**셀 13**

```python
import torch

torch.cuda.empty_cache()
base = torch.cuda.memory_allocated()

x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
y = x
for _ in range(5):
    y = torch.sin(y)

used = (torch.cuda.memory_allocated() - base) / 1024**2
print(f"grad 있음 : {used:.2f} MiB")
```

**셀 14**

```python
del x, y
torch.cuda.empty_cache()
base = torch.cuda.memory_allocated()

with torch.no_grad():
    x = torch.randn(4096, 4096, device="cuda")
    y = x
    for _ in range(5):
        y = torch.sin(y)

used = (torch.cuda.memory_allocated() - base) / 1024**2
print(f"no_grad   : {used:.2f} MiB")

del x, y
torch.cuda.empty_cache()
```

#### ③ 조사 (4분)

| 경우 | 살아있는 텐서 | 실측 |
|---|---|---|
| grad 있음 | `x` + `sin` 출력 5개 = 6개 | 384 MiB |
| `no_grad` | `x` + 현재 `y` = 2개 | 128 MiB |

되짚을 때 각 `sin`의 입력이 필요하므로 전부 붙잡아둔다. `no_grad` 구역에서는 되짚을 일이 없으니 다음 단계로 넘어가는 순간 이전 텐서가 해제된다.

**이 3배가 오늘의 핵심 수치다.** 학습 시 VRAM이 파라미터 크기의 몇 배로 뛰는 이유의 큰 부분이 여기 있다. 2일차에 기록한 "학습 중에는 중간 활성값이 보통 가장 크다"가 이 현상이다.

#### ④ 실전 대응 (2분)

| 오늘 확인한 것 | 실전 | 나오는 곳 |
|---|---|---|
| 그래프 유지 시 3배 | 평가 루프를 `no_grad()`로 감싼다 | 매 에폭 검증 |
| 활성값이 메모리를 먹는다 | gradient checkpointing | 16주차 DreamBooth |
| 배치가 클수록 활성값도 커진다 | 배치 크기를 줄여 OOM 회피 | 파인튜닝 전 구간 |

> 이 노트북은 중간에 커널 재시작이 들어가므로 **위에서부터 한 번에 다시 실행할 수 없다.** 커밋 전에 셀 13·14 위에 "커널 재시작 후 단독 실행" 마크다운 주석을 남겨둔다.

### Step 6 — 스칼라가 아닌 출력 (6분)

> **볼 것** — 출력이 숫자 3개일 때 `backward()`가 **거부하는** 것. 그리고 `.sum()`으로 뭉치면 통과하는 것.
> **끝나면** — "손실은 왜 항상 숫자 하나인가"에 답할 수 있다.

지금까지는 출력이 늘 숫자 하나였다. 벡터를 넣으면 무슨 일이 생기는지 확인한다. **실전에서 손실 함수가 왜 평균이나 합을 취하는지**가 여기서 설명된다.

**셀 15**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2      # [1, 4, 9] — 숫자 3개
y.backward()    # RuntimeError
```

`backward()`는 **출력이 숫자 하나일 때만** 인자 없이 쓸 수 있다. 숫자가 3개면 "무엇을 줄일 방향인가"가 정해지지 않는다.

실전에서 문제가 안 되는 이유는 **손실이 항상 숫자 하나**이기 때문이다. 배치 안 샘플별 오차를 평균이나 합으로 뭉친다.

**셀 16**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)      # [2, 4, 6] = 2x
```

**셀 17 — 굳이 뭉치지 않으려면**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward(gradient=torch.ones_like(y))
print(x.grad)      # [2, 4, 6] — 셀 16과 같다
```

각 출력에 가중치 1을 주고 합치는 것이 `sum`이니 같은 계산이다. 4주차에 이 구조를 직접 구현한다. 지금은 **"손실은 스칼라로 만들어 넘긴다"**만 챙기면 된다.

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `(x ** 2).sum()` | `nn.CrossEntropyLoss()` 기본 `reduction='mean'` | 손실 함수가 알아서 뭉침 |

### Step 7 — 경사하강 1회전 (6분) · 순서 맞추기

> **볼 것** — 세 줄의 **순서**가 왜 이래야 하는지. `w`가 3을 향해 움직이고 `loss`가 0으로 떨어지는 것.
> **끝나면** — 학습 루프의 세 줄을 순서대로 쓸 수 있다.

오늘 배운 4개를 전부 한 루프에 넣는다. 데이터도 모델도 없이 숫자 하나(`w`)를 최적값으로 미는 게 전부지만, **구조는 실제 학습 루프와 완전히 동일하다.**

#### ① 블록 배열 (3분)

아래 블록 중 **4개를 골라 올바른 순서로** 루프 본문을 채운다. 두 개는 쓰지 않는다.

```python
# 블록 A
loss = (w - 3) ** 2

# 블록 B
loss.backward()

# 블록 C
if w.grad is not None:
    w.grad.zero_()

# 블록 D
with torch.no_grad():
    w -= lr * w.grad

# 블록 E
w -= lr * w.grad

# 블록 F
w.grad.zero_()
```

**골격**

```python
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(20):
    # 여기에 블록 4개를 순서대로

    if step % 5 == 0:
        print(f"step {step:2d}  w={w.item():.4f}  loss={loss.item():.4f}")

print(f"최종 w = {w.item():.4f}")
```

**쓰지 않을 블록 두 개가 왜 틀렸는지도 한 줄씩 적는다.** 둘 다 실전에서 흔한 실수라 이유를 말할 수 있어야 한다.

#### ② 실행과 확인 (2분)

`w`가 3에 가까워지고 `loss`가 0으로 떨어지면 성공이다. 막히면 개념 C·D의 5번 표를 본다. 네 블록이 각각 `optimizer.zero_grad()` / 순전파 / `loss.backward()` / `optimizer.step()`에 해당한다.

#### ③ 한 가지 더 (1분)

배열한 4개 중 **앞의 두 블록은 서로 순서를 바꿔도 동작한다.** 어느 둘이고 왜 그런지 한 줄로 적어보자. 나머지 둘은 순서를 바꾸면 무너진다.

> 이 루프가 모든 학습 루프의 골격이다. 4일차에는 여기에 데이터와 모델이 붙는다. 구조는 그대로다.

---

## 3교시 (20분) · 실패 케이스

### 먼저 고쳐본다 (10분)

**아래 표를 보기 전에** 두 코드를 진단한다. 자료를 뒤지지 말고 오늘 실습에서 본 것을 떠올린다.

**문제 ① — 학습이 진행될수록 손실이 커진다**

```python
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(20):
    loss = (w - 3) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
    print(f"step {step}  w={w.item():.4f}")
```

- 무엇이 빠졌는가
- **어느 출력을 보고 그렇게 판단했는가** — 진단 근거를 한 줄로 적는다
- 고쳐서 `w`가 3에 수렴하게 만든다

**문제 ② — 평가 루프에서 VRAM이 계속 늘다가 OOM**

```python
model_out = []
losses = []

for i in range(1000):
    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    loss = (x ** 2).mean()
    losses.append(loss)
```

- 이 코드가 왜 메모리를 계속 먹는가. 원인은 **두 가지**다
- 각각 한 줄로 고친다

두 문제 모두 오늘 실습에서 이미 본 현상이다. 표를 보고 답을 찾는 게 아니라 **기억에서 꺼내는 것**이 목적이므로, 5분은 자료 없이 버틴다.

### 정리 표 (10분)

이제 대조한다. 오늘 안 겪은 것도 읽어둔다. 전부 앞으로 반복해서 만난다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `element 0 of tensors does not require grad and does not have a grad_fn` | 되짚을 그래프가 없다. 입력에 `requires_grad`가 없거나 `no_grad()` 구역에서 만든 텐서에 `backward()`를 불렀다 | `y.grad_fn`을 찍어본다. `None`이면 확정 |
| `grad can be implicitly created only for scalar outputs` | 출력이 숫자 하나가 아니다 | `.sum()`·`.mean()`으로 뭉친 뒤 `backward()` |
| `Trying to backward through the graph a second time` | 그래프는 `backward()` 한 번에 해제된다 | 순전파를 다시 실행한다. 정말 두 번 필요하면 `retain_graph=True` — 메모리를 계속 잡으니 남용 금지 |
| `.grad`가 `None` + `UserWarning: ...not a leaf...` | 중간 결과의 `.grad`를 읽었다 | 리프에서 읽는다. 필요하면 순전파 **전에** `retain_grad()` |
| `a leaf Variable that requires grad is being used in an in-place operation` | 추적 대상 리프에 값을 직접 덮어썼다 | 갱신을 `with torch.no_grad():` 안에서 |
| `Only Tensors of floating point and complex dtype can require gradients` | 정수 텐서에 `requires_grad=True`를 붙였다 | `dtype=torch.float32` 또는 `.float()` |
| 학습이 진행될수록 그래디언트가 비정상적으로 커진다 | `zero_grad()`를 안 해 매 스텝 누적됐다 | 스텝마다 초기화 ← **문제 ①** |
| 평가 루프 중 VRAM이 계속 증가하다 OOM | ① 평가에 `no_grad()`를 안 썼다 ② 손실을 텐서째 리스트에 쌓았다 | ① `with torch.no_grad():` ② `losses.append(loss.item())` ← **문제 ②** |
| 수치미분 검산이 안 맞는다 | float32에서 `h`가 너무 작아 정밀도에 묻혔다 | float64로 계산하고 `h`는 `1e-6` 근처 |

> 위 9개 중 **1·2·5번이 입문 구간에서 가장 자주 나온다.** 에러 첫 줄만 읽고 검색하지 말고 셋 중 어느 경우인지 먼저 대조하면 진단이 30초로 끝난다.

---

## 마무리 (15분)

### 노트북 정리

- 셀 9와 셀 15는 **의도적으로 에러가 나는 셀**이다. 지우지 말고 왜 나는 에러인지 주석 한 줄씩 남긴다
- 셀 13·14 위에 "커널 재시작 후 단독 실행" 주석을 남긴다
- Step 3 ①·Step 5 ①·Step 7 ①에 적은 **예측과 배열을 지우지 않는다.** 틀렸다면 특히 남긴다. 복습할 때 가장 쓸모 있는 기록이다
- 3교시 두 문제의 **진단 근거 한 줄**을 마크다운 셀로 남긴다
- Step 5 실측값과 Step 4의 `h` 실험 결과를 마크다운 셀로 정리한다

### 통과 기준

아래 4개가 노트북 출력으로 확인되면 오늘은 끝이다.

1. `y = x**2`, `x=3`에서 `x.grad`가 **6.0**으로 찍힘
2. 셀 6에서 `6 → 12 → 18`, 셀 8에서 `6 → 6 → 6`으로 찍힘
3. Step 4에서 float64 · `h=1e-6` 기준 autograd와 수치미분의 차이가 **1e-5 미만**
4. Step 5에서 `grad 있음`이 `no_grad`보다 **2배 이상** 크게 찍힘

Step 7의 `w`가 3에 수렴하는 것은 보너스다. 통과 기준에는 넣지 않는다.

여기까지면 멈춘다. 진도를 더 나가지 않는다.

### 다음 세션 예고

**1주차 4일 · 첫 학습 루프 (2시간)** — 오늘 셀 18의 골격에 데이터를 붙여 선형 회귀를 텐서만으로 학습시킨다. 새 개념은 적고 오늘 배운 걸 조립하는 세션이라 오늘보다 가볍게 느껴질 것이다. 대신 **손실이 안 떨어질 때 어디를 보는지**를 처음 다룬다.

---

## 퀴즈 (5문항)

답을 제출하면 채팅에서 채점하고 해설과 함께 오늘 배운 것을 정리해준다.

**1.** 다음 코드의 출력은?

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
y = x ** 3
y.backward()
print(x.grad)
```

- (A) `tensor(12.)`
- (B) `tensor(24.)`
- (C) `None`
- (D) `RuntimeError`

---

**2.** `a.grad`가 `None`으로 나오고 경고가 뜬다.

```python
x = torch.tensor(2.0, requires_grad=True)
a = x * 3
c = a ** 2
c.backward()
print(a.grad)     # None + UserWarning
```

원인으로 맞는 설명은?

- (A) `a`에 `requires_grad=True`를 안 붙였다
- (B) `a`는 중간 결과라서 그래디언트가 저장되지 않는다
- (C) `c.backward()`가 실패해 아무 그래디언트도 계산되지 않았다
- (D) `a`가 스칼라가 아니라서 그렇다

---

**3.** Step 5에서 `no_grad()` 구역이 메모리를 적게 쓴 직접적인 이유는?

- (A) `no_grad()`가 계산을 float16으로 낮춰 수행한다
- (B) `no_grad()`가 각 단계 후 `empty_cache()`를 자동 호출한다
- (C) 되짚을 때 쓸 각 단계의 입력값을 붙잡아두지 않으므로 다음 단계로 넘어가면 이전 텐서가 해제된다
- (D) `no_grad()` 안에서는 텐서가 GPU가 아니라 CPU에 올라간다

---

**4.** Step 4에서 float32로 `h=1e-8`을 쓰면 오차가 오히려 커진다. 이유는?

- (A) `h`가 작을수록 근사식의 절단오차가 커지기 때문
- (B) 두 함수값의 차이가 float32 정밀도에 묻혀 유효숫자가 사라지기 때문
- (C) autograd 쪽 값이 float32에서 부정확해지기 때문
- (D) `torch.sin`이 작은 입력에서 오차가 크기 때문

---

**5.** 학습 루프에서 손실이 스텝마다 오히려 커지고 그래디언트 값이 계속 불어난다. 코드에서 가장 먼저 확인할 것은?

- (A) `loss`가 스칼라인지
- (B) 파라미터에 `requires_grad=True`가 붙었는지
- (C) 스텝마다 그래디언트를 초기화하고 있는지
- (D) 파라미터 갱신이 `no_grad()` 안에서 이뤄지는지
