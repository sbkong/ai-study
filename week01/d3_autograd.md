# 1주차 3일 · autograd

> 배정 시간 **2시간** · 산출물 `week01/d3_autograd.ipynb`

---

## 도입 (5분) · 직전 세션 정리

### ① VRAM 측정값 재확인 (2분)

2일차 Step 5-3에서 `텐서 1개` 증가분이 0.00 MiB로 찍힌 건 셀 재실행 때문으로 추정했다. 확정하고 넘어간다.

**커널을 재시작한 직후, 아래 셀을 딱 한 번만 실행한다.**

```python
import torch

base = torch.cuda.memory_allocated()
batch = torch.randn(32, 3, 224, 224, device="cuda")
delta = torch.cuda.memory_allocated() - base
print(f"{delta / 1024**2:.2f} MiB")   # 18.38 이 나와야 정상
```

18.38이 나오면 원인 추정이 맞았던 것이고, 미해결 항목은 여기서 닫는다. 다시 0.00이 나오면 셀 재실행이 아닌 다른 원인이니 그때 채팅으로 알려달라.

> **오늘 이후의 규칙:** 메모리를 재는 셀은 커널 재시작 직후 1회만 실행한다. 오늘 Step 5에서 다시 쓴다.

### ② `reshape` 복사 여부 (3분)

2일차 퀴즈 2번 오답 지점이다. 오늘 실습에 직접 영향이 있어서 한 번 더 짚는다.

```python
a = torch.arange(6.).reshape(2, 3)
b = a.t().reshape(6)      # a.t()는 연속이 아니라 → 복사본
c = a.reshape(6)          # a는 연속이라 → 원본 공유

print(b.data_ptr() == a.data_ptr())   # False
print(c.data_ptr() == a.data_ptr())   # True
```

판별은 `.data_ptr()` 비교. **오늘 중요한 이유는 이거다** — 복사가 일어나면 그래디언트가 흐르는 경로가 한 단계 더 늘어난다. 오늘 Step 2에서 `grad_fn`으로 그 경로를 눈으로 확인한다.

---

## 1교시 (25분) · 개념 4개

오늘 처음 나오는 용어는 네 개다. 하나씩 끊어서 본다.

### 개념 A (6분) · 그래디언트와 `requires_grad`

**1. 무엇인가**

그래디언트는 **"이 숫자를 조금 올리면 결과가 얼마나 변하는가"**를 나타내는 값이다.

**2. 왜 필요한가**

딥러닝 모델은 결국 숫자 수백만 개(파라미터)의 묶음이다. 학습은 그 숫자들을 조금씩 바꿔서 오차를 줄여가는 과정이다. 그런데 어느 방향으로 바꿔야 오차가 줄어드는지 모르면 아무것도 못 한다.

숫자 하나당 "올리면 오차가 커지나 작아지나"를 알면 방향이 정해진다. 그게 그래디언트다. 커진다면 내리고, 작아진다면 올린다.

문제는 파라미터가 수백만 개라는 것. 하나씩 손으로 미분할 수는 없다. **PyTorch가 이걸 자동으로 해주는 기능이 autograd다.** 오늘 수업의 전부가 이거다.

**3. 언제 쓰는가**

추적할 대상에 `requires_grad=True` 표시를 달아둔다. 실전에서는 모델 파라미터에 자동으로 붙어 있어서 직접 쓸 일은 드물지만, 오늘은 손으로 붙여가며 동작을 본다.

**4. 최소 예시**

```python
x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)   # True

y = torch.tensor(2.0)
print(y.requires_grad)   # False — 기본값
```

> **정수 텐서에는 붙지 않는다.** `torch.tensor(2, requires_grad=True)`는 에러다. 미분은 실수 위에서만 정의되기 때문이다. 3교시 실패 케이스에 다시 나온다.

### 개념 B (6분) · 계산 그래프

**1. 무엇인가**

계산 그래프는 **"이 결과가 어떤 계산을 거쳐 나왔는지"를 기록한 목록**이다.

**2. 왜 필요한가**

`z = (x + 2) * 3`을 계산하면 보통은 결과 숫자만 남는다. 그런데 x의 그래디언트를 구하려면 z가 어떻게 만들어졌는지를 거꾸로 되짚어야 한다. 곱하기가 마지막이었고, 그 앞에 더하기가 있었다는 정보가 필요하다.

PyTorch는 순전파(계산을 앞으로 진행할 때) 이 기록을 자동으로 남긴다. 그래서 나중에 거꾸로 되짚을 수 있다.

**3. 언제 쓰는가**

직접 만들지는 않는다. `requires_grad=True`인 텐서로 연산을 하면 저절로 쌓인다. 대신 **`grad_fn`으로 들여다보는 것**은 자주 한다. 그래디언트가 안 흐른다는 진단을 할 때 첫 번째로 보는 곳이다.

**4. 최소 예시**

```python
x = torch.tensor(2.0, requires_grad=True)
z = (x + 2) * 3

print(z.grad_fn)              # <MulBackward0 ...> — 마지막 연산이 곱하기
print(z.grad_fn.next_functions)  # 그 앞에 AddBackward0
```

두 종류의 텐서가 있다는 것만 구분해두자.

| 종류 | 뜻 | `grad_fn` |
|---|---|---|
| **리프(leaf)** | 내가 직접 만든 텐서 | `None` |
| 중간 결과 | 연산으로 나온 텐서 | 연산 종류가 들어있음 |

`x.is_leaf`는 True, `z.is_leaf`는 False다. 이 구분이 개념 C에서 바로 필요해진다.

### 개념 C (7분) · `backward()`와 `.grad`, 그리고 누적

**1. 무엇인가**

`backward()`는 **계산 그래프를 거꾸로 훑어서 그래디언트를 계산하는 명령**이다. 계산된 값은 각 텐서의 `.grad`에 들어간다.

**2. 왜 필요한가**

개념 A에서 그래디언트가 필요하다고 했고, B에서 되짚을 기록이 쌓인다고 했다. `backward()`가 실제로 되짚는 실행 버튼이다. 이걸 부르지 않으면 `.grad`는 계속 `None`이다.

**3. 언제 쓰는가**

학습 루프에서 매 스텝마다 `loss.backward()` 한 줄로 나온다. 오늘은 손으로 부르며 값을 확인한다.

**4. 최소 예시**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2          # y = x²
y.backward()        # dy/dx = 2x = 6
print(x.grad)       # tensor(6.)
```

**여기서 오늘 가장 중요한 성질 하나.** `.grad`는 **덮어쓰지 않고 더한다.**

```python
y = x ** 2
y.backward()
print(x.grad)       # tensor(12.)  ← 6이 아니라 6+6
```

두 번 불렀으니 6이 두 번 더해져 12가 됐다. 이게 자동으로 지워지지 않는 건 의도된 설계다. 배치를 여러 조각으로 나눠 넣고 그래디언트를 합산하는 기법(gradient accumulation)이 이 성질에 기대고 있다 — VRAM이 부족할 때 쓰는 수단이라 나중에 다시 만난다.

대가로, **매번 직접 지워야 한다.** 안 지우면 이전 스텝의 그래디언트가 계속 섞인다.

```python
x.grad.zero_()      # 0으로 초기화
```

2주차 3일에 나오는 `optimizer.zero_grad()`가 정확히 이 일을 대신 해주는 함수다. 지금 손으로 지워보면 그때 왜 그 줄이 있는지 설명이 필요 없어진다.

### 개념 D (6분) · `no_grad()`와 `detach()`

**1. 무엇인가**

`no_grad()`는 **"이 구역에서는 계산 그래프를 만들지 마라"**는 지시다. `detach()`는 **한 텐서를 그래프에서 떼어내 복사본을 만드는 것**이다.

**2. 왜 필요한가**

계산 그래프를 쌓는 건 공짜가 아니다. 되짚기 위해 중간 결과를 메모리에 붙잡아둔다. 학습할 때는 그게 필요하지만, 학습이 끝난 모델로 추론만 할 때는 순전파 결과만 있으면 되므로 전부 낭비다. VRAM도 먹고 조금 느리다.

**3. 언제 쓰는가**

세 곳에서 거의 항상 쓴다.

- 추론·평가할 때 (`with torch.no_grad():`)
- 파라미터를 직접 갱신할 때 — 갱신 자체는 학습 대상이 아니다
- 손실값을 로그로 모을 때 (`loss.item()`이나 `.detach()`) — 텐서째로 리스트에 쌓으면 그래프가 통째로 남아 메모리가 계속 늘어난다

**4. 최소 예시**

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 3
print(y.requires_grad, y.grad_fn)   # False None — 그래프가 없다

z = (x * 3).detach()
print(z.requires_grad)              # False
```

> **차이:** `no_grad()`는 블록 전체에 적용되는 스위치, `detach()`는 텐서 하나를 떼어내는 도구다. 평가 루프는 `no_grad()`, 값 하나 빼올 때는 `detach()`/`.item()`.

---

## 2교시 (55분) · 실습

`week01/d3_autograd.ipynb`를 새로 만든다. **커널을 재시작한 상태에서 시작한다.**

### Step 1 — 첫 `backward` (5분)

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

손으로 검산하자. `y = x²`이면 `dy/dx = 2x`, x가 3이니 6. 숫자가 맞는지 눈으로 확인하고 넘어간다. 오늘은 계속 이 방식으로, **손계산값과 대조**하며 진행한다.

### Step 2 — 계산 그래프 들여다보기 (8분)

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

`a.grad`가 `None`인 게 정상이다. **중간 결과의 그래디언트는 계산 도중 쓰이고 버려진다.** 저장하면 메모리만 먹고, 학습에 필요한 건 리프(= 파라미터)의 그래디언트뿐이라서다.

굳이 봐야 하면 순전파 **전에** 표시를 달아둔다.

**셀 5**

```python
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(4.0, requires_grad=True)
a = x * w
a.retain_grad()          # 이 줄이 backward보다 앞에 있어야 한다
c = (a + 1) ** 2
c.backward()
print("a.grad:", a.grad)  # dc/da = 2b = 18
```

### Step 3 — 누적과 `zero_()` (8분)

**셀 6**

```python
x = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(f"{i}회차 x.grad = {x.grad.item()}")   # 6, 12, 18
```

6씩 더해지는 걸 확인한다. **셀 7**에서 지우는 쪽을 본다.

**셀 7**

```python
x = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    if x.grad is not None:
        x.grad.zero_()
    y = x ** 2
    y.backward()
    print(f"{i}회차 x.grad = {x.grad.item()}")   # 6, 6, 6
```

`if x.grad is not None`이 필요한 이유는, 첫 회차에는 `.grad`가 아직 `None`이라 `.zero_()`를 부를 대상이 없기 때문이다. 실전에서는 옵티마이저가 이걸 처리해준다.

**셀 8 — 같은 그래프를 두 번 되짚기**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
y.backward()        # RuntimeError
```

에러 메시지를 읽어두자. **Step 3의 첫 루프는 매 회차 `y = x ** 2`를 다시 실행해서 그래프를 새로 만들었기 때문에** 통과했다. 이 셀은 그래프 하나로 두 번 되짚으려 해서 실패한다. 그래프는 `backward()` 한 번에 해제된다. 3교시 실패 케이스에서 대응법을 본다.

### Step 4 — 수치미분과 대조 (10분)

autograd를 믿을 수 있는지 직접 확인한다. 4주차 4일 gradient check의 축소판이다.

**셀 9**

```python
def f(t):
    return torch.sin(t) * t ** 2

x = torch.tensor(1.3, requires_grad=True)
y = f(x)
y.backward()
analytic = x.grad.item()
print("autograd :", analytic)
```

**셀 10**

```python
h = 1e-4
with torch.no_grad():
    x0 = torch.tensor(1.3)
    numeric = ((f(x0 + h) - f(x0 - h)) / (2 * h)).item()

print("수치미분 :", numeric)
print("차이     :", abs(analytic - numeric))
```

차이가 `1e-5` 수준 이하로 나오면 일치로 본다. 완전히 같게 나오지 않는 건 수치미분 쪽이 근사값이라서다 — `h`를 너무 작게 잡으면 오히려 부동소수점 오차로 더 나빠진다. `h`를 `1e-8`로 바꿔서 한 번 확인해보면 감이 온다.

여기서 `with torch.no_grad():`를 쓴 이유도 짚어두자. 검산에는 그래프가 필요 없다.

### Step 5 — 그래프의 메모리 비용 측정 (10분)

개념 D에서 "그래프를 쌓는 건 공짜가 아니다"라고 했다. 숫자로 확인한다.

**측정 셀이므로 커널을 재시작하고 셀 11 → 12를 순서대로 한 번만 실행한다.** 도입부에서 확인한 규칙 그대로다.

**셀 11 — 그래프를 쌓는 경우**

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

**셀 12 — 정리 후 `no_grad`**

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

**손계산으로 예측해보고 대조한다.** 4096×4096 float32 = 정확히 64 MiB.

| 경우 | 살아있는 텐서 | 예상값 |
|---|---|---|
| grad 있음 | `x` + `sin` 출력 5개 = 6개 | 384 MiB |
| `no_grad` | `x` + 현재 `y` = 2개 | 128 MiB |

왜 차이가 나는가. `sin`의 미분은 `cos`이고, `cos`를 계산하려면 **입력값이 필요하다.** 그래서 되짚을 때 쓰려고 각 단계의 입력을 붙잡아둔다. `no_grad` 구역에서는 되짚을 일이 없으니 다음 단계로 넘어가는 순간 이전 텐서가 해제된다.

> 이 3배가 오늘의 핵심 수치다. 학습 시 VRAM이 파라미터 크기의 몇 배로 뛰는 이유의 큰 부분이 여기 있다. 2일차에 기록한 "학습 중에는 중간 활성값이 보통 가장 크다"가 이 현상이다. 배치 크기 산정은 7주차 이후에 실제로 한다.

### Step 6 — 스칼라가 아닌 출력 (7분)

**셀 13**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2      # [1, 4, 9] — 숫자 3개
y.backward()    # RuntimeError
```

`backward()`는 **출력이 숫자 하나일 때만** 인자 없이 쓸 수 있다. 숫자가 3개면 "무엇을 줄일 방향인가"가 정해지지 않는다. 방향이 셋이면 답도 셋인데 `.grad`에 넣을 값은 하나뿐이다.

실전에서 이게 문제가 안 되는 이유는, **손실(loss)은 항상 숫자 하나**이기 때문이다. 배치 안의 샘플별 오차를 평균이나 합으로 뭉쳐서 하나로 만든다.

**셀 14**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)      # [2, 4, 6] = 2x
```

**셀 15 — 굳이 뭉치지 않으려면**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
y.backward(gradient=torch.ones_like(y))
print(x.grad)      # [2, 4, 6] — 셀 14와 같다
```

`.sum()` 후 `backward()`와 `gradient=ones`가 같은 결과인 건 우연이 아니다. 각 출력에 가중치 1을 주고 합치는 것이 `sum`이니 같은 계산이다. 4주차에 이 구조를 직접 구현한다. 지금은 **"손실은 스칼라로 만들어서 넘긴다"**만 챙기면 된다.

### Step 7 — 경사하강 1회전 (7분)

오늘 배운 4개를 다 쓴다. 1주차 4일 학습 루프의 축소판이다.

**셀 16**

```python
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(20):
    loss = (w - 3) ** 2          # w가 3일 때 최소

    if w.grad is not None:
        w.grad.zero_()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad         # 갱신은 학습 대상이 아니다

    if step % 5 == 0:
        print(f"step {step:2d}  w={w.item():.4f}  loss={loss.item():.4f}")

print(f"최종 w = {w.item():.4f}")
```

`w`가 3에 가까워지고 `loss`가 0으로 떨어지면 성공이다.

**`with torch.no_grad():`를 빼고 한 번 돌려보자.** 에러가 난다. `w`는 리프이면서 그래디언트 추적 대상인데, 그 자리에 값을 덮어쓰려 했기 때문이다. 이 에러도 3교시 표에 있다.

> 이 16번 셀이 모든 학습 루프의 골격이다. 4일차에는 여기에 데이터와 모델이 붙는다. 구조는 그대로다.

---

## 3교시 (20분) · 실패 케이스

오늘 다 겪지 않아도 읽어둔다. 전부 앞으로 반복해서 만난다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `element 0 of tensors does not require grad and does not have a grad_fn` | 되짚을 그래프가 없다. 입력에 `requires_grad`가 없거나, `no_grad()` 구역 안에서 만든 텐서에 `backward()`를 불렀다 | `y.grad_fn`을 찍어본다. `None`이면 확정. `requires_grad=True` 확인, `no_grad()` 블록 범위 확인 |
| `grad can be implicitly created only for scalar outputs` | 출력이 숫자 하나가 아니다 | `.sum()`이나 `.mean()`으로 뭉친 뒤 `backward()`. 또는 `backward(gradient=...)` |
| `Trying to backward through the graph a second time` | 그래프는 `backward()` 한 번에 해제된다 | 순전파를 다시 실행해 그래프를 새로 만든다. 진짜로 두 번 필요하면 `backward(retain_graph=True)` — 메모리를 계속 잡고 있으니 남용 금지 |
| `.grad`가 `None`이고 `UserWarning: The .grad attribute of a Tensor that is not a leaf...` | 중간 결과의 `.grad`를 읽었다 | 리프에서 읽는다. 정말 필요하면 순전파 **전에** `retain_grad()` |
| `a leaf Variable that requires grad is being used in an in-place operation` | 추적 대상인 리프에 값을 직접 덮어썼다 | 갱신을 `with torch.no_grad():` 안에서 한다 |
| `Only Tensors of floating point and complex dtype can require gradients` | 정수 텐서에 `requires_grad=True`를 붙였다 | `dtype=torch.float32`로 만들거나 `.float()` |
| 학습이 진행될수록 그래디언트가 비정상적으로 커진다 | `zero_grad()`를 안 해서 매 스텝 누적됐다 | 스텝마다 초기화. 2주차 3일에서 옵티마이저로 처리 |
| 평가 루프를 돌리는 중 VRAM이 계속 증가하다 OOM | ① 평가에 `no_grad()`를 안 썼다 ② 손실을 텐서째로 리스트에 쌓았다 — 그래프 전체가 따라 남는다 | ① `with torch.no_grad():` ② `losses.append(loss.item())` |
| GPU 실습 후 다음 셀에서 OOM | 이전 텐서가 아직 살아있다 | `del` 후 `torch.cuda.empty_cache()`, 또는 커널 재시작 |

> 위 8개 중 **1·2·5번이 입문 구간에서 가장 자주 나온다.** 에러 메시지 첫 줄만 읽고 검색하지 말고, 셋 중 어느 경우인지 먼저 대조하는 습관을 들이면 진단이 30초로 끝난다.

---

## 마무리 (15분)

### 커밋 과제

```
week01/d3_autograd.ipynb
```

커밋 전에 정리할 것:

- 셀 8과 셀 13은 **의도적으로 에러가 나는 셀**이다. 지우지 말고, 왜 나는 에러인지 주석 한 줄씩 남긴다. 3개월 뒤 복습할 때 이게 자산이 된다
- Step 5의 측정값(384 / 128 MiB 예상 대비 실측)을 마크다운 셀로 기록한다
- Step 4의 `h`를 바꿔봤다면 그 결과도 한 줄 남긴다

커밋 메시지 예시:

```
week01 d3: autograd — 계산 그래프, backward, 그래디언트 누적, no_grad

- 그래프 유지 시 활성값 메모리 3배 실측 (384 vs 128 MiB)
- 수치미분과 autograd 일치 확인 (차이 ~1e-6)
- 실패 케이스 8종 정리
```

### 통과 기준

아래 4개가 노트북 출력으로 확인되면 오늘은 끝이다.

1. `y = x**2`, `x=3`에서 `x.grad`가 **6.0**으로 찍힘
2. `zero_()` 없는 루프에서 `6 → 12 → 18`, `zero_()` 있는 루프에서 `6 → 6 → 6`으로 찍힘
3. Step 4에서 autograd 값과 수치미분 값의 차이가 **1e-5 미만**
4. Step 5에서 `grad 있음`이 `no_grad`보다 **2배 이상** 크게 찍힘

Step 7의 `w`가 3에 수렴하는 것은 보너스다. 통과 기준에는 넣지 않는다.

여기까지면 멈춘다. 진도를 더 나가지 않는다.

### 다음 세션 예고

**1주차 4일 · 첫 학습 루프 (2시간)** — 오늘 셀 16의 골격에 데이터를 붙여 선형 회귀를 텐서만으로 학습시킨다. 새 개념은 적고 오늘 배운 걸 조립하는 세션이라 오늘보다 가볍게 느껴질 것이다. 대신 여기서 **손실이 안 떨어질 때 어디를 보는지**를 처음 다룬다.

---

## 퀴즈 (5문항)

객관식이다. 답을 제출하면 채팅에서 채점하고 해설을 준다.

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
- (C) `c.backward()`가 실패해서 아무 그래디언트도 계산되지 않았다
- (D) `a`가 스칼라가 아니라서 그렇다

---

**3.** 4096×4096 float32 텐서에 `torch.sin`을 5번 연쇄 적용했을 때, `no_grad()` 구역 안에서 실행하면 밖에서 실행할 때보다 메모리를 적게 쓴다. 그 직접적인 이유는?

- (A) `no_grad()`가 계산을 float16으로 낮춰 수행한다
- (B) `no_grad()`가 각 단계 후 `empty_cache()`를 자동 호출한다
- (C) 되짚을 때 쓸 각 단계의 입력값을 붙잡아두지 않으므로 다음 단계로 넘어가면 이전 텐서가 해제된다
- (D) `no_grad()` 안에서는 텐서가 GPU가 아니라 CPU에 올라간다

---

**4.** 다음 두 코드의 `x.grad`를 비교하면?

```python
# 코드 ①
x = torch.tensor([1., 2., 3.], requires_grad=True)
(x ** 2).sum().backward()

# 코드 ②
x = torch.tensor([1., 2., 3.], requires_grad=True)
(x ** 2).backward(gradient=torch.ones(3))
```

- (A) ①은 `[2, 4, 6]`, ②는 `RuntimeError`
- (B) ①과 ② 모두 `[2, 4, 6]`
- (C) ①은 `[2, 4, 6]`, ②는 `[1, 1, 1]`
- (D) ①은 `RuntimeError`, ②만 동작한다

---

**5.** 학습 루프에서 손실이 스텝마다 오히려 커지고, 그래디언트 값이 계속 불어난다. 코드에서 가장 먼저 확인할 것은?

- (A) `loss`가 스칼라인지
- (B) 파라미터에 `requires_grad=True`가 붙었는지
- (C) 스텝마다 그래디언트를 초기화하고 있는지
- (D) 파라미터 갱신이 `no_grad()` 안에서 이뤄지는지
