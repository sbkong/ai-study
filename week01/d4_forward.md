# 1주차 4일 · 첫 학습 루프

> **배정 시간 2시간** · 선형 회귀를 텐서만으로 — 순전파 · 손실 · 역전파 · 갱신 네 단계를 손으로 조립한다.
> 산출물 `week01/d4_train_loop.ipynb` · 장치 **CPU**

| 교시 | 시간 | 내용 |
|---|---|---|
| 도입 | 5분 | 누적 복습 퀴즈 3문항 |
| 목표 | 2분 | 오늘 얻는 것 |
| 1교시 | 25분 | 개념 네 덩어리 |
| 2교시 | 53분 | 실습 Step 1~6 |
| 3교시 | 20분 | 실패 케이스 |
| 마무리 | 15분 | 통과 기준 + 2주차 대응표 |

---

## 도입 (5분) · 누적 복습 퀴즈

답을 채팅으로 주세요. 채점과 해설은 그때 합니다. 한 문항에 1분을 넘기지 마세요 — 모르면 모르는 채로 넘어가는 것도 정보입니다.

**복습 1 · 3일차** — 다음 코드의 마지막 출력은?

```python
x = torch.tensor(2.0, requires_grad=True)

y = x ** 3
y.backward()

y2 = x ** 3
y2.backward()

print(x.grad)
```

① `tensor(12.)`  ② `tensor(24.)`  ③ `RuntimeError`  ④ `None`

**복습 2 · 2일차** — 다음 코드의 출력은?

```python
a = torch.arange(6.).reshape(2, 3)
b = a.t().reshape(6)
b[0] = 99.
print(a[0, 0].item())
```

① `99.0`  ② `0.0`  ③ `RuntimeError`  ④ `nan`

**복습 3 · 1일차** — GPU 행렬곱 시간을 아래처럼 쟀더니 실제 계산 시간보다 훨씬 짧게 찍혔다. 이유는?

```python
t = time.time()
c = a @ b                  # a, b 는 cuda 텐서
print(time.time() - t)
```

① GPU가 실제로 그만큼 빠르다
② CUDA 호출은 비동기라, 커널이 끝나기 전에 시간이 찍힌다
③ `time.time()`의 해상도가 그 정도라 반올림됐다
④ 행렬곱이 지연 평가되어 `print` 시점에 실행된다

---

## 목표 (2분) · 오늘 얻는 것

| 배우는 것 | 이걸로 할 수 있게 되는 일 | 중요도 |
|---|---|---|
| 순전파(forward) | 남의 학습 코드를 열었을 때 "모델이 예측을 만드는 줄"을 짚어낼 수 있다 | ★★★ 필수 |
| 손실 함수와 MSE | 예측이 얼마나 틀렸는지를 숫자 하나로 만들고, 그 숫자가 줄어드는지로 학습 성공을 판정한다 | ★★★ 필수 |
| 그래디언트 강하와 학습률 | `w -= lr * w.grad`가 왜 뺄셈인지 설명하고, loss가 `nan`이 될 때 학습률을 먼저 의심한다 | ★★★ 필수 |
| 학습 루프 네 단계의 순서 | 학습 코드에서 빠진 줄·뒤바뀐 줄을 찾아낸다 | ★★★ 필수 |
| `torch.no_grad()`로 파라미터 갱신 | 갱신을 감싸지 않았을 때 나는 에러 메시지를 읽고 바로 고친다 | ★★★ 필수 |
| `loss.item()` | 손실을 로그에 남길 때 계산 그래프를 통째로 붙잡아 메모리가 새는 것을 막는다 | ★★ 권장 · 3주차 2일 손실 곡선에서 다시 |
| 에폭과 스텝 | 학습 로그의 `Epoch 3/10` 같은 표기를 정확히 읽는다 | ★★ 권장 · 2주차 4일 DataLoader에서 정식으로 |
| 파라미터 초기값 | 같은 코드가 실행할 때마다 다른 숫자를 내는 이유를 안다 | ★ 참고 · 8주차 3일 재현성에서 정식으로 |

> 시간이 부족하면 ★부터 버린다.

---

## 1교시 (25분) · 개념 네 덩어리

3일차까지는 텐서 하나에 `backward()`를 부르고 `.grad`를 확인하는 데서 끝났습니다. 오늘은 그 `.grad`를 *쓰는* 쪽으로 넘어갑니다. 데이터에서 규칙을 찾아내는 과정 전체가 사실 네 줄이라는 걸 보게 됩니다.

### A · 모델과 파라미터, 그리고 순전파 (6분)

**모델은 입력을 넣으면 출력이 나오는 계산식입니다.** 오늘 쓸 식은 `y = w * x + b` 하나입니다. 여기서 `x`는 우리가 가진 입력이고, `w`와 `b`는 **파라미터** — 식 안에서 우리가 바꿀 수 있는 숫자입니다. 입력에서 출력까지 이 식을 한 번 계산하는 것을 **순전파**라고 합니다.

왜 이런 게 필요할까요. 데이터에서 규칙을 찾으려면 먼저 **규칙의 후보를 식으로 써야** 하기 때문입니다. "`x`가 커지면 `y`도 비례해서 커지는 것 같다"는 짐작을 `w * x + b`라고 적어두면, 남은 일은 `w`와 `b`에 들어갈 숫자를 찾는 것으로 좁혀집니다. 학습이란 이 두 숫자를 고쳐 나가는 작업 전체를 말합니다.

그래서 손이 가는 순간은 언제나 학습 루프의 첫 줄입니다. 데이터를 모델에 통과시키는 그 줄이 순전파입니다.

```python
w = torch.tensor(0.5, requires_grad=True)   # 파라미터 — 바뀔 숫자
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor([1., 2., 3.])              # 입력 — 고정된 데이터

y_pred = w * x + b                          # 순전파
print(y_pred)                               # tensor([0.5000, 1.0000, 1.5000], grad_fn=...)
```

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `w = torch.tensor(..., requires_grad=True)` | `nn.Linear(1, 1)` | 모델 정의부 |
| `y_pred = w * x + b` | `y_pred = model(x)` | 학습 루프 안 |

### B · 손실 함수 (6분)

**손실은 예측이 정답에서 얼마나 벗어났는지를 숫자 하나로 나타낸 값입니다.** 오늘 쓸 방식은 MSE(평균 제곱 오차)입니다. 예측과 정답의 차이를 각각 제곱해서 평균 냅니다.

이게 없으면 "지금 모델이 나아지고 있는가"를 판정할 방법이 없습니다. 데이터 100개에 대해 예측이 100개 나오는데, 그걸 눈으로 훑어서는 `w`를 0.5에서 0.6으로 올린 게 나아진 건지 나빠진 건지 알 수 없습니다. 숫자 하나로 줄여야 비교가 됩니다.

제곱하는 이유는 두 가지입니다. 부호를 없애서 +2와 −2를 똑같이 "2만큼 틀렸다"로 세고, 크게 틀린 것에 더 큰 벌점을 주기 위해서입니다.

그리고 여기가 중요한데, **손실은 반드시 스칼라(숫자 하나)여야 합니다.** 3일차에 비스칼라 텐서에 `backward()`를 걸어 `RuntimeError`를 냈던 셀을 기억하실 겁니다. 오늘 `.mean()`을 빼먹으면 정확히 그 에러가 다시 납니다.

```python
y = torch.tensor([3., 5., 7.])              # 정답

loss = ((y_pred - y) ** 2).mean()           # MSE
print(loss.shape)                           # torch.Size([]) ← 0차원, 스칼라
```

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `((y_pred - y) ** 2).mean()` | `criterion = nn.MSELoss()` → `criterion(y_pred, y)` | 순전파 바로 다음 줄 |

### C · 그래디언트 강하와 학습률 (7분)

**그래디언트 강하는 손실이 줄어드는 방향으로 파라미터를 조금씩 옮기는 방법입니다.** 방향은 이미 3일차에 구했습니다. `w.grad`가 그것입니다.

`w.grad`가 정확히 무슨 뜻인지 다시 짚겠습니다. **`w`를 아주 조금 늘렸을 때 손실이 얼마나 늘어나는가**입니다. 그러니까 `w.grad`가 양수라면 `w`를 늘릴수록 손실이 커진다는 뜻이고, 우리는 손실을 *줄이고* 싶으니 반대로 가야 합니다. `w.grad`가 음수라면 늘리는 쪽이 이득입니다. 두 경우를 한 줄로 쓰면 이렇게 됩니다.

```python
w = w - lr * w.grad
```

뺄셈인 이유가 여기 있습니다. grad가 양수면 빼서 줄이고, 음수면 빼는 게 곧 더하는 것이 되어 늘립니다. 부호 판단을 코드가 알아서 하는 셈입니다.

`lr`은 **학습률** — 한 번에 얼마나 옮길지를 정하는 값입니다. grad는 방향만 알려줄 뿐 얼마나 가야 하는지는 말해주지 않기 때문에 이 값이 따로 필요합니다. 너무 작으면 도착하기 전에 학습이 끝나고, 너무 크면 목표를 지나쳐 반대편 더 먼 곳에 착지합니다. 그게 반복되면 발산합니다. 오늘 2교시 Step 6에서 직접 폭주시켜 봅니다.

실제 코드에서는 이 갱신을 `torch.no_grad()`로 감쌉니다. 감싸지 않으면 "`w`를 갱신했다"는 연산까지 계산 그래프에 기록되고, 다음 스텝의 `backward()`가 갱신 과정까지 되짚으려 들기 때문입니다. 파라미터를 고치는 일은 학습의 결과지 학습의 대상이 아닙니다.

```python
lr = 0.1

with torch.no_grad():         # 이 안의 연산은 그래프에 기록되지 않는다
    w -= lr * w.grad          # in-place 로 고친다 (w = w - ... 가 아니라)
    b -= lr * b.grad
```

> ⚠️ `w -= ...`와 `w = w - ...`는 오늘 전혀 다르게 동작합니다. 앞은 `w`라는 텐서의 내용물을 고치고, 뒤는 **새 텐서를 만들어 `w`라는 이름을 거기 다시 붙입니다.** 새로 만들어진 텐서는 연산 결과라서 `.grad`가 채워지지 않습니다. 3교시 과제 2가 정확히 이 사고입니다.

### D · 학습 루프 (6분)

**학습 루프는 지금까지의 A·B·C를 순서대로 반복하는 구조입니다.** 한 바퀴를 **스텝**이라고 부릅니다. (전체 데이터를 한 바퀴 다 쓰는 것은 **에폭**이라고 하는데, 오늘은 데이터 100개를 매번 통째로 쓰므로 1스텝 = 1에폭입니다. 둘이 갈라지는 건 2주차 4일 DataLoader부터입니다.)

| 순서 | 코드 | 하는 일 |
|---|---|---|
| 1 | `w.grad.zero_()` | 지난 스텝 그래디언트 지우기 |
| 2 | `y_pred = w * x + b` | 순전파 |
| 3 | `loss = ((y_pred - y) ** 2).mean()` | 손실 계산 |
| 4 | `loss.backward()` | 역전파 — grad 채우기 |
| 5 | `w -= lr * w.grad` | 갱신 |

1번이 왜 맨 앞에 있는지가 오늘의 핵심입니다. 3일차에 확인했듯 **`.grad`는 덮어쓰지 않고 더합니다.** 지우지 않으면 2스텝째에는 1스텝의 grad가 그대로 얹힌 값으로 움직이고, 3스텝째에는 또 얹힙니다. 실효 학습률이 스텝마다 커지는 셈이라 결국 발산합니다.

지우는 위치는 5번 다음이어도 논리적으로는 같지만, **관례는 맨 앞**입니다. 루프에 들어가기 전 초기화를 잊어도 안전하고, 중간에 `continue`로 스텝을 건너뛰어도 다음 스텝이 오염되지 않기 때문입니다.

**실전 대응**

| 오늘 손으로 | 실전 | 나오는 곳 |
|---|---|---|
| `w.grad.zero_(); b.grad.zero_()` | `optimizer.zero_grad()` | 루프 첫 줄 |
| `loss.backward()` | 그대로 | 손실 다음 줄 |
| `with torch.no_grad(): w -= lr * w.grad` | `optimizer.step()` | 그다음 줄 |

2주차 3일에 옵티마이저를 배우면 이 세 줄이 저 세 줄로 바뀝니다. 오늘 손으로 써보는 이유는, 저 세 줄이 무엇을 대신해 주는지 모르면 학습이 안 될 때 어디를 볼지 알 수 없기 때문입니다.

---

## 2교시 (53분) · 실습 — `d4_train_loop.ipynb`

오늘은 **전부 CPU**로 합니다. 데이터가 100개뿐이라 GPU로 보내면 계산보다 전송이 더 오래 걸립니다 — 2일차 기준값 ②에서 64 MiB 전송에 0.04초가 걸렸는데, 오늘 데이터는 400 바이트입니다. GPU가 언제 이득인지는 2주차 4일에 배치 크기와 함께 다시 봅니다.

### Step 1 · 정답을 아는 문제 만들기 (5분)

> **볼 것** — `y`가 `3x + 2`에서 조금씩 어긋나 있다는 것. 그 어긋남이 잡음입니다
> **끝나면** — 학습이 끝났을 때 `w`와 `b`가 어디로 가야 정답인지 알고 시작한다

```python
import torch
torch.manual_seed(0)

x = torch.linspace(-3, 3, 100)            # 입력 100개
y_clean = 3.0 * x + 2.0                   # 찾아내야 할 진짜 규칙
y = y_clean + 0.5 * torch.randn(100)      # 관측값에는 잡음이 섞여 있다

print(x.shape, y.shape)
print("정답:  w = 3.0,  b = 2.0")
print(y[:5])
```

실전에서는 정답을 모르는 채로 학습합니다. 오늘 굳이 정답을 알고 시작하는 건, **루프가 제대로 도는지 판정할 기준이 필요해서**입니다. loss가 줄어드는 것만으로는 부족합니다 — 엉뚱한 값으로 수렴하면서도 loss는 줄 수 있습니다.

### Step 2 · 순전파와 손실 한 번 (8분)

> **볼 것** — `loss.shape`가 `torch.Size([])`라는 것, 그리고 `w.grad`가 아직 `None`이라는 것
> **끝나면** — 임의의 `w`, `b`로 예측을 만들고 얼마나 틀렸는지 숫자 하나로 잴 수 있다

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * x + b                        # 순전파
loss = ((y_pred - y) ** 2).mean()         # MSE

print("loss  :", loss.item())
print("shape :", loss.shape)
print("w.grad:", w.grad)                  # backward() 전이라 None
```

`w = b = 0`이면 모델은 무엇을 넣든 0을 예측합니다. 가장 무식한 출발점이고, 여기서 loss가 얼마인지가 오늘의 기준선입니다. 이 숫자를 적어두세요.

### Step 3 · 스텝 하나를 손으로 (12분 · PRIMM)

> **볼 것** — `w.grad`의 *부호*. 크기가 아니라 부호입니다
> **끝나면** — grad 부호를 보고 파라미터가 어느 쪽으로 움직일지 예측할 수 있다

**P 예측 (2분)** — Step 2 상태(`w = 0`, `b = 0`)에서 `loss.backward()`를 부르면 `w.grad`의 부호는?

① 양수  ② 음수  ③ 0  ④ `None`인 채로 남는다

노트북 마크다운 셀에 고른 번호와 **그렇게 생각한 이유 한 줄**을 적고 다음 셀로 넘어가세요.
힌트: `w`는 앞으로 0에서 3으로 *커져야* 합니다.

**R 실행 (1분)**

```python
loss.backward()

print("w.grad:", w.grad.item())
print("b.grad:", b.grad.item())
```

**I 조사 (4분)** — autograd가 낸 값이 손계산과 맞는지 확인합니다. MSE를 `w`로 미분하면 `mean(2 * (y_pred - y) * x)`입니다.

```python
with torch.no_grad():                     # 확인용 계산이므로 그래프를 만들지 않는다
    manual_gw = (2 * (y_pred - y) * x).mean()
    manual_gb = (2 * (y_pred - y)).mean()

print("손계산 w:", manual_gw.item(), " autograd:", w.grad.item())
print("손계산 b:", manual_gb.item(), " autograd:", b.grad.item())
```

두 값이 소수점 아래까지 같아야 합니다. 그리고 `b.grad`의 부호도 함께 보세요. `b`도 0에서 2로 커져야 하는데 부호가 `w.grad`와 같은지 다른지, 왜 그런지 한 줄로 적어두세요.

**M 수정 (3분)** — 한 스텝만 갱신하고 loss가 실제로 줄었는지 확인합니다.

```python
lr = 0.1

with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

y_pred = w * x + b
loss2 = ((y_pred - y) ** 2).mean()

print(f"{loss.item():.4f}  →  {loss2.item():.4f}")
print(f"w {w.item():.4f}   b {b.item():.4f}")
```

그다음 `lr`만 `0.5`로 바꿔서 이 셀을 처음부터 다시 돌려보세요. 한 스텝만에 `w`가 얼마나 더 갔는지, loss는 더 줄었는지 봅니다.

**실전 대응 (2분)**

| 방금 쓴 줄 | 실전 |
|---|---|
| `loss.backward()` | `loss.backward()` |
| `w -= lr * w.grad` (no_grad 안) | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

### Step 4 · 루프로 만들기 (12분)

> **볼 것** — loss가 단조롭게 줄어드는지, 그리고 `w`·`b`가 3.0·2.0으로 다가가는지
> **끝나면** — 학습 스크립트의 뼈대를 보지 않고 쓸 수 있다

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(100):
    if w.grad is not None:                # 첫 스텝에는 grad 가 아직 없다
        w.grad.zero_()
        b.grad.zero_()

    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

print(f"\n최종  w {w.item():.4f}   b {b.item():.4f}   (정답 3.0 / 2.0)")
```

`if w.grad is not None` 이 한 줄이 눈에 걸릴 겁니다. 실전에서는 이런 게 없습니다. `optimizer.zero_grad()`는 `grad`가 `None`이든 아니든 알아서 처리하기 때문입니다. 옵티마이저가 대신 해주는 일이 무엇인지 보여주는 자리라 오늘은 남겨둡니다.

`loss.item()`을 쓴 것도 의도적입니다. `loss`를 그대로 출력하거나 리스트에 담으면 그 텐서에 달린 계산 그래프가 통째로 살아남습니다. 3일차에 `sin` 5연쇄만으로 메모리가 3배(384 MiB)가 됐던 걸 떠올리면, 100스텝이면 어떻게 되는지 짐작이 갈 겁니다. `.item()`은 파이썬 `float` 하나만 꺼내오므로 그래프와의 연결이 끊깁니다.

> **여기까지가 오늘의 통과 기준 1번입니다.** `w`가 2.9~3.1, `b`가 1.9~2.1 안에 들어왔는지 확인하세요. 잡음 때문에 정확히 3.0·2.0이 되지는 않습니다 — 되면 오히려 이상합니다.

### Step 5 · 순서 맞추기 (8분)

> **볼 것** — 순서가 하나만 어긋나도 *에러 없이* 학습이 망가진다는 것
> **끝나면** — 남의 학습 루프에서 빠진 줄과 잘못 놓인 줄을 찾아낼 수 있다

아래 블록 일곱 개 중 **다섯 개**를 골라 학습 루프 한 스텝을 완성하세요. 두 개는 쓰면 안 되는 블록입니다. 노트북 마크다운 셀에 **순서(기호)** 와 **버린 블록 두 개를 왜 버렸는지** 각각 한 줄로 적으세요.

**(가)**
```python
loss.backward()
```

**(나)**
```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

**(다)**
```python
w.grad.zero_()
b.grad.zero_()
```

**(라)**
```python
with torch.no_grad():
    loss = ((y_pred - y) ** 2).mean()
```

**(마)**
```python
y_pred = w * x + b
```

**(바)**
```python
loss = ((y_pred - y) ** 2).mean()
```

**(사)**
```python
w -= lr * w.grad
b -= lr * b.grad
```

골랐으면 실제로 그 순서대로 셀에 옮겨 적어 100스텝 돌려보고, Step 4와 같은 결과가 나오는지 확인하세요.

### Step 6 · 학습률을 망가뜨리기 (8분 · PRIMM)

> **볼 것** — loss가 `inf`를 거쳐 `nan`이 되기까지 몇 스텝이 걸리는지
> **끝나면** — loss에 `nan`이 뜨면 학습률부터 의심할 수 있다

**P 예측 (2분)** — Step 4의 루프에서 `lr = 0.1`을 `lr = 1.0`으로만 바꾸면 어떻게 될까요?

① 10배 빠르게 3.0·2.0에 도달한다
② 비슷하게 수렴하되 값이 조금 흔들린다
③ loss가 폭주해 `inf`나 `nan`이 된다
④ `RuntimeError`가 난다

**R 실행 (1분)**

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 1.0

for step in range(40):
    if w.grad is not None:
        w.grad.zero_(); b.grad.zero_()

    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    print(f"step {step:2d}  loss {loss.item():>11.3e}  w {w.item():>11.3e}")
```

**I 조사 (3분)** — `w`가 스텝마다 어떻게 움직였는지 보세요. 부호가 매번 뒤집히면서 크기는 커지고 있을 겁니다. 목표를 지나쳐 반대편에 착지하는데, 착지 지점이 출발점보다 *더 멀어서* 다음 스텝의 grad는 더 커집니다. 그게 반복되면 float32가 표현할 수 있는 범위를 넘어 `inf`가 되고, `inf - inf`가 나오는 순간 `nan`이 됩니다.

한 번 `nan`이 되면 그 뒤 모든 연산이 `nan`입니다. **그래서 loss 로그는 첫 `nan`이 뜬 스텝을 찾는 게 중요합니다.** 마지막 줄만 보면 언제부터 망가졌는지 알 수 없습니다.

**M 수정 (2분)** — `lr`을 `0.5` → `0.4` → `0.3`으로 내려가며 **발산하지 않는 가장 큰 값**을 찾으세요. 그 값과, 그때 100스텝 후 `w`가 어디에 있는지를 노트북에 적어두세요.

실전에서 `lr`은 `optim.SGD(model.parameters(), lr=0.01)`처럼 옵티마이저를 만들 때 넘깁니다. 자동으로 조절하는 방법(스케줄러)은 5주차 4일에 다룹니다.

---

## 3교시 (20분) · 실패 케이스

먼저 고쳐보고, 그다음에 표를 보세요. 표를 먼저 읽으면 답을 찾는 게 되고, 그러면 남지 않습니다.

### 과제 1 · 에러는 없는데 학습이 안 된다 (5분)

```python
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(50):
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    if step % 10 == 0:
        print(f"step {step:2d}  loss {loss.item():.4f}  w {w.item():.3f}")
```

이 루프는 예외 없이 50번 돕니다. 그런데 `w`가 3.0으로 가지 않습니다. **무엇이 빠졌고, 그것 때문에 무슨 일이 일어나는지** 한 문장으로 쓰고 고치세요. 그리고 **어느 출력을 보고 그렇게 판단했는지**도 한 줄 적으세요.

### 과제 2 · 첫 스텝 마지막 줄에서 터진다 (5분)

```python
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(3):
    loss = ((w * x - y) ** 2).mean()
    loss.backward()

    w = w - lr * w.grad          # ← 갱신
    w.grad.zero_()               # ← 여기서 AttributeError
```

`AttributeError: 'NoneType' object has no attribute 'zero_'`가 납니다. `backward()`를 분명히 불렀는데 왜 `w.grad`가 `None`일까요. **한 줄만 고쳐서** 돌아가게 만드세요.

### 참조표 (10분)

| 증상 | 원인 | 해결 |
|---|---|---|
| loss가 줄다가 커지고, 스텝이 갈수록 진폭이 커진다 | `zero_()` 누락 — 그래디언트가 누적돼 실효 학습률이 스텝마다 커진다 | 루프 첫 줄에서 지운다. 실전에서는 `optimizer.zero_grad()` |
| `AttributeError: 'NoneType' object has no attribute 'zero_'` | `backward()`를 아직 안 불렀거나, `w = w - ...`로 재대입해 `w`가 leaf 텐서가 아니게 됐다 | 갱신은 `with torch.no_grad(): w -= ...`로 in-place. 첫 스텝은 `if w.grad is not None`으로 건너뛴다 |
| `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` | `no_grad()` 없이 `w -= ...`를 했다 | 갱신 전체를 `with torch.no_grad():`로 감싼다 |
| loss가 몇 스텝 만에 `inf` → `nan` | 학습률이 너무 크다 | 1/10로 줄여 재실행. 그래도 나면 입력 데이터의 스케일을 의심한다(7주차 2일 정규화) |
| 100스텝을 돌려도 loss가 거의 안 움직인다 | 학습률이 너무 작다 | 10배씩 키우며 발산 직전을 찾는다 |
| `RuntimeError: grad can be implicitly created only for scalar outputs` | 손실에 `.mean()`을 빠뜨려 벡터 상태로 `backward()`를 걸었다 | 스칼라로 줄인 뒤 호출한다 (3일차 셀 13과 같은 에러) |
| 학습이 길어질수록 메모리가 계속 늘어난다 | `losses.append(loss)`로 텐서를 그대로 담아 계산 그래프가 스텝 수만큼 살아 있다 | `losses.append(loss.item())`로 파이썬 `float`만 담는다 |
| 매번 실행할 때마다 최종 `w`가 조금씩 다르다 | 잡음 생성과 초기값이 난수라 실행마다 달라진다 | 정상이다. 시드를 고정하면 재현된다 — 8주차 3일에서 정식으로 |

---

## 마무리 (15분) · 통과 기준

1. `lr = 0.1`로 100스텝 학습 후 `w`가 **2.9~3.1**, `b`가 **1.9~2.1** 범위에 들어온다
2. Step 3에서 `w.grad`의 부호를 예측한 것과 실측이 일치한다. 틀렸다면 **왜 틀렸는지 한 줄**이 노트북에 적혀 있다
3. Step 3 I단계의 손계산 값과 autograd 값이 일치한다
4. `zero_()`를 뺐을 때(과제 1) 학습이 망가지는 것을 재현하고, 원인을 "그래디언트가 누적돼 실효 학습률이 커진다"로 설명할 수 있다
5. `lr = 1.0`에서 loss가 `inf`/`nan`이 되는 것을 재현하고, 발산하지 않는 최대 `lr`을 찾아 적었다
6. Step 5 순서 배열이 맞고, 버린 블록 두 개의 이유가 각각 한 줄로 적혀 있다

### 오늘 쓴 코드가 2주차에 무엇이 되는가

| 오늘 손으로 | 2주차부터 | 언제 |
|---|---|---|
| `w`, `b`를 직접 만들고 `requires_grad` 지정 | `nn.Linear(1, 1)` | 2주차 1일 |
| `y_pred = w * x + b` | `y_pred = model(x)` | 2주차 1일 |
| `((y_pred - y) ** 2).mean()` | `nn.MSELoss()` | 2주차 2일 |
| `w.grad.zero_(); b.grad.zero_()` | `optimizer.zero_grad()` | 2주차 3일 |
| `loss.backward()` | 그대로 | — |
| `with torch.no_grad(): w -= lr * w.grad` | `optimizer.step()` | 2주차 3일 |
| `for step in range(100)` — 매번 전체 데이터 | `for batch in dataloader` | 2주차 4일 |

오늘 만든 루프는 앞으로 29주 동안 형태만 바뀌고 골격은 그대로입니다. ResNet을 파인튜닝하든 LoRA를 붙이든 저 다섯 단계입니다.

> **다음 세션** — 1주차 5일 · 병행① LLM API 첫 호출. 메인 트랙에서 잠깐 벗어나 API 키 관리와 요청/응답 구조, 토큰과 비용을 다룹니다. 오늘보다 가볍습니다. 이후 8회에 걸쳐 학습 보조 도구를 하나 만들어 갑니다.

---

## 확인 퀴즈 (6문항)

답만 채팅으로 보내주세요 (예: `1-② 2-④ ...`). 정답과 해설은 그때 공개합니다.

**1.** 다음 루프는 에러 없이 100번 돌지만 `w`가 3.0 근처로 가지 않는다. 가장 가능성 높은 원인은?

```python
for step in range(100):
    y_pred = w * x + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
```

① 학습률이 너무 작다
② 그래디언트를 지우지 않아 스텝마다 누적된다
③ `no_grad()`의 위치가 잘못됐다
④ 순전파가 루프 안에 있어서 매번 다시 계산된다

**2.** `w.grad`가 음수로 나왔다. 이것이 뜻하는 바는?

① `w`를 줄이면 손실이 줄어든다
② `w`를 늘리면 손실이 줄어든다
③ `w`가 이미 최적값에 도달했다
④ 학습률이 너무 커서 부호가 뒤집혔다

**3.** 파라미터 갱신을 `with torch.no_grad():`로 감싸는 이유로 가장 정확한 것은?

① 뺄셈 연산이 더 빨라지기 때문
② 갱신 연산이 그래프에 기록되면, 다음 스텝의 `backward()`가 갱신 과정까지 되짚어 원하지 않는 그래디언트가 생기기 때문
③ `w.grad`를 읽으려면 `no_grad()` 안이어야 하기 때문
④ CPU 텐서는 `no_grad()` 밖에서 in-place 수정이 불가능하기 때문

**4.** `losses.append(loss)`와 `losses.append(loss.item())`의 실질적 차이는?

① 없다. 출력 형식만 다르다
② 앞쪽은 계산 그래프를 통째로 붙잡아, 스텝이 늘수록 메모리가 계속 증가한다
③ 앞쪽이 더 정밀한 값을 담는다
④ 뒤쪽은 GPU 텐서에는 쓸 수 없다

**5.** 학습률을 0.1에서 1.0으로 올렸더니 loss가 몇 스텝 만에 `nan`이 됐다. 무슨 일이 일어난 것인가?

① 그래디언트가 0이 되어 0으로 나누는 연산이 발생했다
② 한 스텝의 이동폭이 커서 최소점을 지나쳐 더 먼 곳에 착지하고, 그게 반복되며 값이 폭주했다
③ float32의 소수점 정밀도가 부족해 반올림 오차가 쌓였다
④ 원본 데이터 `y`에 `nan`이 섞여 있었다

**6.** 다음 코드가 첫 스텝 마지막 줄에서 `AttributeError: 'NoneType' object has no attribute 'zero_'`를 내는 이유는?

```python
loss.backward()
w = w - lr * w.grad
w.grad.zero_()
```

① `backward()` 직후 `.grad`는 자동으로 `None`이 되기 때문
② `w = w - ...`가 새 텐서를 만들었고, 그 텐서는 연산 결과라 `.grad`가 채워지지 않기 때문
③ `zero_()`는 `no_grad()` 안에서만 호출할 수 있기 때문
④ `lr`이 파이썬 `float`이라 텐서 타입이 깨졌기 때문
