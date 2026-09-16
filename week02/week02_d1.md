# 2주차 1일 · `nn.Module`

**배정 시간: 2시간** · 산출물: `week02/d1_module.ipynb`

1주차 4일에 손으로 쓴 학습 루프를, 오늘부터 실전 코드의 모양으로 바꿔 나간다. 4일차 마무리에 예고한 대응표의 첫 줄(`w`, `b` → `nn.Linear`)을 오늘 채운다.

| 교시 | 시간 |
|---|---|
| 도입 · 누적 복습 | 10분 |
| 오늘 얻는 것 | 2분 |
| 1교시 · 개념 | 25분 |
| 2교시 · 실습 | 48분 |
| 3교시 · 실패 케이스 | 20분 |
| 마무리 | 15분 |

---

## 도입 (10분) · 누적 복습 퀴즈 3문항

> 1주차 4일 메모에 「2주차 1일 도입부에서 `backward()`·`.grad`를 반드시 재확인」이 남아 있어 도입을 10분으로 잡았고, 2교시를 53분 → 48분으로 줄였습니다.

**답을 적어서 채팅으로 보내세요.** 채점과 해설은 채팅에서 합니다. 한 문항에 1분을 넘기지 마세요.

### 복습 1

```python
losses = []
for step in range(1000):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    losses.append(loss)      # ← 이 줄
    ...
```

`losses.append(loss)`로 1000스텝을 돌렸다. 무슨 일이 일어나는가?

1. 텐서는 파이썬 리스트에 담을 수 없어 `TypeError`가 난다
2. 정상 동작하지만, 각 손실을 만든 계산 기록이 함께 살아남아 메모리가 계속 는다
3. 정상 동작하고 메모리도 문제없다. `.item()`은 출력 형식을 예쁘게 하려고 쓰는 것이다
4. `loss`가 스칼라라서 파이썬 float으로 자동 변환되어 담긴다

### 복습 2

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
out = y.backward()
print(out, x.grad)
```

출력은?

1. `tensor(12.) tensor(12.)`
2. `None tensor(12.)`
3. `None None`
4. `tensor(8.) tensor(12.)`

### 복습 3

```python
a = torch.arange(6).reshape(2, 3)
b = a.t().reshape(6)
c = a.reshape(6)
print(b.data_ptr() == a.data_ptr(), c.data_ptr() == a.data_ptr())
```

출력은?

1. `True True`
2. `True False`
3. `False True`
4. `False False`

---

## 오늘 얻는 것 (2분)

| 배우는 것 | 이걸로 할 수 있게 되는 일 | 중요도 |
|---|---|---|
| `nn.Module` 상속 · `__init__`과 `forward` | 남의 모델 코드를 열었을 때 어디가 구조 정의이고 어디가 계산인지 구분해서 읽힌다 | ★★★ 필수 |
| `nn.Parameter` 등록 원리 | 학습이 안 되는데 에러도 안 날 때, `len(list(model.parameters()))` 한 줄로 원인을 판별한다 | ★★★ 필수 |
| `state_dict` | 체크포인트 파일을 열어 키와 shape만 보고 그게 어떤 구조의 모델인지 읽는다 | ★★★ 필수 |
| `torch.save` / `load_state_dict` | 학습한 모델을 껐다 켜서 이어 쓴다. GPU에서 저장한 것을 CPU에서 연다 | ★★★ 필수 |
| 손실을 기록할 때 `.item()`을 쓰는 이유 | 검증 루프에서 VRAM이 왜 터지는지 진단하고, 세 가지 기록 방식의 메모리 차이를 숫자로 댄다 | ★★★ 필수 |
| `model(x)`가 `forward`를 부르는 경로 | `model.forward(x)`를 직접 부르면 안 되는 이유를 말할 수 있다 | ★★ 권장 · **5주차 3일**(BatchNorm)에서 실제로 물린다 |
| `nn.ModuleList` · 중첩 모듈의 키 이름 | 층을 리스트에 담았더니 일부만 학습되는 상황을 진단한다 | ★★ 권장 · **7주차 3일**(ResNet head 교체)에서 쓴다 |
| `load_state_dict(strict=False)` | 사전학습 가중치를 부분만 가져올 때 무엇이 빠졌는지 확인한다 | ★ 참고 · **7주차 3일**에 정식으로 다룬다 |

> **시간이 부족하면 ★부터 버린다.**

---

## 1교시 (25분) · 개념

### 개념 A (6분) · `nn.Module` — 파라미터와 계산을 한 덩어리로 묶는 상자

**무엇인가.** 모델을 만들 때 상속받는 파이썬 클래스입니다. 안에 **값**(학습되는 숫자들)과 **계산 방법**(그 숫자로 무엇을 하는지)을 함께 담습니다.

**왜 필요한가.** 1주차 4일에는 `w`와 `b`를 각각 변수로 들고 다녔습니다. 변수 두 개라 괜찮았죠. 그런데 층이 10개면 변수가 20개가 됩니다. 학습 루프마다 20개를 하나씩 `zero_()` 하고, 저장할 때 20개를 챙기고, 다른 파일로 옮길 때 또 20개를 챙겨야 합니다. **하나를 빠뜨려도 에러가 안 납니다.** 그냥 그 값만 조용히 학습이 안 됩니다. 이 실수를 구조적으로 막는 것이 `nn.Module`입니다.

**언제 쓰는가.** 학습되는 값을 하나라도 가진 것은 전부입니다. 모델 전체도 `nn.Module`이고, 그 안의 층 하나하나도 `nn.Module`입니다. 같은 타입이 중첩됩니다.

**최소 예시**

```python
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()              # 이 줄이 없으면 등록 장치가 준비되지 않는다
        self.linear = nn.Linear(1, 1)   # 구조: 무엇을 가지고 있는가

    def forward(self, x):
        return self.linear(x)           # 계산: 입력이 들어오면 무엇을 하는가
```

규칙은 두 개뿐입니다.

- `__init__`에는 **가진 것**을 적는다. `super().__init__()`을 맨 위에 반드시 쓴다
- `forward`에는 **하는 일**을 적는다

그리고 쓸 때는 `model.forward(x)`가 아니라 **`model(x)`**로 부릅니다. `nn.Module`에는 **후크(hook)** 라는 장치가 있습니다 — 계산 앞이나 뒤에 "이것도 같이 해라"라고 끼워 넣어 둔 추가 동작입니다. 중간 결과를 꺼내 보거나, 층의 동작을 바꿔 끼울 때 씁니다. `model(x)`는 후크까지 함께 실행하지만, `forward`를 직접 부르면 후크가 건너뛰어집니다. 지금은 후크를 쓰지 않아 결과가 같지만, 5주차 3일 BatchNorm과 9주차 파인튜닝에서 이 차이가 실제로 문제가 됩니다. **지금부터 `model(x)`로 부르는 습관을 들입니다.**

**실전 대응**

| 1주차 4일 손으로 | 오늘 | 나오는 곳 |
|---|---|---|
| `w = torch.randn(1, requires_grad=True)`<br>`b = torch.randn(1, requires_grad=True)` | `self.linear = nn.Linear(1, 1)` | 모델 클래스의 `__init__` |
| `pred = x * w + b` | `return self.linear(x)` | 모델 클래스의 `forward` |
| `pred = x * w + b` (루프 안에서) | `pred = model(x)` | 학습 루프의 첫 줄 |

### 개념 B (7분) · `nn.Parameter` — 등록되어야 파라미터다

**무엇인가.** "이 텐서는 이 모델이 학습할 값이다"라고 표시하는 껍데기입니다. 텐서를 `nn.Parameter(...)`로 감싸서 속성에 대입합니다.

**왜 필요한가.** `nn.Module`에는 `parameters()`라는 메서드가 있습니다. 이 모델이 가진 학습 대상을 전부 모아서 내주는 통로이고, 2주차 3일에 배울 옵티마이저는 **이 통로로 받은 것만** 갱신합니다.

문제는 파이썬에서는 아무 객체나 속성에 대입할 수 있다는 점입니다. `self.w = torch.randn(1)`도 문법상 아무 문제가 없습니다. 그래서 `nn.Module`은 속성에 뭔가 대입될 때마다 **타입을 검사해서** `nn.Parameter`와 `nn.Module`만 골라 등록 목록에 넣습니다. 일반 텐서는 그냥 평범한 속성이 되고, `parameters()`에 나오지 않습니다.

여기서 헷갈리기 쉬운 지점 하나를 미리 못 박습니다. **`requires_grad=True`와 "모델의 파라미터다"는 별개입니다.**

- `requires_grad=True` → 이 텐서에 대한 미분을 추적한다 (1주차 3일)
- `nn.Parameter` → 이 텐서는 이 모델의 학습 대상 목록에 들어간다 (오늘)

일반 텐서에 `requires_grad=True`를 붙여 속성에 대입하면, **그래디언트는 정상적으로 계산됩니다.** `.grad`도 채워집니다. 그런데 `parameters()`에는 없으니 옵티마이저가 그 값을 못 봅니다. 그래서 에러 없이 손실만 제자리인 상태가 됩니다. 이것이 오늘 3교시 첫 번째 과제입니다.

**언제 쓰는가.** 직접 만든 텐서를 학습시킬 때만 씁니다. `nn.Linear` 같은 기성 층은 내부에서 이미 `nn.Parameter`를 쓰고 있고, 그 층을 속성에 대입하면 안에 든 파라미터까지 재귀적으로 딸려옵니다.

**최소 예시**

```python
self.w = torch.randn(1, requires_grad=True)   # 등록 안 됨
self.w = nn.Parameter(torch.randn(1))         # 등록됨 (requires_grad=True가 기본값)
```

**실전 대응**

| 상황 | 쓰는 것 | 확인 방법 |
|---|---|---|
| 기성 층을 쌓는다 | `self.fc = nn.Linear(...)` | `named_parameters()`에 `fc.weight`가 보인다 |
| 층을 여러 개 담는다 | `nn.ModuleList([...])` · `nn.Sequential(...)` | 파이썬 `list`는 등록 안 된다 |
| 직접 만든 텐서를 학습시킨다 | `nn.Parameter(...)` | `len(list(model.parameters()))` |
| 학습시키지 않지만 저장은 하고 싶다 | `register_buffer(...)` | ★ 참고 · **5주차 3일** BatchNorm의 평균·분산이 이것이다 |

### 개념 C (6분) · `state_dict` — 구조는 빼고 값만 꺼낸 딕셔너리

**무엇인가.** `{이름: 텐서}` 형태의 딕셔너리입니다. 모델이 가진 학습된 값들을 이름표와 함께 꺼내 줍니다. **구조는 들어 있지 않습니다.** 층이 몇 개인지, 어떻게 연결되는지는 없고 숫자만 있습니다.

**왜 필요한가.** 모델을 저장한다는 것은 결국 그 숫자들을 파일에 적는 일입니다. 구조는 파이썬 코드에 이미 있으니 다시 저장할 필요가 없고, 오히려 구조까지 통째로 저장하면 그 클래스가 정의된 파일 경로가 저장 파일에 박혀서 다른 환경에서 열리지 않습니다.

**언제 쓰는가.** 저장할 때, 로드할 때, 그리고 7주차 이후 전이학습에서 남의 가중치 중 일부만 가져올 때입니다.

**최소 예시**

```python
sd = model.state_dict()
for k, v in sd.items():
    print(k, tuple(v.shape))
# linear.weight (1, 1)
# linear.bias   (1,)
```

키 이름은 **속성 이름을 점으로 이은 경로**입니다. `self.linear`이라는 속성 안의 `weight`이므로 `linear.weight`입니다. 모델이 중첩되면 경로도 계속 이어집니다.

한 가지 함정을 미리 적어 둡니다. `state_dict()`가 주는 텐서는 복사본이 아니라 **모델 안의 그 텐서 자체**입니다. 그래서 학습 도중 `best = model.state_dict()`로 받아 두고 학습을 계속하면, `best` 안의 값도 같이 바뀝니다. 파일로 저장할 때는 그 순간의 값이 파일에 적히므로 안전하고, **메모리에 들고 있을 때만** 문제가 됩니다. 3주차 2일 early stopping에서 바로 마주칩니다.

**실전 대응**

| 하려는 일 | 쓰는 코드 | 나오는 곳 |
|---|---|---|
| 이 체크포인트가 어떤 구조인지 확인 | `[(k, tuple(v.shape)) for k, v in sd.items()]` | 남의 가중치를 처음 받았을 때 |
| 내 모델 키와 파일 키를 대조 | `set(model.state_dict()) - set(sd)` | 로드가 실패했을 때 첫 진단 |
| 최고 성능 시점 가중치를 메모리에 보관 | `copy.deepcopy(model.state_dict())` | 3주차 2일 early stopping |

### 개념 D (6분) · `torch.save` / `load_state_dict` — 껐다 켜기

**무엇인가.** `torch.save`는 파이썬 객체를 파일로 적고, `load_state_dict`는 딕셔너리의 값을 모델에 **밀어 넣습니다.**

**왜 필요한가.** 7주차부터는 한 번 학습에 수십 분이 걸립니다. 중간 결과를 저장하지 않으면 노트북을 닫는 순간 전부 날아갑니다.

**언제 쓰는가.** 학습이 끝났을 때, 그리고 7주차 4일부터는 에폭마다 체크포인트로 저장합니다.

**최소 예시**

```python
# 저장 — 값만
torch.save(model.state_dict(), "d1_model.pt")

# 로드 — 구조를 먼저 만들고, 그 안에 값을 넣는다
model2 = LinearModel()
model2.load_state_dict(torch.load("d1_model.pt", weights_only=True))
```

**로드는 2단계입니다.** 파일에는 값만 있으므로, 같은 구조의 모델을 먼저 만들어야 넣을 곳이 생깁니다.

그리고 `load_state_dict`는 **제자리에서 모델을 바꾸는(in-place) 함수**입니다. 반환값은 모델이 아니라 "빠진 키 / 남는 키" 목록을 담은 작은 객체입니다. `model = model.load_state_dict(...)`라고 쓰면 모델이 그 목록 객체로 덮어써집니다. 3교시 두 번째 과제가 이것입니다.

**실전 대응**

| 하려는 일 | 쓰는 코드 |
|---|---|
| 저장 (권장) | `torch.save(model.state_dict(), path)` |
| 저장 (비권장) | `torch.save(model, path)` — 클래스 정의 경로가 파일에 박힌다 |
| 로드 | `m = Model()` → `m.load_state_dict(torch.load(path, weights_only=True))` |
| GPU에서 저장한 것을 CPU에서 로드 | `torch.load(path, map_location="cpu")` |

> `weights_only=True`는 파일에서 텐서만 읽고 임의의 파이썬 객체는 실행하지 않겠다는 뜻입니다. 최신 torch에서 기본값이며, `state_dict`만 저장했다면 그대로 두면 됩니다.

---

## 2교시 (48분) · 실습

`week02/d1_module.ipynb`를 새로 만들고 시작합니다. **Step 1~4는 CPU로 진행합니다.** 데이터가 작아 GPU로 보내면 전송이 계산보다 비쌉니다(1주차 2일 기준값 ②). Step 5만 GPU를 씁니다.

### Step 1 — `nn.Module`을 상속해 선형회귀 모델 정의하기 (7분)

> **볼 것** — `print(model)`이 `__init__`에 적은 구조를 그대로 출력한다는 것, 그리고 `named_parameters()`에 `linear.weight` / `linear.bias` 두 개가 나온다는 것
> **끝나면** — 모델 클래스를 보고 어느 줄이 구조이고 어느 줄이 계산인지 짚을 수 있다
> **쓰는 상황** — 모든 PyTorch 모델 코드의 출발점. 7주차 이후 남의 코드를 읽을 때도 이 두 메서드부터 찾는다

| 새로 나온 것 | 하는 일 |
|---|---|
| `nn.Module` | 상속받으면 파라미터 등록·저장·로드 기능이 딸려온다 |
| `super().__init__()` | 그 등록 장치를 준비시킨다. 빼먹으면 에러가 난다 |
| `nn.Linear(in, out)` | `입력 @ weight.T + bias`를 하는 층. `weight`와 `bias`를 안에 들고 있다 |
| `model.named_parameters()` | 등록된 파라미터를 `(이름, 텐서)` 쌍으로 하나씩 내준다 |

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
print(model)
```

```python
for name, p in model.named_parameters():
    print(f"{name:16s} shape={tuple(p.shape)}  requires_grad={p.requires_grad}")
```

```python
x = torch.linspace(-3, 3, 100).unsqueeze(1)   # (100, 1)
print("x     :", tuple(x.shape))
print("out   :", tuple(model(x).shape))
```

`nn.Linear(1, 1)`의 `weight` shape가 `(1, 1)`인데 왜 `(출력, 입력)` 순서인지는 6주차 1일 선형대수 보충에서 다룹니다. 지금은 **왼쪽이 출력 개수**라는 것만 기억하세요.

### Step 2 — 파라미터가 등록되는 경우와 안 되는 경우 비교하기 (11분) · PRIMM

> **볼 것** — 두 클래스가 거의 같은 코드인데 `parameters()` 결과가 다르다는 것
> **끝나면** — 학습이 안 되는데 에러도 안 나는 상황에서 원인을 한 줄로 판별할 수 있다
> **쓰는 상황** — 커스텀 층을 직접 만들 때. 9주차 파인튜닝에서 층을 갈아 끼울 때

#### P · 예측 (2분)

아래 두 클래스를 읽고, **실행하기 전에** `len(list(a.parameters()))`와 `len(list(b.parameters()))`가 각각 몇으로 나올지 예측해서 노트북 마크다운 셀에 적으세요.

```python
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(1, requires_grad=True)
        self.b = torch.randn(1, requires_grad=True)
    def forward(self, x):
        return x * self.w + self.b

class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return x * self.w + self.b
```

1. A = 2, B = 2
2. A = 0, B = 2
3. A = 2, B = 0
4. A는 정의 단계에서 에러가 난다

#### R · 실행 (1분)

```python
a, b = A(), B()
print("A:", len(list(a.parameters())))
print("B:", len(list(b.parameters())))
```

#### I · 조사 (4분)

예측과 맞았든 틀렸든, 아래를 실행해서 **A의 `w`에 그래디언트가 계산되는지**를 확인하세요.

```python
loss_a = ((a(x) - (3 * x + 2)) ** 2).mean()
loss_a.backward()

print("a.w 는 텐서인가      :", type(a.w).__name__)
print("a.w.requires_grad   :", a.w.requires_grad)
print("a.w.grad            :", a.w.grad)          # ← 여기를 보세요
print("parameters() 개수    :", len(list(a.parameters())))
```

`a.w.grad`에 숫자가 들어 있는데 `parameters()`는 0입니다. 두 가지가 별개라는 뜻입니다.

- `requires_grad=True`라서 **미분은 계산된다**
- `nn.Parameter`가 아니라서 **학습 대상 목록에는 없다**

`nn.Module`은 속성에 뭔가 대입될 때 타입을 보고 `nn.Parameter`와 `nn.Module`만 목록에 넣습니다. 일반 텐서는 그냥 평범한 파이썬 속성이 됩니다.

#### M · 수정 (3분)

한 줄만 바꿔서 A를 고치세요. 그리고 아래 두 클래스로 **리스트에 담으면 어떻게 되는지**도 확인하세요.

```python
class C(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(1, 1), nn.Linear(1, 1)]        # 파이썬 list

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])

print("C:", len(list(C().parameters())))
print("D:", len(list(D().parameters())))
```

#### 실전 대응 (1분)

| 증상 | 한 줄 진단 |
|---|---|
| 손실이 안 떨어지는데 에러도 없다 | `len(list(model.parameters()))`를 찍어 본다 |
| 층을 여러 개 만들었는데 일부만 학습된다 | `[name for name, _ in model.named_parameters()]`에 빠진 층이 있는지 본다 |

### Step 3 — 4일차 학습 루프를 `nn.Module` 버전으로 바꾸기 (10분) · Parsons

> **볼 것** — 4일차에 손으로 쓴 5줄이 어느 줄로 바뀌었는지, 그리고 `model.parameters()` 순회가 그중 두 줄을 대신한다는 것
> **끝나면** — 4일차와 같은 결과(`w≈3`, `b≈2`)를 `nn.Module` 버전으로 재현한다
> **쓰는 상황** — 2주차 3일에 이 루프의 마지막 두 줄이 `optimizer`로 한 번 더 줄어든다

#### 배열 과제 (4분)

아래 블록 중 **5개를 골라** 학습 루프 한 스텝의 올바른 순서로 배열하세요. 나머지 2개는 넣으면 안 되는 블록입니다. **배제한 두 블록에 대해 왜 안 되는지 한 줄씩 적으세요.**

```
(가)  loss.backward()

(나)  with torch.no_grad():
          for p in model.parameters():
              p -= lr * p.grad

(다)  pred = model(x)

(라)  model.zero_grad()

(마)  loss = loss.backward()

(바)  loss = ((pred - y) ** 2).mean()

(사)  with torch.no_grad():
          for p in model.parameters():
              p = p - lr * p.grad
```

| 새로 나온 것 | 하는 일 |
|---|---|
| `model.zero_grad()` | 모델이 가진 모든 파라미터의 `.grad`를 한 번에 비운다. 4일차의 `w.grad.zero_(); b.grad.zero_()`를 대신한다 |

#### 실행 (6분)

배열한 순서대로 아래 셀을 완성해서 돌리세요. 빈칸 세 곳을 채웁니다.

```python
torch.manual_seed(0)
model = LinearModel()

x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = 3 * x + 2
lr = 0.1

losses = []
for step in range(100):
    model.zero_grad()
    pred = ______                       # (1)
    loss = ______                       # (2)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            ______                      # (3)
    losses.append(loss.item())

print("w =", model.linear.weight.item())
print("b =", model.linear.bias.item())
print("첫 손실 =", losses[0], " 마지막 손실 =", losses[-1])
```

`w`가 2.9~3.1, `b`가 1.9~2.1로 나오면 4일차와 같은 결과입니다.

> `losses.append(loss.item())`에 `.item()`이 붙은 이유는 Step 5에서 숫자로 확인합니다.

### Step 4 — `state_dict`를 저장하고 새 인스턴스에 로드하기 (10분) · PRIMM

> **볼 것** — `state_dict()`의 키 이름이 속성 이름에서 나온다는 것, 그리고 `load_state_dict`의 반환값이 모델이 아니라는 것
> **끝나면** — 저장한 파일을 새 인스턴스에 넣어 같은 입력에 같은 출력을 내게 만든다
> **쓰는 상황** — 7주차 4일 체크포인트 저장. 7주차 3일 사전학습 가중치 로드

#### P · 예측 (2분)

Step 1에서 만든 `LinearModel`의 `state_dict()` 키는 무엇으로 나올까요? 실행 전에 적으세요.

1. `['w', 'b']`
2. `['linear.weight', 'linear.bias']`
3. `['weight', 'bias']`
4. `['0.weight', '0.bias']`

#### R · 실행 (1분)

```python
sd = model.state_dict()
print(type(sd).__name__)
for k, v in sd.items():
    print(f"{k:20s} {tuple(v.shape)}  {v.flatten().tolist()}")
```

#### I · 조사 (3분)

키는 **속성 이름을 점으로 이은 경로**입니다. `self.linear`이라는 속성이 가진 `weight`이므로 `linear.weight`입니다. 중첩되면 계속 이어집니다.

```python
class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x):
        return self.block(x)

print(list(Nested().state_dict().keys()))
```

출력에서 **숫자 하나가 건너뛰어진 것**을 확인하고, 왜인지 생각해 보세요.

#### M · 수정 (4분)

저장하고, 새 인스턴스에 넣고, 같은 출력이 나오는지 확인합니다.

| 새로 나온 것 | 하는 일 |
|---|---|
| `torch.save(obj, path)` | 객체를 파일로 적는다 |
| `torch.load(path, weights_only=True)` | 파일에서 텐서만 읽어 온다 |
| `model.load_state_dict(sd)` | 딕셔너리의 값을 모델에 제자리로 밀어 넣는다 |

```python
torch.save(model.state_dict(), "d1_model.pt")

model2 = LinearModel()
print("로드 전 최대 차이 :", (model2(x) - model(x)).abs().max().item())

ret = model2.load_state_dict(torch.load("d1_model.pt", weights_only=True))
print("반환값            :", ret)
print("반환값 타입        :", type(ret).__name__)
print("로드 후 최대 차이 :", (model2(x) - model(x)).abs().max().item())
```

**반환값이 무엇인지 반드시 눈으로 확인하세요.** 모델이 아닙니다.

### Step 5 — 손실을 모으는 세 방식의 VRAM 증가량 재기 (10분)

> **볼 것** — 세 방식의 증가량이 몇 배 차이로 찍히는지
> **끝나면** — 검증 루프에서 VRAM이 왜 터지는지 설명하고, 그 자리에 무엇을 써야 하는지 말할 수 있다
> **쓰는 상황** — 3주차 1일 검증 루프를 짤 때. 6주차 3일 학습 디버깅 체크리스트

이 Step만 GPU를 씁니다. 텐서를 키워야 차이가 보이기 때문입니다.

2일차에 배운 규칙 — 메모리 측정은 커널 재시작 직후 1회 — 대신, 여기서는 매 측정마다 정리하고 재는 함수를 씁니다. 앞으로도 쓸 패턴입니다.

| 새로 나온 것 | 하는 일 |
|---|---|
| `gc.collect()` | 파이썬이 붙잡고 있던 객체를 즉시 정리한다 |
| `torch.cuda.empty_cache()` | PyTorch가 캐시로 쥐고 있던 VRAM을 드라이버에 돌려준다 |

```python
import gc, torch, torch.nn as nn

dev = "cuda"
x = torch.randn(65536, 1, device=dev)
y = 3 * x + 2
model = nn.Linear(1, 1).to(dev)
N = 300

def measure(fn):
    gc.collect(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    kept = fn()
    used = torch.cuda.memory_allocated() - base
    del kept
    gc.collect(); torch.cuda.empty_cache()
    return used / 1024**2
```

```python
def case_a():                      # 텐서를 그대로 담는다
    out = []
    for _ in range(N):
        out.append(((model(x) - y) ** 2).mean())
    return out

def case_b():                      # .item() 으로 숫자만 꺼내 담는다
    out = []
    for _ in range(N):
        out.append(((model(x) - y) ** 2).mean().item())
    return out

def case_c():                      # no_grad 안에서 텐서를 그대로 담는다
    out = []
    with torch.no_grad():
        for _ in range(N):
            out.append(((model(x) - y) ** 2).mean())
    return out

for name, f in [("A  텐서 그대로", case_a),
                ("B  .item()", case_b),
                ("C  no_grad + 텐서", case_c)]:
    print(f"{name:22s} {measure(f):8.2f} MiB")
```

A가 B·C보다 최소 10배 이상 크게 나오면 됩니다. A에서 붙잡히는 것은 손실값 자체가 아니라 **그 손실을 만든 계산 기록이 붙들고 있는 중간 텐서**입니다(1주차 3일 기준값 ③과 같은 원리).

C가 A보다 훨씬 작다는 점도 함께 보세요. 텐서를 담는 것 자체가 문제가 아니라 **계산 기록이 딸려 오느냐**가 문제입니다. 그래서 검증 루프는 `no_grad()` 안에서 돌리고, 손실은 `.item()`으로 기록합니다. 3주차 1일에 실제로 그렇게 짭니다.

> 참고로 매 스텝 `backward()`를 부르는 학습 루프에서는 backward가 중간값을 해제하므로 증가폭이 A보다 작습니다. 이 함정이 가장 크게 터지는 곳은 **backward를 부르지 않는 검증 루프**입니다.

---

## 3교시 (20분) · 실패 케이스

### 고치기 과제 1 (6분) · 손실이 제자리다

아래 코드를 그대로 실행하면 100스텝을 돌려도 손실이 처음 값에서 거의 움직이지 않습니다. **에러는 나지 않습니다.**

```python
import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.zeros(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    def forward(self, x):
        return x * self.w + self.b

torch.manual_seed(0)
model = Model()
x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = 3 * x + 2
lr = 0.1

for step in range(100):
    model.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad

print("loss =", loss.item(), " w =", model.w.item(), " b =", model.b.item())
```

1. 원인을 찾아 고치세요
2. **어느 출력을 보고 그렇게 판단했는지 한 줄로 적으세요**

### 고치기 과제 2 (6분) · 로드한 모델을 부르면 터진다

```python
model2 = LinearModel()
model2 = model2.load_state_dict(torch.load("d1_model.pt", weights_only=True))
print(model2(x))
```

```
TypeError: '_IncompatibleKeys' object is not callable
```

1. 원인을 찾아 고치세요
2. **에러 메시지의 어느 부분이 단서였는지 한 줄로 적으세요**

### 실패 케이스 표 (8분)

오늘 안 겪은 것도 언젠가 만납니다. 읽어 두세요.

| 증상 | 원인 | 해결 |
|---|---|---|
| 손실이 안 떨어지고 파라미터도 그대로. 에러는 없음 | 텐서를 그냥 속성에 대입해 `parameters()`에 등록되지 않음 | `nn.Parameter`로 감싼다. `len(list(model.parameters()))`로 확인 |
| 층을 여러 개 만들었는데 일부만 학습됨 | 파이썬 `list`에 담은 서브모듈은 등록되지 않음 | `nn.ModuleList` 또는 `nn.Sequential` |
| `'_IncompatibleKeys' object is not callable` | `load_state_dict`의 반환값을 모델 변수에 재대입 | in-place 함수다. 반환값을 쓰지 않는다 |
| `Missing key(s) in state_dict` / `Unexpected key(s)` | 저장 시점과 모델의 구조·속성 이름이 다름 | 양쪽 키를 출력해 대조. 의도적 부분 로드면 `strict=False` |
| `size mismatch for linear.weight` | 키 이름은 같은데 shape가 다름 | 층 크기 확인. 전이학습에서 클래스 수가 바뀔 때 흔하다 |
| GPU에서 저장한 파일이 CPU 환경에서 로드 실패 | 텐서에 device 정보가 박혀 있음 | `torch.load(path, map_location="cpu")` |
| `torch.load`에서 `UnpicklingError ... weights_only` | 모델을 통째로 저장한 파일이라 임의 객체 복원이 필요함 | `state_dict`만 저장하는 방식으로 바꾼다 |
| `best = model.state_dict()`로 보관했는데 마지막 값과 같음 | `state_dict()`는 값을 복사하지 않고 같은 텐서를 담는다 | `copy.deepcopy(model.state_dict())` 또는 파일로 저장 |
| `AttributeError: cannot assign module before Module.__init__() call` | `super().__init__()`을 빼먹음 | `__init__` 맨 위에 넣는다 |

---

## 마무리 (15분)

### 실전 대응표 — 4일차에 예고한 표의 현재 상태

| 1주차 4일 손으로 | 오늘 | 남은 것 |
|---|---|---|
| `w`, `b`를 각각 텐서로 | `nn.Linear(1, 1)` · `nn.Parameter` | — |
| `pred = x * w + b` | `pred = model(x)` | — |
| `loss = ((pred - y) ** 2).mean()` | 아직 손으로 | `nn.MSELoss` — **2주차 2일** |
| `w.grad.zero_(); b.grad.zero_()` | `model.zero_grad()` | `optimizer.zero_grad()` — **2주차 3일** |
| `loss.backward()` | 동일 | 동일 |
| `w -= lr * w.grad` (변수마다) | `for p in model.parameters()` 순회 | `optimizer.step()` — **2주차 3일** |
| 전체 데이터를 한 번에 | 동일 | `DataLoader` — **2주차 4일** |

### 오늘의 통과 기준

1. `nn.Parameter`로 감싼 모델은 `len(list(model.parameters()))`가 **2**, 일반 텐서로 대입한 모델은 **0**으로 찍힘 (Step 2)
2. `nn.Module` 버전 학습 루프를 `lr=0.1`로 100스텝 돌려 `w`가 2.9~3.1, `b`가 1.9~2.1 (Step 3)
3. `state_dict()`의 키가 `linear.weight` / `linear.bias`로 찍히고 shape가 각각 `(1, 1)` / `(1,)` (Step 4)
4. 저장 → 새 인스턴스에 로드 후, 같은 입력에 대한 두 모델 출력의 최대 차이가 **0** (Step 4)
5. Step 5에서 A(텐서 그대로)의 VRAM 증가량이 B(`.item()`)보다 **10배 이상** 크게 찍힘

### 확인 퀴즈 (6문항)

답을 적어서 채팅으로 보내세요. 정답과 해설은 채팅에서 공개합니다.

**1.**

```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.ones(1, requires_grad=True)
    def forward(self, x):
        return x * self.w

m = M()
loss = (m(torch.tensor([2.0])) - 10.0) ** 2
loss.backward()
print(len(list(m.parameters())), m.w.grad)
```

출력은?

1. `1 tensor([-32.])`
2. `0 tensor([-32.])`
3. `0 None`
4. `1 None`

**2.** `model.forward(x)`를 직접 부르지 않고 `model(x)`로 부르는 이유로 가장 정확한 것은?

1. `forward`는 내부 전용 메서드라 외부에서 부르면 에러가 난다
2. `model(x)`는 `forward` 앞뒤에 등록된 후크를 함께 실행하는데, 직접 부르면 그것이 건너뛰어진다
3. `model(x)`가 더 빠르다
4. `model.forward(x)`는 그래디언트를 계산하지 않는다

**3.**

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x):
        return self.block(x)

print(list(Net().state_dict().keys()))
```

출력은?

1. `['block.weight', 'block.bias']`
2. `['block.0.weight', 'block.0.bias', 'block.1.weight', 'block.1.bias']`
3. `['block.0.weight', 'block.0.bias', 'block.2.weight', 'block.2.bias']`
4. `['0.weight', '0.bias', '2.weight', '2.bias']`

**4.** `ret = model.load_state_dict(torch.load(path, weights_only=True))`에서 `ret`에 들어 있는 것은?

1. 가중치가 로드된 새 모델
2. `None`
3. 파일에서 읽어 온 `state_dict` 딕셔너리
4. 누락·초과 키 목록을 담은 객체. 모델 자체는 제자리에서 바뀐다

**5.** 학습 중 최고 성능 시점의 가중치를 메모리에 보관하려고 `best = model.state_dict()`를 실행했다. 이후 100스텝을 더 돌린 뒤 `best`의 값을 확인하면?

1. 저장 시점의 값 그대로다
2. 100스텝 뒤의 최신 값으로 바뀌어 있다
3. `RuntimeError`가 난다
4. 키만 남고 값이 비어 있다

**6.**

```python
losses_a, losses_b = [], []
for _ in range(500):
    loss = ((model(x) - y) ** 2).mean()
    losses_a.append(loss)
    losses_b.append(loss.item())
```

이 코드에 대한 설명으로 옳은 것은?

1. `losses_a`는 텐서를 리스트에 담으므로 `TypeError`가 난다
2. 둘은 같은 양의 메모리를 쓴다. `.item()`은 출력 형식을 위한 것이다
3. `losses_a`는 각 손실을 만든 계산 기록을 500개 붙잡아 두지만, `losses_b`는 파이썬 float 500개만 담는다
4. `losses_b`는 float32가 파이썬 float으로 바뀌며 정밀도가 떨어진다

---

## 다음 세션 예고

**2주차 2일 · 손실 함수** — MSE와 CrossEntropy가 왜 서로 다른 형태인지, `CrossEntropyLoss`가 softmax를 안에 품고 있는 이유를 다룹니다. **오늘보다 수식이 많습니다.** 식과 코드를 한 줄씩 대응시켜 진행하니, 수식 자체보다 "어느 항이 어느 코드인가"에 집중하면 됩니다.
