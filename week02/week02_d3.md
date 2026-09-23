# 2주차 3일 · 옵티마이저

> 배정 시간 **2시간** · 노트북 `week02/d3_optimizer.ipynb`

오늘은 1주차 4일에 만든 실전 대응표의 남은 두 줄, `optimizer.zero_grad()`와 `optimizer.step()`을 채웁니다. 손으로 쓰던 `p -= lr * p.grad`를 옵티마이저 객체로 바꾸고, 그 한 줄이 모멘텀·Adam으로 바뀌면 무엇이 달라지는지를 숫자로 확인합니다. 결론은 하나로 모입니다 — **lr이라는 숫자의 뜻이 옵티마이저마다 다릅니다.**

직전 세션의 미해결 항목은 없습니다. 2주차 2일처럼 각 Step 끝에 **⟶ 한 줄로 적기**가 있고, 그 시간은 Step 시간에 포함돼 있습니다.

| 교시 | 시간 | 내용 |
|---|---|---|
| 도입 | 5분 | 누적 복습 3문항 |
| 목표 | 2분 | 오늘 얻는 것 |
| 1교시 | 25분 | 개념 A 옵티마이저 · B 모멘텀 · C Adam · D AdamW |
| 2교시 | 53분 | 실습 Step 1~5 |
| 3교시 | 20분 | 실패 케이스 — 고치기 과제 2개 + 표 |
| 마무리 | 15분 | 통과 기준 · 실전 대응표 · 확인 퀴즈 |

---

## 도입 (5분) · 누적 복습

문항당 1분. 답은 채팅으로 보내 주세요. 채점과 보강은 채팅에서 합니다.

**복습 1.** 출력되는 `z.grad`와, 이어서 `z -= 0.1 * z.grad`로 한 번 갱신했을 때 정답 칸 로짓 `z[0, 2]`의 변화로 맞는 것은?

```python
z = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)   # 세 클래스의 로짓이 같다
t = torch.tensor([2])                                      # 정답은 2번 클래스
nn.CrossEntropyLoss()(z, t).backward()
print(z.grad)
```

| | `z.grad` | `z[0, 2]` |
|---|---|---|
| A | `[[ 0.3333,  0.3333, -0.6667]]` | 올라간다 |
| B | `[[-0.3333, -0.3333,  0.6667]]` | 올라간다 |
| C | `[[ 0.3333,  0.3333, -0.6667]]` | 내려간다 |
| D | `[[-0.3333, -0.3333,  0.6667]]` | 내려간다 |

**복습 2.** 출력은?

```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.tensor([1.0], requires_grad=True)   # 일반 텐서
        self.b = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.w * x + self.b

m = M()
m(torch.tensor([2.0])).sum().backward()       # 손실 = 2w + b
with torch.no_grad():
    for p in m.parameters():
        p -= 0.5 * p.grad
print(m.w.grad, m.w.item(), m.b.item())
```

| | 출력 |
|---|---|
| A | `None 1.0 0.5` |
| B | `tensor([2.]) 1.0 0.5` |
| C | `tensor([2.]) 0.0 0.5` |
| D | `None 0.0 0.5` |

**복습 3.** 출력은?

```python
src = nn.Linear(1, 1)
dst = nn.Linear(1, 1)
out = dst.load_state_dict(src.state_dict())
print(type(out).__name__, torch.equal(dst.weight, src.weight))   # torch.equal: shape·값이 모두 같으면 True
```

| | 출력 |
|---|---|
| A | `Linear True` |
| B | `NoneType True` |
| C | `_IncompatibleKeys True` |
| D | `_IncompatibleKeys False` |

---

## 오늘 얻는 것 (2분)

| 배우는 것 | 이걸로 할 수 있게 되는 일 | 중요도 |
|---|---|---|
| `torch.optim.SGD`와 표준 루프 — `zero_grad()` → 순전파·손실 → `backward()` → `step()` | 손으로 쓰던 갱신을 표준 네 줄로 바꿔 쓰고, 남의 학습 코드에서 이 네 줄의 순서가 틀린 곳을 짚는다 | ★★★ 필수 |
| 옵티마이저는 생성할 때 받은 명단만 본다 | 에러 없이 학습이 안 될 때 `len(list(model.parameters()))`와 `is` 비교로 "명단 밖의 텐서"와 "옛 모델을 붙잡은 옵티마이저"를 가려낸다 | ★★★ 필수 |
| Adam — 그래디언트를 자기 크기로 나눈다 | Adam의 첫 스텝이 그래디언트 크기와 무관하게 lr만큼 움직인다는 것을 식으로 보이고, 입력 스케일이 바뀌어도 Adam이 버티는 이유를 설명한다 | ★★★ 필수 |
| lr의 뜻은 옵티마이저마다 다르다 | SGD의 lr(그래디언트에 곱하는 배율)과 Adam의 lr(한 스텝의 이동 폭)을 구분해, 옵티마이저를 바꿀 때 lr을 다시 잡는다 | ★★★ 필수 |
| 모멘텀 | `momentum=0.9` 설정을 보고 "같은 방향이면 최대 10배, 번갈아 나오면 약 절반"을 식으로 설명한다 | ★★ 권장 · **9주차 4일** 옵티마이저·스케줄러 선택에서 다시 |
| AdamW — weight decay를 나눗셈 밖으로 | Adam 계열을 쓸 때 AdamW를 고르는 이유를 한 문장으로 말한다 | ★★ 권장 · **3주차 3일** weight decay, **9주차 4일** 분리 효과 비교 |
| 옵티마이저 상태의 메모리 | 학습 VRAM을 "파라미터 + 그래디언트 + 옵티마이저 상태 + 활성값"으로 나눠 손계산한다 | ★★ 권장 · **7주차 이후** 배치 크기 산정, **16주차 1일** 8-bit optimizer |
| `zero_grad()` 뒤 `.grad`는 `None` | `.grad`를 찍거나 가공하는 코드를 `backward()` 뒤에 둔다 | ★ 참고 · **5주차 4일** gradient clipping에서 다시 |

**시간이 부족하면 ★부터 버린다.** 오늘 ★은 개념 A의 각주 하나라, 그다음은 ★★인 Step 3의 M(AdamW) → Step 4 순서로 줄입니다.

---

## 1교시 (25분) · 개념

### 개념 A · 옵티마이저 (6분)

**무엇인가.** 파라미터 명단을 받아 두었다가, `step()` 한 번에 명단 전체를 정해진 규칙으로 고쳐 쓰는 객체입니다.

**왜 필요한가.** 2주차 1일까지는 갱신을 직접 썼습니다.

```python
with torch.no_grad():
    for p in model.parameters():
        p -= lr * p.grad
```

규칙이 바뀔 때마다(개념 B·C) 이 루프를 다시 짜야 하고, 규칙이 지난 스텝의 값을 기억해야 하면 파라미터마다 저장소를 따로 관리해야 합니다. 옵티마이저는 **규칙과 기억을 한 객체에 담습니다.**

오늘의 모든 식은 아래 한 줄에서 출발합니다. 1주차 4일부터 손으로 써 온 그 식입니다.

$$\theta \leftarrow \theta - \eta\, g \quad \cdots (1)$$

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| $\theta$ | 세타 | 파라미터 하나 | `p` |
| $\eta$ | 에타 | 학습률 | `lr` |
| $g$ | 지 | 그 파라미터의 그래디언트 | `p.grad` |
| $\leftarrow$ | — | 오른쪽 값으로 바꿔 넣는다 | `-=` (제자리 수정) |

`torch.optim.SGD`는 식 (1)을 명단 전체에 적용하는 옵티마이저입니다.

> SGD(확률적 경사하강)의 "확률적"은 데이터를 무작위 배치로 쪼개 넣는 데서 온 이름입니다. 배치는 2주차 4일에 다룹니다. 오늘은 데이터 전체를 한 번에 넣으므로 그냥 경사하강과 같습니다.

**언제 쓰는가.** 모든 학습 루프입니다. 생성은 루프 밖에서 한 번, `zero_grad()`와 `step()`은 매 스텝.

**최소 예시.**

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1)   # 명단을 넘기며 생성 — 루프 밖에서 한 번
for step in range(100):
    opt.zero_grad()                  # 명단 전체의 .grad 지우기
    loss = loss_fn(model(x), y)
    loss.backward()                  # .grad 채우기 — 옵티마이저는 관여하지 않는다
    opt.step()                       # 명단 전체에 식 (1)
```

`zero_grad()`가 필요한 이유는 1주차 3일과 같습니다. `.grad`는 덮어쓰지 않고 **더해지므로** 매 스텝 지워야 합니다. 위치 규칙은 하나입니다 — `backward()`보다 앞, 직전 `step()`보다 뒤. 이 커리큘럼은 **루프 첫 줄**에 둡니다.

**명단이 전부입니다.** 2주차 2일에 확정한 한 줄이 그대로 옵티마이저의 동작 규칙입니다.

> **`requires_grad`는 "계산할지", `nn.Parameter`는 "명단에 넣을지". `backward()`는 전자만 보고, `optimizer`와 `zero_grad()`는 후자만 본다.**

여기서 두 가지가 따라 나옵니다. 둘 다 3교시에서 직접 고칩니다.

- 명단 밖의 텐서는 `.grad`가 계산되는데도 **갱신도 초기화도 되지 않습니다**
- 옵티마이저는 값을 복사해 가지 않고 **파라미터 객체 자체**를 가리킵니다. 모델을 새로 만들면 옵티마이저는 옛 모델을 계속 가리킵니다

> **각주 두 개**
>
> - `zero_grad()`는 `.grad`를 0이 아니라 **`None`**으로 만듭니다(기본값 `set_to_none=True`). 2주차 2일에 `m.zero_grad()` 뒤 `b.grad`가 `None`이 된 것이 이것입니다 — ★ 참고
> - 밑줄 규칙(이름 끝에 `_`가 붙은 것만 제자리 수정, 2주차 2일)은 **텐서 연산**의 규칙입니다. `zero_grad()`·`step()`은 옵티마이저 객체의 명령이라 밑줄이 없어도 명단 속 텐서를 직접 바꾸고, 반환값은 `None`입니다. 받아 둘 값이 없습니다

**실전 대응.**

| 2주차 1일 코드 | 실전 | 나오는 곳 |
|---|---|---|
| `model.zero_grad()` | `opt.zero_grad()` | 모든 학습 루프의 첫 줄 |
| `for p in model.parameters(): p -= lr * p.grad` | `opt.step()` | 모든 학습 루프의 마지막 줄 |

### 개념 B · 모멘텀 (6분)

**무엇인가.** 지금까지 움직여 온 방향을 기억해 두고, 이번 그래디언트를 거기에 더한 방향으로 움직이는 방식입니다. 비탈을 굴러 내려가는 공을 떠올리면 됩니다. 같은 방향으로 계속 기울어 있으면 속도가 붙고, 좌우로 번갈아 밀리면 그 흔들림은 쌓이지 않습니다.

**왜 필요한가.** 기준값 ④에서 SGD의 lr 상한이 $x$의 스케일, 즉 `w` 방향이 얼마나 가파른지로 정해지는 것을 봤습니다. 파라미터가 여럿이면 그중 **가장 가파른 방향**이 상한을 정합니다. 한쪽은 가파르고 한쪽은 완만한 지형에서 lr을 가파른 쪽에 맞춰 낮추면, 완만한 쪽으로는 한 스텝에 조금밖에 못 갑니다. 그런데 완만한 쪽 그래디언트는 매 스텝 **같은 부호**로 나오므로, 그걸 쌓아 두면 속도를 낼 수 있습니다. Step 5에서 이 지형을 직접 만듭니다.

$$m \leftarrow \mu\, m + g \quad \cdots (2)$$

$$\theta \leftarrow \theta - \eta\, m \quad \cdots (3)$$

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| $m$ | 엠 | 지금까지의 이동 방향(버퍼). 처음엔 0 | `opt.state[p]["momentum_buffer"]` |
| $\mu$ | 뮤 | 지난 방향을 얼마나 남길지. 보통 0.9 | `momentum` |
| $g,\ \eta,\ \theta$ | | 식 (1)과 같다 | `p.grad`, `lr`, `p` |

```python
m = mu * m + g        # (2)
p -= lr * m           # (3)
```

| 식의 부분 | 코드 | 뜻 |
|---|---|---|
| $\mu\, m$ | `mu * m` | 지난 방향의 90%를 남긴다 |
| $+\, g$ | `+ g` | 이번 그래디언트를 **그대로** 더한다 (평균이 아니라 누적) |
| $\eta\, m$ | `lr * m` | 식 (1)의 $g$ 자리에 $m$이 들어갔을 뿐 |

$\mu = 0$이면 $m = g$가 되어 식 (1), 즉 그냥 SGD입니다.

**손계산** — 매 스텝 $g = 1$, $\mu = 0.9$, $\eta = 0.1$

| $t$ | $m$ (식 2) | 이동 $\eta\, m$ |
|---|---|---|
| 1 | $0.9 \times 0 + 1 = 1$ | 0.1 |
| 2 | $0.9 \times 1 + 1 = 1.9$ | 0.19 |
| 3 | ? | ? ← Step 2에서 예측 |

같은 방향이 끝없이 이어지면 $m$은 더 이상 변하지 않는 값 $m^*$에 다가갑니다. 식 (2)의 양변이 같아지는 지점을 풀면

$$m^* = \mu\, m^* + g \;\Rightarrow\; m^*(1-\mu) = g \;\Rightarrow\; m^* = \frac{g}{1-\mu} = 10\,g \quad \cdots (4)$$

**같은 방향이면 최대 10배까지 빨라집니다.** 부호가 번갈아 나올 때는 반대로 줄어드는데, 그건 Step 2에서 숫자로 봅니다.

**언제 쓰는가.** `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)` — CNN을 처음부터 학습하는 고전적인 설정입니다(ResNet 원 논문이 이 조합). 개념 C의 Adam 안에도 같은 장치가 들어 있습니다.

**실전 대응.**

| 오늘 | 실전 | 나오는 곳 |
|---|---|---|
| $\mu = 0.9$ | `momentum=0.9` | SGD를 쓰는 CNN 학습 설정 |
| $m$ | `opt.state[p]["momentum_buffer"]` | 체크포인트에 옵티마이저 상태로 함께 저장된다 (7주차 4일) |

### 개념 C · Adam (8분)

**무엇인가.** 파라미터마다 **그래디언트의 평균 방향**과 **그래디언트의 평균 크기**를 따로 기록해 두고, 방향을 크기로 나눈 값으로 움직이는 옵티마이저입니다.

**왜 필요한가.** 파라미터마다 그래디언트 크기가 수백 배씩 다를 수 있습니다(Step 5에서 실제로 만듭니다). SGD는 lr 하나를 모두에게 곱하므로, 그래디언트가 가장 큰 파라미터에 맞춰 lr을 낮춰야 하고 나머지는 거의 못 움직입니다. Adam은 각 파라미터의 그래디언트를 **그 파라미터 자신의 평균 크기**로 나눕니다.

**식 — 네 줄이지만 하는 일은 "평균 방향 ÷ 평균 크기" 하나입니다.**

$$m \leftarrow \beta_1\, m + (1-\beta_1)\, g \quad \cdots (5)$$

$$v \leftarrow \beta_2\, v + (1-\beta_2)\, g^2 \quad \cdots (6)$$

$$\hat{m} = \frac{m}{1-\beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1-\beta_2^{\,t}} \quad \cdots (7)$$

$$\theta \leftarrow \theta - \eta\, \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} \quad \cdots (8)$$

- (5)는 개념 B의 (2)와 모양이 같습니다. 다른 점은 $g$에 $(1-\beta_1)$을 곱해 **누적이 아니라 평균**으로 만든 것 하나입니다
- (6)은 같은 방식으로 $g^2$의 평균을 냅니다. 제곱이라 부호가 사라지고 크기만 남습니다. $\sqrt{v}$가 "최근 그래디언트의 평균 크기"입니다
- (7)은 보정입니다. $m$과 $v$는 0에서 출발하므로 처음 몇 스텝은 실제보다 작게 나옵니다. 그만큼 키워 줍니다. $t$가 커지면 분모가 1에 가까워져 보정이 사라집니다
- (8)은 방향 $\hat{m}$을 크기 $\sqrt{\hat{v}}$로 나눈 값에 lr을 곱해 움직입니다

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| $m$ | 엠 | 그래디언트의 평균 방향 | `opt.state[p]["exp_avg"]` |
| $v$ | 브이 | 그래디언트 제곱의 평균 | `opt.state[p]["exp_avg_sq"]` |
| $\beta_1$ | 베타 원 | $m$이 과거를 남기는 비율, 0.9 | `betas[0]` |
| $\beta_2$ | 베타 투 | $v$가 과거를 남기는 비율, 0.999 | `betas[1]` |
| $t$ | 티 | 지금이 몇 번째 `step()`인가 | `opt.state[p]["step"]` |
| $\hat{m},\ \hat{v}$ | 엠 햇, 브이 햇 | 보정한 $m$, $v$ | `m_hat`, `v_hat` |
| $\epsilon$ | 엡실론 | 0으로 나누기 방지, $10^{-8}$ | `eps` |

```python
m = beta1 * m + (1 - beta1) * g            # (5)
v = beta2 * v + (1 - beta2) * g**2         # (6)
m_hat = m / (1 - beta1**t)                 # (7)
v_hat = v / (1 - beta2**t)                 # (7)
p -= lr * m_hat / (v_hat.sqrt() + eps)     # (8)
```

| 식의 부분 | 코드 | 뜻 |
|---|---|---|
| $(1-\beta_1)\, g$ | `(1 - beta1) * g` | 이번 그래디언트는 10%만 반영 |
| $g^2$ | `g**2` | 부호를 지우고 크기만 |
| $\sqrt{\hat{v}}$ | `v_hat.sqrt()` | 최근 그래디언트의 평균 크기 |
| $\hat{m} / \sqrt{\hat{v}}$ | `m_hat / v_hat.sqrt()` | 방향 ÷ 크기 |

손계산은 Step 3에서 `opt.state`에 실제로 들어 있는 값으로 합니다.

**언제 쓰는가.** 새 문제를 빠르게 돌려볼 때의 기본값이고, Transformer 계열(13주차 ViT, 18주차 이후 LLM)은 사실상 전부 Adam 계열입니다. PyTorch 기본값은 `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`입니다.

**실전 대응.**

| 오늘 | 실전 | 나오는 곳 |
|---|---|---|
| 식 (5)~(8) | `torch.optim.Adam(model.parameters(), lr=1e-3)` | Adam을 쓰는 모든 코드 |
| $m$, $v$ | 파라미터마다 `exp_avg`, `exp_avg_sq` 두 개 | 메모리를 차지한다 — Step 4에서 잰다 |

### 개념 D · AdamW (5분)

**무엇인가.** Adam에 weight decay를 식 (8)의 나눗셈 **바깥에서** 따로 적용하는 옵티마이저입니다. **weight decay**는 매 스텝 파라미터를 0 쪽으로 일정 비율씩 줄이는 것입니다.

**왜 필요한가.** weight decay를 옛 방식, 즉 그래디언트에 $\lambda\theta$를 더하는 방식(식 9)으로 Adam에 넣으면 그 $\lambda\theta$도 식 (8)의 나눗셈을 통과합니다. 당기는 힘까지 그 파라미터의 그래디언트 크기로 나눠지므로, 의도한 $\lambda$대로 당겨지지 않습니다. 얼마나 틀어지는지는 Step 3에서 숫자로 봅니다. AdamW는 나눗셈과 따로 $\theta$에 $(1-\eta\lambda)$를 곱해(식 10) 모든 파라미터를 같은 비율로 당깁니다.

`Adam(weight_decay=λ)` — 옛 방식. (9)를 한 뒤 (5)~(8)

$$g \leftarrow g + \lambda\,\theta \quad \cdots (9)$$

`AdamW(weight_decay=λ)` — 분리 방식. (10)을 한 뒤 (5)~(8)

$$\theta \leftarrow \theta\,(1-\eta\lambda) \quad \cdots (10)$$

| 기호 | 읽는 법 | 뜻 | 코드 |
|---|---|---|---|
| $\lambda$ | 람다 | weight decay 계수. AdamW 기본값 0.01 | `weight_decay` |

| 옵티마이저 | 당기는 코드 | 위치 |
|---|---|---|
| `Adam(weight_decay=lam)` | `g = g + lam * p` … (9) | 나눗셈 **안** — (8)에서 $\sqrt{\hat{v}}$로 나눠진다 |
| `AdamW(weight_decay=lam)` | `p *= (1 - lr * lam)` … (10) | 나눗셈 **밖** — 늘 $\eta\lambda$ 비율로 줄어든다 |

**언제 쓰는가.** Adam 계열을 쓰는 실전 코드는 거의 AdamW입니다. 오늘은 **(9)와 (10)의 위치 차이** 하나만 잡습니다. weight decay를 왜 거는지는 3주차 3일, 분리 효과를 실제 학습으로 비교하는 건 9주차 4일입니다.

**실전 대응.**

| 오늘 | 실전 | 나오는 곳 |
|---|---|---|
| 식 (10) + (5)~(8) | `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)` | 13주차 이후 Transformer 파인튜닝 |

---

## 2교시 (53분) · 실습

노트북 `week02/d3_optimizer.ipynb` 하나로 진행합니다. Step 1~3·5는 CPU, Step 4만 GPU입니다. 텐서가 작아 GPU 전송이 계산보다 비싼 구간입니다(기준값 ②). 커널을 재시작했다면 셀 1-1부터 다시 실행합니다.

| Step | 시간 | 내용 | 방식 |
|---|---|---|---|
| 1 | 8분 | `optim.SGD`로 바꾸고 손 갱신과 대조 | 완성 코드 |
| 2 | 11분 | 모멘텀 버퍼 손계산 대조 | PRIMM |
| 3 | 12분 | Adam 첫 스텝의 이동 거리 | PRIMM |
| 4 | 7분 | 옵티마이저 상태의 VRAM | 완성 코드 · GPU |
| 5 | 15분 | 루프 배열 + 입력 ×10에서 옵티마이저 비교 | 종합 · 빈칸 |

### Step 1 · `optim.SGD`로 갱신하고 손 갱신 결과와 대조하기 (8분)

> **볼 것** — 셀 1-5에서 두 방식의 차이가 `0.0`으로 찍히는 것, 셀 1-6에서 옵티마이저 안의 명단이 `model_b`의 파라미터 **그 자체**라는 것
>
> **끝나면** — 손으로 쓰던 갱신 루프를 `zero_grad()`·`step()` 두 줄로 바꿔 쓸 수 있다
>
> **쓰는 상황** — 앞으로의 모든 학습 루프. 셀 1-4가 3주차 이후 모든 루프의 뼈대다

| 새로 나온 것 | 하는 일 |
|---|---|
| `torch.optim.SGD(params, lr=)` | 명단 `params`를 받아 두고 식 (1)로 갱신하는 옵티마이저를 만든다 |
| `opt.zero_grad()` | 명단 전체의 `.grad`를 지운다 |
| `opt.step()` | 명단 전체를 한 번 갱신한다 |
| `opt.param_groups` | 옵티마이저가 들고 있는 명단과 설정(lr 등). dict를 담은 리스트 |

시간 배분 — 셀 1-1~1-5 실행 5분 · 셀 1-6 조사 2분 · 한 줄로 적기 1분

**셀 1-1 — 데이터** (2주차 2일과 같은 식. 기준값 ⑦의 교훈대로 `y`도 `(100, 1)`)

```python
import torch
import torch.nn as nn

x = torch.linspace(-3, 3, 100).unsqueeze(1)   # (100, 1)
y = 3 * x + 2                                  # (100, 1)
loss_fn = nn.MSELoss()
print(x.shape, y.shape)
```

**셀 1-2 — 시작점이 같은 모델 두 개**

```python
torch.manual_seed(0)
model_a = nn.Linear(1, 1)          # 손으로 갱신할 모델
model_b = nn.Linear(1, 1)          # 옵티마이저로 갱신할 모델
r = model_b.load_state_dict(model_a.state_dict())   # a의 값을 b에 복사
print(r)
print(model_a.weight.item(), model_b.weight.item())
```

**셀 1-3 — 손 갱신** (2주차 1일 루프 그대로)

```python
lr = 0.1
for step in range(100):
    model_a.zero_grad()
    loss = loss_fn(model_a(x), y)
    loss.backward()
    with torch.no_grad():
        for p in model_a.parameters():
            p -= lr * p.grad
print(f"손 갱신   : w={model_a.weight.item():.6f}  b={model_a.bias.item():.6f}")
```

**셀 1-4 — 옵티마이저 갱신**

```python
opt = torch.optim.SGD(model_b.parameters(), lr=0.1)   # 루프 밖에서 한 번
for step in range(100):
    opt.zero_grad()
    loss = loss_fn(model_b(x), y)
    loss.backward()
    opt.step()
print(f"옵티마이저: w={model_b.weight.item():.6f}  b={model_b.bias.item():.6f}")
```

**셀 1-5 — 대조**

```python
for (name, pa), pb in zip(model_a.named_parameters(), model_b.parameters()):   # (이름, 파라미터) 쌍
    print(name, (pa - pb).abs().max().item())
```

**셀 1-6 — 옵티마이저 안 들여다보기**

```python
group = opt.param_groups[0]
print(len(group["params"]), group["lr"])
print(group["params"][0] is model_b.weight, group["params"][1] is model_b.bias)
print(group["params"][0] is model_a.weight)
```

```python
print(model_b.weight.grad)     # 마지막 backward()가 남긴 값 — 0에 가까운 아주 작은 수
opt.zero_grad()
print(model_b.weight.grad)
```

`is`는 값이 같은지가 아니라 **같은 객체인지**를 묻습니다. `model_a`와 `model_b`는 지금 값이 똑같으므로, 값 비교로는 둘을 구분할 수 없습니다.

**⟶ 한 줄로 적기 (1분)** — 셀 1-6 첫 칸의 `True True`와 `False`가 옵티마이저에 대해 뜻하는 것.

### Step 2 · 모멘텀 버퍼를 손계산하고 `step()` 결과와 대조하기 (11분) — PRIMM

> **볼 것** — 세 번째 `step()`의 이동 거리가 개념 B의 손계산 표를 한 줄 더 이어 계산한 값과 같은지
>
> **끝나면** — `momentum=0.9`가 들어간 설정에서 이동 폭이 어떻게 변하는지 식 (2)로 계산할 수 있다

| 새로 나온 것 | 하는 일 |
|---|---|
| `SGD([p], lr=, momentum=0.9)` | 명단을 리스트로 직접 준다. 옵티마이저가 보는 건 텐서의 종류가 아니라 **이 명단**이다 |
| `opt.state[p]` | 옵티마이저가 파라미터 `p`마다 따로 기억하는 값(dict) |

> 아래 `p`는 `nn.Parameter`가 아닌 일반 텐서인데도 갱신됩니다. 명단에 **직접** 넣었기 때문입니다. `nn.Parameter`는 텐서를 `model.parameters()`라는 명단에 올려 주는 장치이고, 옵티마이저는 받은 명단만 봅니다. 개념 A의 한 줄과 같은 말입니다.

**P · 예측 (2분)** — 실행하지 말고, 마크다운 셀에 답을 적습니다. 세 번째 줄(`t=3`)에 찍힐 이동 거리는?

```python
p = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)

for t in range(1, 4):
    before = p.item()
    opt.zero_grad()
    p.sum().backward()             # 손실 = p 그대로 → p.grad는 매번 1
    opt.step()
    print(t, f"이동 {before - p.item():.4f}")
```

| | `t=3`의 이동 |
|---|---|
| A | 0.1000 |
| B | 0.1900 |
| C | 0.2710 |
| D | 0.3000 |

**R · 실행 (1분)** — 실행해서 예측과 대조합니다.

**I · 조사 (3분)** — 옵티마이저가 기억하는 값을 꺼내고, 같은 방향으로 97번 더 밀어 봅니다.

```python
print(opt.state[p])
```

```python
for t in range(4, 101):
    before = p.item()
    opt.zero_grad()
    p.sum().backward()
    opt.step()
print(f"100번째 이동 {before - p.item():.3f}   버퍼 {opt.state[p]['momentum_buffer'].item():.4f}")
```

<details>
<summary>두 셀을 실행한 뒤에 펼치기</summary>

- `momentum_buffer`가 식 (2)의 $m$입니다. $1 \to 1.9 \to 2.71$로 커졌고, 세 번째 이동은 $\eta\, m = 0.1 \times 2.71 = 0.271$입니다
- 100번째에는 버퍼가 9.9997, 이동이 1.000입니다. 식 (4)의 $m^* = g/(1-\mu) = 10$에 다가간 것이고, 같은 lr의 SGD(0.1)보다 **10배** 빠릅니다
- 보기 B(0.19)는 두 번째 스텝의 값, D(0.3)는 $\mu$ 없이 더하기만 한 값($m = 1, 2, 3$)입니다

</details>

**M · 수정 (3분)**

① P 셀의 `momentum=0.9`를 `momentum=0.0`으로 바꿔 다시 실행하고, `print(opt.state[p])`도 찍습니다.

② 그래디언트의 부호를 매 스텝 번갈아 바꿉니다.

```python
p = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
for t in range(1, 101):
    sign = 1.0 if t % 2 == 1 else -1.0      # +1, -1, +1, ...
    opt.zero_grad()
    (sign * p).sum().backward()             # p.grad = sign
    opt.step()
    if t >= 97:
        print(t, f"버퍼 {opt.state[p]['momentum_buffer'].item():+.4f}")
```

<details>
<summary>실행한 뒤에 펼치기</summary>

- ① 이동이 0.1000으로 일정하고 `opt.state[p]`는 `{}`입니다. $\mu = 0$이면 식 (1)과 같고, **기억할 것이 없으니 상태도 만들지 않습니다.** Step 4에서 메모리 0으로 다시 나옵니다
- ② 버퍼가 $+0.5263$과 $-0.5263$을 오갑니다. 부호가 번갈아 나오면 $m$은 $+a$와 $-a$를 오가고, 이를 식 (2)에 넣으면 $-a = \mu a - 1$이므로

$$a = \frac{1}{1+\mu} = \frac{1}{1.9} \approx 0.53$$

- SGD라면 버퍼 자리에 그래디언트 $\pm 1$이 그대로 들어갑니다. **흔들림이 절반으로 눌렸습니다**
- 같은 방향이면 10배, 번갈아 나오면 0.53배 — 이 비대칭이 모멘텀이 하는 일의 전부입니다

</details>

**실전 대응 (1분)**

| 오늘 | 실전 | 나오는 곳 |
|---|---|---|
| `SGD([p], lr=0.1, momentum=0.9)` | `SGD(model.parameters(), lr=0.1, momentum=0.9)` | CNN을 처음부터 학습하는 설정 |
| `opt.state[p]["momentum_buffer"]` | `opt.state_dict()` 안의 `state` | 학습 재개용 체크포인트 (7주차 4일) |

**⟶ 한 줄로 적기 (1분)** — `버퍼 9.9997`과 `버퍼 ±0.5263`, 두 출력을 함께 놓고 모멘텀이 하는 일을.

### Step 3 · Adam 첫 스텝의 이동 거리를 예측하고 확인하기 (12분) — PRIMM

> **볼 것** — 그래디언트가 백만 배 차이 나는 두 파라미터가 Adam의 첫 `step()`에서 각각 얼마나 움직이는지
>
> **끝나면** — Adam의 lr이 SGD의 lr과 다른 뜻이라는 것을 식 (7)·(8)로 보일 수 있다

| 새로 나온 것 | 하는 일 |
|---|---|
| `torch.optim.Adam(params, lr=)` | 식 (5)~(8)로 갱신한다 |
| `torch.optim.AdamW(params, lr=, weight_decay=)` | 식 (10)을 한 뒤 (5)~(8)로 갱신한다 |

**P · 예측 (2분)** — 첫 `step()` 뒤 두 파라미터가 움직인 거리는?

```python
big   = torch.tensor([5.0], requires_grad=True)
small = torch.tensor([5.0], requires_grad=True)
opt = torch.optim.Adam([big, small], lr=0.1)

loss = 1000 * big.sum() + 0.001 * small.sum()   # big.grad = 1000, small.grad = 0.001
loss.backward()
opt.step()
print(f"big {5 - big.item():.4f}   small {5 - small.item():.4f}")
```

| | `big` | `small` |
|---|---|---|
| A | 100.0000 | 0.0001 |
| B | 0.1000 | 0.1000 |
| C | 0.1000 | 0.0001 |
| D | 0.3162 | 0.3162 |

**R · 실행 (1분)**

**I · 조사 (4분)** — 옵티마이저가 기록한 값을 꺼내 식 (5)~(8)을 손으로 따라갑니다.

```python
for name, q in (("big", big), ("small", small)):
    s = opt.state[q]
    print(f"{name:5s} t={s['step'].item():.0f}  m={s['exp_avg'].item():.4g}  v={s['exp_avg_sq'].item():.4g}")
```

```python
s = opt.state[big]
m_hat = s["exp_avg"] / (1 - 0.9 ** s["step"])          # (7)
v_hat = s["exp_avg_sq"] / (1 - 0.999 ** s["step"])     # (7)
print(f"{(0.1 * m_hat / (v_hat.sqrt() + 1e-8)).item():.4f}")   # (8)의 이동 거리
```

<details>
<summary>실행한 뒤에 펼치기 — 손계산</summary>

`big`($g = 1000$)의 첫 스텝($t = 1$)을 식 순서대로 따라갑니다.

$$m = 0.1 \times 1000 = 100, \qquad v = 0.001 \times 1000^2 = 1000 \qquad \cdots (5),(6)$$

$$\hat{m} = \frac{100}{1-0.9} = 1000, \qquad \hat{v} = \frac{1000}{1-0.999} = 10^6 \qquad \cdots (7)$$

$$0.1 \times \frac{1000}{\sqrt{10^6}} = 0.1 \times \frac{1000}{1000} = 0.1 \qquad \cdots (8)$$

그래디언트를 숫자 대신 $g$로 두고 다시 하면 이유가 보입니다. 첫 스텝에서는 $\hat{m} = g$, $\hat{v} = g^2$이 되므로

$$\eta\, \frac{\hat{m}}{\sqrt{\hat{v}}} = \eta\, \frac{g}{|g|} = \pm\,\eta$$

**첫 스텝은 그래디언트의 크기와 무관하게 정확히 lr만큼, 그래디언트의 반대 방향으로 움직입니다.** `small`도 $0.1 \times \frac{0.001}{0.001 + 10^{-8}} \approx 0.1$입니다. $\epsilon$은 그래디언트가 $10^{-8}$ 근처일 때만 차이를 만듭니다.

- 보기 A는 SGD의 답입니다. $0.1 \times 1000 = 100$, $0.1 \times 0.001 = 0.0001$. 백만 배 차이가 Adam에서는 1배가 됐습니다
- 보기 D는 식 (7)의 보정을 빼먹은 값입니다. $0.1 \times 100 / \sqrt{1000} \approx 0.316$
- **SGD의 lr은 그래디언트에 곱하는 배율이고, Adam의 lr은 한 스텝에 움직이는 거리의 기준입니다.** 오늘 가장 중요한 한 줄입니다

</details>

**M · 수정 (3분)** — 그래디언트를 0으로 두고 weight decay만 켭니다. 움직임은 전부 weight decay에서 나옵니다. 실행 전에 **AdamW의 결과를 식 (10)으로 먼저 계산해** 적습니다.

```python
for Opt in (torch.optim.Adam, torch.optim.AdamW):
    q = torch.tensor([1.0], requires_grad=True)
    opt = Opt([q], lr=0.1, weight_decay=0.01)
    (0 * q).sum().backward()          # 손실이 q와 무관 → q.grad는 None이 아니라 0
    opt.step()
    print(f"{Opt.__name__:6s} q = {q.item():.4f}")
```

실행한 다음 `weight_decay=0.01`을 `0.001`로 바꿔 한 번 더 실행합니다. 두 줄 중 **바뀌지 않는 쪽**이 있습니다.

<details>
<summary>실행한 뒤에 펼치기</summary>

| | `weight_decay=0.01` | `weight_decay=0.001` |
|---|---|---|
| Adam | 0.9000 | 0.9000 |
| AdamW | 0.9990 | 0.9999 |

- AdamW는 식 (10) 그대로입니다. $1.0 \times (1 - 0.1 \times 0.01) = 0.999$. $\lambda$를 10배 줄이면 당김도 10배 줄어 0.9999입니다
- Adam은 식 (9)로 $g = 0 + 0.01 \times 1.0 = 0.01$을 만든 뒤 (8)을 탑니다. 첫 스텝은 그래디언트 크기와 무관하게 lr만큼이므로 0.1을 당겨 0.9. $\lambda$를 10배 줄여도 여전히 0.9입니다 — **$\lambda$의 크기가 나눗셈에 지워졌습니다.** AdamW가 따로 있는 이유가 이것입니다
- `(0 * q)`로 그래디언트를 **0으로** 만든 데는 이유가 있습니다. `.grad`가 `None`이면(`backward()`를 아예 안 했으면) 두 옵티마이저 모두 그 파라미터를 통째로 건너뛰고, weight decay도 적용하지 않습니다

</details>

**실전 대응 (1분)**

| 오늘 | 실전 | 나오는 곳 |
|---|---|---|
| `Adam([big, small], lr=0.1)` | `AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)` | 13주차 이후 Transformer 파인튜닝 |
| `opt.state[q]`의 `exp_avg`·`exp_avg_sq` | 파라미터마다 두 개씩 생기는 상태 | 다음 Step에서 크기를 잰다 |

**⟶ 한 줄로 적기 (1분)** — `big 0.1000   small 0.1000`이 lr이라는 숫자에 대해 뜻하는 것.

### Step 4 · 옵티마이저 상태가 차지하는 VRAM 재기 — SGD · 모멘텀 · Adam (7분) · GPU

> **볼 것** — 첫 `step()` 전후의 `memory_allocated` 증가분이 손계산(파라미터 크기 × 상태 텐서 개수)과 맞는지
>
> **끝나면** — 학습에 드는 VRAM을 "파라미터 + 그래디언트 + 옵티마이저 상태 + 활성값"으로 나눠 손계산할 수 있다
>
> **쓰는 상황** — 모델은 VRAM에 올라가는데 학습 첫 스텝에서 OOM이 날 때. 7주차 이후 배치 크기 산정

**손계산 먼저 (2분).** `nn.Linear(4096, 4096)`의 파라미터는 $4096 \times 4096 + 4096 = 16{,}781{,}312$개이고, float32로 $16{,}781{,}312 \times 4$바이트 $\approx 64.02$ MiB입니다. Step 2·3에서 본 `opt.state`의 내용으로 빈칸을 먼저 채웁니다.

| 옵티마이저 | 파라미터 하나당 상태 텐서 | 예상 증가량 (MiB) |
|---|---|---|
| SGD | | |
| SGD + 모멘텀 | | |
| Adam | | |

Adam의 `state["step"]`은 CPU에 있는 숫자 하나라 VRAM에 잡히지 않습니다.

**측정 (4분)**

**셀 4-1 — 측정 함수** (2주차 1일의 "함수로 감싸 측정" 패턴. 함수가 끝나면 안의 `net`, `opt`가 사라지므로 측정끼리 섞이지 않는다)

```python
import gc
dev = "cuda"

def state_mib(opt_cls, **kw):
    net = nn.Linear(4096, 4096).to(dev)
    opt = opt_cls(net.parameters(), **kw)
    net(torch.randn(8, 4096, device=dev)).sum().backward()   # .grad까지 만들어 둔 상태에서
    before = torch.cuda.memory_allocated()
    opt.step()                                               # 상태는 첫 step()에서 생긴다
    after = torch.cuda.memory_allocated()
    return (after - before) / 2**20
```

**셀 4-2 — 세 옵티마이저 측정**

```python
print(f"SGD        : {state_mib(torch.optim.SGD, lr=0.1):7.2f} MiB")
print(f"SGD + mom  : {state_mib(torch.optim.SGD, lr=0.1, momentum=0.9):7.2f} MiB")
print(f"Adam       : {state_mib(torch.optim.Adam, lr=1e-3):7.2f} MiB")
gc.collect(); torch.cuda.empty_cache()                       # 측정 뒤 정리
```

숫자가 손계산과 크게 다르면 커널을 재시작하고 셀 1-1과 Step 4만 다시 실행합니다.

<details>
<summary>실행한 뒤에 펼치기</summary>

- 상태 텐서는 SGD 0개, 모멘텀 1개(`momentum_buffer`), Adam 2개(`exp_avg`, `exp_avg_sq`)이고, 각각 파라미터와 같은 shape입니다. 증가량이 파라미터 크기의 0배·1배·2배로 찍혔다면 손계산과 일치한 것입니다
- 이 층 하나를 Adam으로 학습하는 한 스텝의 VRAM은 파라미터 64 + 그래디언트 64 + 상태 128 = 약 256 MiB, **파라미터의 4배**입니다. 여기에 활성값(기준값 ③)이 더해집니다
- 상태는 첫 `step()`에서야 만들어집니다. 모델을 올리고 순전파·역전파까지 됐는데 첫 `step()`에서 OOM이 나는 이유가 이것입니다

</details>

**⟶ 한 줄로 적기 (1분)** — Adam 줄의 숫자가 모델 크기와 어떤 관계인지.

### Step 5 · 종합 — 학습 루프를 배열하고, 입력 ×10에서 옵티마이저 비교하기 (15분)

> **볼 것** — ① `zero_grad()` 위치 하나로 학습이 통째로 멈추는 것 ② 입력을 10배로 키우면 SGD는 무너지는데 Adam의 `w`, `b`는 원래 데이터와 같은 값으로 끝나는 것
>
> **끝나면** — 표준 학습 루프를 기억에서 꺼내 쓸 수 있고, 옵티마이저를 바꿀 때 lr을 다시 잡아야 하는 이유를 출력으로 설명할 수 있다
>
> **쓰는 상황** — 3주차 1일 MLP부터 모든 학습 함수의 몸통

**5-1 · 순서 배열 (4분)** — 아래 여섯 블록 중 **네 개**를 골라 `train`의 루프 몸통을 완성합니다. 순서가 여럿 가능하면 개념 A의 규칙(루프 첫 줄)을 따릅니다. 남은 두 블록은 왜 빠지는지 마크다운 셀에 한 줄씩 적습니다.

```
(가)  opt.step()
(나)  loss.backward()
(다)  opt.zero_grad()
(라)  loss = loss_fn(model(x), y)
(마)  opt = torch.optim.Adam(model.parameters(), lr=0.1)
(바)  loss = loss_fn(model(x), y).item()
```

```python
def train(model, opt, x, y, steps=200):
    for step in range(steps):
        # ↓ (가)~(바) 중 네 개를 골라 순서대로
        ...
    return model.weight.item(), model.bias.item(), loss.item()
```

**5-2 · 한 줄 옮겨 보기 (2분)** — 완성한 `train`을 복사해 `train_wrong`을 만들고, `opt.zero_grad()` **한 줄만** `backward()`와 `step()` 사이로 옮깁니다. 그리고 실행합니다.

```python
torch.manual_seed(0)
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
print("시작   :", model.weight.item(), model.bias.item())
print("200스텝:", train_wrong(model, opt, x, y))
```

**5-3 · 입력을 10배로 키우고 그래디언트 크기 보기 (1분)**

```python
x10 = x * 10
y10 = 3 * x10 + 2                        # 정답 w=3, b=2는 그대로

for name, (xx, yy) in {"원래": (x, y), "x×10": (x10, y10)}.items():
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    loss_fn(model(xx), yy).backward()
    print(f"{name:5s} w.grad={model.weight.grad.item():10.2f}   b.grad={model.bias.grad.item():6.2f}")
```

**5-4 · 예측 (2분)** — 모든 실행은 `torch.manual_seed(0)`의 같은 시작점에서 200스텝입니다. 5-3의 출력을 보고 ❓ 칸 네 개를 먼저 적습니다.

| # | 데이터 | 옵티마이저 | lr | 예측 |
|---|---|---|---|---|
| ① | 원래 | SGD | 0.1 | 비교 기준 |
| ② | 원래 | Adam | 0.1 | 비교 기준 |
| ③ | ×10 | SGD | 0.1 | ❓ 수렴 / 느리게 수렴 / 발산 |
| ④ | ×10 | SGD | 0.003 | ❓ `w`와 `b` 중 덜 도착하는 쪽은? |
| ⑤ | ×10 | SGD + 모멘텀 0.9 | 0.003 | ❓ ④보다 나아지는가, 왜? |
| ⑥ | ×10 | Adam | 0.1 | ❓ ②와 비교하면? |

**5-5 · 실행 (2분)**

```python
runs = [
    ("원래", x,   y,   torch.optim.SGD,  dict(lr=0.1)),
    ("원래", x,   y,   torch.optim.Adam, dict(lr=0.1)),
    ("x×10", x10, y10, torch.optim.SGD,  dict(lr=0.1)),
    ("x×10", x10, y10, torch.optim.SGD,  dict(lr=0.003)),
    ("x×10", x10, y10, torch.optim.SGD,  dict(lr=0.003, momentum=0.9)),
    ("x×10", x10, y10, torch.optim.Adam, dict(lr=0.1)),
]
for i, (name, xx, yy, Opt, kw) in enumerate(runs, 1):
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    opt = Opt(model.parameters(), **kw)          # 모델을 만든 바로 다음 줄에서 만든다
    w, b, _ = train(model, opt, xx, yy)
    print(f"{i} {name:5s} {Opt.__name__:4s} {str(kw):32s} w={w:8.4f}  b={b:8.4f}")
```

**5-6 · 조사 (3분)**

① 기준값 ④의 lr 한계식 $2 / (2 \cdot \mathrm{mean}(x^2))$을 두 데이터에 적용합니다.

```python
print(1 / (x ** 2).mean().item(), 1 / (x10 ** 2).mean().item())
```

② 한 줄 바꾸기 — ⑥의 lr을 Adam 기본값 `1e-3`으로 바꾸면, 200스텝 뒤 `w`는 시작점(약 −0.0075)에서 **최대 얼마나** 움직일 수 있을까요? Step 3의 결론으로 먼저 적고 실행합니다.

```python
torch.manual_seed(0)
model = nn.Linear(1, 1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
print(train(model, opt, x10, y10)[:2])
```

<details>
<summary>실행한 뒤에 펼치기</summary>

- **③ 발산합니다.** 5-3에서 `w`의 그래디언트만 100배가 됐습니다. 예측 오차에 한 번, 그래디언트 식에 한 번, $x$가 두 번 곱해지기 때문입니다. `b`의 그래디언트는 그대로입니다. lr 한계는 0.33에서 0.0033으로 100분의 1이 됐고, lr 0.1은 그 30배입니다
- **④ `w`는 도착하고 `b`가 1.56 근처에서 기어갑니다.** lr을 `w`의 한계 아래로 맞추자, 그래디언트가 수백분의 1인 `b`는 한 스텝에 너무 조금 움직입니다. 개념 B의 "한쪽은 가파르고 한쪽은 완만한 지형"이 이것입니다
- **⑤ 모멘텀이 `b`를 끌고 옵니다.** `b` 방향 그래디언트는 매 스텝 같은 부호라 버퍼가 쌓여 최대 10배(Step 2의 9.9997)가 됩니다
- **⑥ = ②, 소수점 넷째 자리까지 같습니다.** `w`의 그래디언트가 100배가 됐지만 식 (8)이 $\sqrt{\hat{v}}$로 나눠 그 100배를 지웁니다. lr 0.1은 여전히 "한 스텝에 약 0.1"입니다
- **lr = 1e-3이면 200스텝에 최대 약 0.2**입니다. Adam의 lr은 이동 폭이라 "움직여야 할 거리 ÷ 스텝 수"로 가늠합니다. 기본값 1e-3은 대부분 ±0.1보다 작은 값에서 출발하는 신경망 가중치에 맞춘 값이지, 3까지 가야 하는 이 문제에 맞춘 값이 아닙니다
- 정리하면 **SGD의 lr은 데이터 스케일에 묶인 배율이고, Adam의 lr은 이동 폭입니다. 옵티마이저를 바꾸면 lr도 다시 잡습니다**

</details>

**⟶ 한 줄로 적기 (1분)** — 결과 ⑥과 ②가 같은 값이라는 출력이 뜻하는 것.

---

## 3교시 (20분) · 실패 케이스

오늘의 실패는 전부 **에러 없이 조용히** 일어납니다. 고칠 코드를 먼저 풀고, 표는 그다음에 읽습니다. 두 과제 모두 끝나면 **진단 근거를 한 줄로** 적습니다 — "어느 출력을 보고 그렇게 판단했는가".

### 과제 1 · 손실이 27.5455에서 멈춘다 (8분)

```python
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.zeros(1, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.w * x + self.b

model = LinearModel()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(100):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
print(f"loss={loss.item():.4f}  w={model.w.item():.4f}  b={model.b.item():.4f}")
```

```
loss=27.5455  w=0.0000  b=2.0000
```

이 숫자는 기준값 ⑦에서 본 적이 있습니다. 그러나 **같은 숫자가 같은 원인이라는 보장은 없습니다.**

1. 원인을 확정하는 출력을 찍고, 진단 근거를 한 줄로 적습니다
2. 고칩니다. 고친 뒤 `len(list(model.parameters()))`, `w`, `b`를 찍습니다

<details>
<summary>힌트 1</summary>

기준값 ⑦의 확정법부터 — `print((model(x) - y).shape)`. `(100, 100)`이 아니면 브로드캐스팅은 범인이 아닙니다.

</details>

<details>
<summary>힌트 2</summary>

`model.w.item()`과 `model.w.grad`를 같이 찍습니다. 하나는 한 번도 안 변했고, 하나는 100스텝 동안 불어났습니다. 둘을 **동시에** 설명하는 원인은 하나입니다.

</details>

<details>
<summary>힌트 3</summary>

`len(list(model.parameters()))`

</details>

### 과제 2 · 손실이 소수점까지 똑같이 찍힌다 (7분)

아래를 **세 개의 셀**로 나눠 넣습니다.

```python
# 셀 A — 모델
torch.manual_seed(0)
model = nn.Linear(1, 1)
```

```python
# 셀 B — 옵티마이저
opt = torch.optim.SGD(model.parameters(), lr=0.1)
```

```python
# 셀 C — 학습 5스텝
for step in range(5):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    print(step, f"{loss.item():.6f}")
```

1. A → B → C 순서로 실행합니다. 손실이 줄어드는 것을 확인합니다
2. "처음부터 다시 보고 싶다" — 이번에는 **A → C**만 실행합니다(B를 건너뜀). 손실 다섯 개를 봅니다
3. 진단 근거를 한 줄로 적고, **같은 실수가 다시 안 나도록 셀 구성을 바꿉니다**

<details>
<summary>힌트 1</summary>

Step 1의 셀 1-6. 옵티마이저 명단의 첫 원소가 **지금의** `model.weight`와 같은 객체인가요?

</details>

<details>
<summary>힌트 2</summary>

C를 실행할 때마다 `model.weight.grad`를 찍어 봅니다. 이 값을 지워 주는 코드가 있나요?

</details>

### 실패 케이스 표 (5분)

| 증상 | 원인 | 해결 |
|---|---|---|
| 손실이 어떤 값에서 멈추고, 특정 파라미터만 초깃값 그대로다. 그 파라미터의 `.grad`는 불어나 있다 | `nn.Parameter`가 아닌 텐서 → `model.parameters()` 명단 밖 → `step()`도 `zero_grad()`도 건너뛴다 | `nn.Parameter`로 감싸고 모델·옵티마이저를 다시 만든다. 확인은 `len(list(model.parameters()))` |
| 손실이 매 스텝 소수점까지 같다 | 옵티마이저가 옛 모델의 파라미터를 들고 있다 (노트북에서 모델 셀만 재실행) | 모델과 옵티마이저를 **한 셀에서** 만든다. 확인은 `opt.param_groups[0]["params"][0] is model.weight` |
| 모델의 마지막 층을 새 층으로 바꿔 끼웠는데 그 층이 학습되지 않는다 | 교체 전에 만든 옵티마이저 → 새 층이 명단에 없다 | 층을 교체한 **뒤에** 옵티마이저를 만든다 (7주차 3일 전이학습에서 하는 작업) |
| 손실이 전혀 안 변하고, 에러도 없다 | `zero_grad()`가 `backward()`와 `step()` 사이 → `step()` 시점에 `.grad`가 `None` | `zero_grad()`는 루프 첫 줄 |
| 손실이 커지다가 `inf` → `nan` | SGD의 lr이 데이터 스케일이 정한 한계를 넘었다 (기준값 ④) | lr을 낮추거나 입력을 정규화한다 (7주차 2일) |
| 옵티마이저만 바꿨는데 손실이 요동친다(SGD → Adam) 또는 거의 안 준다(Adam → SGD) | 이전 옵티마이저의 lr을 그대로 썼다. SGD의 lr은 배율, Adam의 lr은 이동 폭 | Adam 계열은 1e-3, SGD는 0.01~0.1에서 다시 출발 |
| 모델도 올라가고 순전파·역전파도 되는데 첫 `step()`에서 CUDA OOM | Adam 상태($m$, $v$)가 첫 `step()`에서 할당된다 — 파라미터 크기의 2배 (Step 4) | 배치를 줄인다. 16주차 1일에 8-bit optimizer로 상태 자체를 줄인다 |
| `p.grad.norm()`에서 `'NoneType' object has no attribute 'norm'` | `zero_grad()` 뒤 `.grad`는 `None`이다 (기본값 `set_to_none=True`) | `.grad` 확인은 `backward()` 뒤, `zero_grad()` 앞 |
| 학습을 이어서 돌리자 초반 손실이 튄다 | 모델 `state_dict`만 저장하고 옵티마이저 상태(버퍼, $m$, $v$)는 안 저장했다 | `opt.state_dict()`도 함께 저장·로드한다 (7주차 4일 체크포인트) |

---

## 마무리 (15분)

### 통과 기준 (4분)

- [ ] 셀 1-5 — `optim.SGD`로 학습한 `w`, `b`와 손 갱신 결과의 최대 차이가 `1e-6` 미만 (보통 `0.0`)
- [ ] 셀 1-6 — `opt.param_groups[0]["params"][0] is model_b.weight`가 `True`, `... is model_a.weight`가 `False`
- [ ] Step 3 — Adam 첫 `step()` 뒤 `big`, `small`이 둘 다 `0.1000`만큼 움직이고, 식 (7)·(8) 손계산 셀도 `0.1000`
- [ ] 5-2 — `zero_grad()`를 `backward()`와 `step()` 사이로 옮기면 200스텝 뒤에도 `w`, `b`가 시작값 그대로
- [ ] 5-5 — ③(×10 · SGD · lr 0.1)이 `nan`이고, ⑥(×10 · Adam · lr 0.1)의 `w`, `b`가 ②와 소수점 넷째 자리까지 같음
- [ ] 과제 1 — 고친 뒤 `len(list(model.parameters()))`가 `2`, `w`가 2.9~3.1, `b`가 1.9~2.1

### 실전 대응표 — 1주차 4일 표 완성 (1분)

| 1주차 4일 손으로 | 실전 | 채운 세션 |
|---|---|---|
| `w`, `b` 텐서 + `w * x + b` | `nn.Linear` · `model(x)` | 2주차 1일 |
| `((y_hat - y)**2).mean()` | `nn.MSELoss()` · `nn.CrossEntropyLoss()` | 2주차 2일 |
| `w.grad.zero_()` · `b.grad.zero_()` | `opt.zero_grad()` | **오늘** (2주차 1일엔 `model.zero_grad()`) |
| `loss.backward()` | 그대로 | — |
| `with torch.no_grad(): w -= lr * w.grad` … | `opt.step()` | **오늘** |
| 데이터 전체를 한 번에 | `DataLoader` | 2주차 4일 |

### 확인 퀴즈 (8분)

오늘 다룬 내용에서만 나옵니다. 답은 채팅으로 보내 주세요. 정답과 해설은 채팅에서 공개합니다.

**퀴즈 1.** 출력은? (`x`, `y`, `loss_fn`은 Step 1의 것)

```python
torch.manual_seed(0)
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
model.weight = nn.Parameter(torch.zeros(1, 1))     # 가중치를 새 파라미터로 교체

for step in range(100):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
print(round(model.weight.item(), 4))
```

A. `3.0`  B. `0.0`  C. 에러  D. `nan`

**퀴즈 2.** `SGD([p], lr=0.1, momentum=0.9)`에서 매 스텝 그래디언트가 `2`로 나온다. 세 번째 `step()`에서 `p`가 움직이는 거리는?

A. 0.2000  B. 0.3800  C. 0.6000  D. 0.5420

**퀴즈 3.** 출력은?

```python
a = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.Adam([a, b], lr=0.01)
(-50 * a.sum() + 0.2 * b.sum()).backward()        # a.grad = -50, b.grad = 0.2
opt.step()
print(round(a.item(), 4), round(b.item(), 4))
```

A. `0.01 -0.01`  B. `0.5 -0.002`  C. `-0.01 0.01`  D. `0.01 -0.002`

**퀴즈 4.** 새로 만든 모델과 `SGD(lr=0.1)`로 아래 루프를 100스텝 돌렸다. 결과는?

```python
for step in range(100):
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.zero_grad()
    opt.step()
```

A. `w`가 3.0 근처 — 순서와 상관없이 학습된다
B. `nan` — 그래디언트가 누적되어 발산한다
C. `w`가 초깃값 그대로
D. `RuntimeError` — `.grad`가 `None`이라 `step()`이 실패한다

**퀴즈 5.** `SGD(lr=0.1)`로 잘 학습되던 신경망 코드에서 옵티마이저만 `Adam(lr=0.1)`로 바꿨더니 손실이 요동친다. 가장 맞는 설명은?

A. Adam은 그래디언트에 lr을 곱하므로, 그래디언트가 큰 층이 폭주한다
B. Adam에는 모멘텀 같은 장치가 없어서 진동이 줄지 않는다
C. Adam은 `zero_grad()`가 필요 없어서 그래디언트가 누적된다
D. Adam은 파라미터마다 매 스텝 약 lr만큼 움직이므로, 대부분 ±0.1보다 작은 가중치가 매 스텝 통째로 흔들린다

**퀴즈 6.** 파라미터 1억 개(float32)인 모델을 Adam으로 학습한다. 파라미터 자체(약 0.4 GB)를 빼고, 그래디언트와 옵티마이저 상태가 **추가로** 차지하는 메모리는? (활성값 제외)

A. 약 0.8 GB  B. 약 1.2 GB  C. 약 0.4 GB  D. 약 1.6 GB

### 다음 세션 (2분)

**2주차 4일 · Dataset / DataLoader.** 오늘 완성한 루프 네 줄은 그대로 두고, 그 바깥에 `for xb, yb in loader:` 한 겹이 생깁니다. 새 수식 없이 MNIST 배치·셔플과 `num_workers`·`pin_memory` 측정이 중심이라 오늘보다 가볍습니다. 시작할 때 `torchvision` 설치가 있습니다.
