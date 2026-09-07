# 1주차 2일 · 텐서 조작

**배정 시간 2시간** · 산출물 `week01/d2_tensor.ipynb`

---

## 도입 (2분) · 직전 미해결 항목 확인

1일차에 비어 있던 두 칸이 채워졌습니다. 오늘 Step 5에서 이 값을 그대로 씁니다.

| 항목 | 값 | 오늘 쓰이는 곳 |
|---|---|---|
| VRAM | 15.99 GiB (16GB 모델) | Step 5 — 텐서 하나가 VRAM을 얼마나 먹는지 계산 |
| 드라이버 / CUDA | 595.97 (13.2) | 추가 조치 없음 |

미해결 항목 둘 다 해소. 도입 시간을 더 쓰지 않고 1교시로 갑니다.

---

## 1교시 (25분) · 개념

오늘 처음 나오는 용어는 네 개입니다. 하나씩 끊어서 봅니다.

### 개념 A (7분) · 텐서, 그리고 그 세 가지 꼬리표

**텐서는 숫자를 담는 상자입니다.** 엑셀 시트를 떠올리면 됩니다. 숫자 하나만 담으면 0차원, 한 줄로 늘어놓으면 1차원, 표가 되면 2차원입니다. 사진 한 장은 가로·세로·색상이 있으니 3차원이고요.

파이썬 리스트로도 숫자는 담을 수 있는데 왜 굳이 텐서를 쓸까요. 두 가지 때문입니다. **GPU에서 돌릴 수 있고, 미분을 자동으로 계산해 줍니다.** 딥러닝은 이 둘이 없으면 성립하지 않습니다.

그래서 앞으로 데이터를 다룰 때는 항상 텐서로 바꿔서 시작합니다.

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])   # 2×2 표
print(x.shape)                        # torch.Size([2, 2])
```

텐서를 다루다 보면 에러의 90%가 세 가지 꼬리표에서 나옵니다. 미리 이름을 붙여둡니다.

| 꼬리표 | 무엇인가 | 안 맞으면 |
|---|---|---|
| `shape` | 숫자가 어떤 모양으로 놓였는가 | 연산이 거부되거나, 더 나쁘게는 **조용히 엉뚱한 모양**이 나온다 |
| `dtype` | 숫자 하나를 몇 비트로 저장하는가 | 타입 에러, 또는 메모리 낭비 |
| `device` | 어느 메모리에 올라와 있는가 (CPU RAM / GPU VRAM) | 다른 device끼리 연산하면 에러 |

이 세 개를 **찍어보는 습관**이 오늘 세션에서 가장 중요합니다. 앞으로 6개월 동안 막힐 때마다 제일 먼저 할 일이 `print(x.shape, x.dtype, x.device)`입니다.

```python
x = torch.zeros(3, 4)
print(x.shape, x.dtype, x.device)   # torch.Size([3, 4]) torch.float32 cpu
```

기본값은 `float32` + `cpu`입니다. 명시하지 않으면 항상 여기서 시작합니다.

### 개념 B (6분) · stride — 텐서는 사실 한 줄이다

**shape는 "우리가 보는 모양"이고, 실제 메모리에는 숫자가 한 줄로 쭉 늘어서 있습니다.** 2×3 표라고 해서 메모리가 2차원인 게 아닙니다. `[0,1,2,3,4,5]` 여섯 칸이 나란히 있을 뿐이고, "3칸마다 줄이 바뀐다"는 정보를 따로 들고 있는 겁니다. 그 정보가 **stride**입니다.

왜 알아야 하냐면, 이걸 모르면 오늘 나오는 `view` 실패와 "복사가 일어났는지 안 일어났는지"를 설명할 수 없습니다. PyTorch가 모양을 바꿀 때 실제로 하는 일은 대부분 **숫자를 옮기는 게 아니라 stride만 고쳐 쓰는 것**입니다. 그게 공짜인 이유고, 가끔 실패하는 이유입니다.

손이 가는 상황은 이렇습니다. `.t()`나 `.permute()`로 축을 바꿔놓고 `.view()`를 부르면 에러가 나는데, 그때 `stride()`와 `is_contiguous()`를 찍어보면 원인이 3초 안에 보입니다.

```python
x = torch.arange(6).reshape(2, 3)
print(x.stride())          # (3, 1)  → 행 하나 넘어가려면 3칸, 열은 1칸
print(x.t().stride())      # (1, 3)  → 데이터는 그대로, 건너뛰는 규칙만 뒤집힘
print(x.t().is_contiguous())  # False
```

`is_contiguous()`가 `True`면 "메모리 순서대로 읽으면 shape 순서와 같다"는 뜻입니다.

### 개념 C (6분) · view와 reshape

**`view`는 같은 메모리를 다른 모양으로 바라보는 것이고, `reshape`은 그게 안 될 때 복사까지 해주는 것입니다.**

이 구분이 왜 필요할까요. 복사는 비용이고, 공유는 부작용입니다. `view`로 만든 텐서를 고치면 **원본도 같이 바뀝니다.** 같은 메모리니까요. 반대로 `reshape`이 복사를 택한 경우엔 원본이 안 바뀝니다. 즉 `reshape`은 **원본과 연결됐는지 아닌지가 상황에 따라 달라지는** 함수입니다. 이걸 모르고 쓰면 "값이 왜 바뀌었지" 또는 "왜 안 바뀌지"로 하루를 씁니다.

손이 가는 상황: 모델 출력을 `(batch, 10)`으로 펴거나, `(N, 1, 28, 28)` 이미지를 `(N, 784)`로 펼 때 매번 씁니다.

```python
a = torch.zeros(2, 3)
b = a.view(6)        # 메모리 공유
b[0] = 99
print(a[0, 0])       # tensor(99.) ← 원본이 바뀐다

c = a.t().reshape(6) # 연속이 아니라 복사됨
c[0] = -1
print(a[0, 0])       # tensor(99.) ← 이번엔 안 바뀐다
```

실무 기준은 단순합니다. **모양만 바꾸고 싶으면 `reshape`을 쓰고, 복사가 절대 없어야 하는 자리에서만 `view`를 씁니다.** `view`가 에러를 내주는 게 오히려 안전장치라, 성능이 걸린 코드에서는 `view`를 선호합니다.

### 개념 D (6분) · 브로드캐스팅

**브로드캐스팅은 모양이 다른 두 텐서를 계산할 때, 작은 쪽을 자동으로 늘려서 맞춰주는 규칙입니다.**

이게 없으면 어떻게 될까요. 이미지 100장의 RGB 채널마다 다른 평균값을 빼려면, 채널 값 3개를 100×3×224×224 크기로 직접 복제한 뒤 빼야 합니다. 메모리도 낭비고 코드도 길어집니다. 브로드캐스팅이 있으면 그냥 뺍니다.

규칙은 **뒤 차원부터 맞춰본다**는 것 하나입니다.

```
   (100, 3, 224, 224)
 - (       3,   1,   1)
   ─────────────────────
뒤에서부터: 224 vs 1 → OK(1은 늘어남)
            224 vs 1 → OK
              3 vs 3 → OK(같음)
            100 vs 없음 → OK(없는 차원은 1로 간주)
   = (100, 3, 224, 224)
```

정리하면 각 자리가 **① 같거나 ② 한쪽이 1이거나 ③ 한쪽에 아예 없으면** 통과합니다. 셋 다 아니면 에러입니다.

```python
img  = torch.randn(2, 3, 4, 4)      # 이미지 2장
mean = torch.tensor([0.5, 0.4, 0.3]).view(3, 1, 1)
print((img - mean).shape)           # torch.Size([2, 3, 4, 4])
```

여기서 `.view(3, 1, 1)`을 왜 붙였는지가 핵심입니다. 그냥 `(3,)` 상태로 빼면 뒤 차원부터 비교하니 `4 vs 3`에서 터집니다. **브로드캐스팅은 오른쪽 정렬**이라, 원하는 축에 값을 걸려면 1로 채운 차원을 직접 만들어줘야 합니다.

> 조용히 잘못되는 경우가 진짜 위험합니다. `(5,)`와 `(5, 1)`을 더하면 에러가 아니라 `(5, 5)`가 나옵니다. 규칙상 완전히 정상이라 아무도 안 막아줍니다. 3교시에서 다시 봅니다.

---

## 2교시 (60분) · 실습

`week01/d2_tensor.ipynb`를 새로 만듭니다. **추가 설치할 패키지는 없습니다.**

셀은 아래 Step 구분대로 나눠 만드세요. 커널은 계속 유지되므로 앞 셀의 변수를 그대로 씁니다. 커널 재시작이 필요한 지점은 Step 5 끝에 표시해 두었습니다.

### Step 1 (8분) · 텐서 만들기와 꼬리표 찍기

**셀 1-1**

```python
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])     # 파이썬 리스트에서
b = torch.zeros(2, 3)                         # 0으로 채워서
c = torch.randn(2, 3)                         # 표준정규분포 난수
d = torch.arange(6)                           # 0~5 연속값

for name, t in [("a", a), ("b", b), ("c", c), ("d", d)]:
    print(f"{name}: shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}")
```

**확인할 것:** `a`만 `int64`고 나머지는 `float32`입니다. 정수 리스트를 넣으면 정수 텐서가 됩니다. 이게 나중에 손실 함수에서 에러를 내는 첫 번째 원인입니다(3교시 표 참조).

**셀 1-2** — 차원 수를 눈으로 익힙니다.

```python
print(torch.tensor(3.0).shape)        # torch.Size([])      0차원(스칼라)
print(torch.tensor([3.0]).shape)      # torch.Size([1])     1차원, 원소 1개
print(a.ndim, a.numel())              # 2 6  ← 차원 수, 전체 원소 수
```

`torch.Size([])`와 `torch.Size([1])`은 다릅니다. 손실값을 다룰 때 이 차이가 나옵니다.

### Step 2 (10분) · dtype과 메모리, 그리고 device 이동

**셀 2-1** — 숫자 하나가 몇 바이트인가.

```python
for dt in [torch.float64, torch.float32, torch.float16, torch.int64, torch.uint8]:
    t = torch.zeros(1000, 1000, dtype=dt)
    mb = t.numel() * t.element_size() / 1024**2
    print(f"{str(dt):<16} element_size={t.element_size()}B   1000x1000 = {mb:6.2f} MiB")
```

**확인할 것:** `float32`는 4MiB, `float64`는 8MiB입니다. 딥러닝이 기본을 `float32`로 잡은 이유가 여기 있습니다. `float64`는 정확도를 두 배 얻지도 못하면서 메모리와 대역폭을 두 배 씁니다.

**셀 2-2** — dtype 바꾸기.

```python
x = torch.arange(5)                  # int64
print(x.dtype, x.float().dtype)      # torch.int64 torch.float32
print(x.to(torch.float16).dtype)     # torch.float16

y = torch.tensor([1.9, -1.9])
print(y.to(torch.int64))             # tensor([ 1, -1]) ← 반올림이 아니라 버림
```

**확인할 것:** float→int 변환은 **0 방향으로 버립니다.** 반올림이 아닙니다.

**셀 2-3** — GPU로 올리기.

```python
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

x_cpu = torch.randn(3, 3)
x_gpu = x_cpu.to(dev)
print(x_cpu.device, x_gpu.device)

try:
    x_cpu + x_gpu
except RuntimeError as e:
    print("RuntimeError:", str(e)[:80])
```

**확인할 것:** 마지막 에러를 반드시 한 번 눈으로 보세요. `.to(dev)`는 **복사**이지 이동이 아니라서, `x_cpu`는 CPU에 그대로 남아 있습니다. 이 에러가 앞으로 가장 자주 만날 에러입니다.

### Step 3 (12분) · shape 다루기 — stride, view, reshape, permute

**셀 3-1** — stride를 관찰합니다.

```python
x = torch.arange(12).reshape(3, 4)
print(x)
print("shape   :", tuple(x.shape))
print("stride  :", x.stride())          # (4, 1)
print("contig  :", x.is_contiguous())   # True
```

**셀 3-2** — 전치하면 stride만 바뀝니다.

```python
xt = x.t()
print("shape   :", tuple(xt.shape))     # (4, 3)
print("stride  :", xt.stride())         # (1, 4)
print("contig  :", xt.is_contiguous())  # False
print("같은 메모리인가:", xt.data_ptr() == x.data_ptr())   # True
```

**확인할 것:** `data_ptr()`이 같습니다. 전치는 숫자를 하나도 옮기지 않았습니다.

**셀 3-3** — `view` 실패를 직접 재현합니다.

```python
try:
    xt.view(12)
except RuntimeError as e:
    print("RuntimeError:", str(e)[:120])

print(xt.reshape(12))              # reshape은 복사해서 성공
print(xt.contiguous().view(12))    # 명시적으로 연속화하면 view도 성공
```

**확인할 것:** 에러 메시지에 `contiguous`라는 단어가 들어 있습니다. 앞으로 이 단어를 보면 "축을 바꿔놓고 view를 불렀구나"로 바로 이어가세요.

**셀 3-4** — `permute`와 차원 추가/제거.

```python
img = torch.randn(2, 3, 32, 32)                 # (N, C, H, W) — PyTorch 기본 순서
print(img.permute(0, 2, 3, 1).shape)            # (2, 32, 32, 3) — 시각화용 순서

v = torch.randn(5)
print(v.unsqueeze(0).shape, v.unsqueeze(1).shape)   # (1,5) (5,1)  차원 추가
print(v.unsqueeze(0).squeeze().shape)               # (5,)         크기 1 차원 제거
```

**확인할 것:** `unsqueeze`는 개념 D에서 `.view(3,1,1)`로 했던 일을 더 읽기 쉽게 해주는 함수입니다. 브로드캐스팅 축을 맞출 때 실제로 이걸 씁니다.

### Step 4 (12분) · 브로드캐스팅

**셀 4-1** — 규칙을 표로 확인합니다.

```python
cases = [((3, 1), (1, 4)), ((2, 3, 4), (4,)), ((5,), (5, 1)), ((2, 3), (3, 2))]
for s1, s2 in cases:
    try:
        r = tuple((torch.zeros(*s1) + torch.zeros(*s2)).shape)
        print(f"{str(s1):<12} + {str(s2):<10} = {r}")
    except RuntimeError:
        print(f"{str(s1):<12} + {str(s2):<10} = 에러")
```

**확인할 것:** 세 번째 줄 `(5,) + (5,1) = (5,5)`입니다. 원소별 덧셈을 의도했다면 **완전히 틀린 결과인데 에러가 안 납니다.** 이게 오늘 배우는 것 중 가장 실전에서 아픈 지점입니다.

**셀 4-2** — 실제 용도: 채널별 정규화.

```python
img  = torch.randn(4, 3, 8, 8)                     # 이미지 4장
mean = torch.tensor([0.485, 0.456, 0.406])
std  = torch.tensor([0.229, 0.224, 0.225])

try:
    (img - mean)
except RuntimeError as e:
    print("그냥 빼면:", str(e)[:70])

m = mean.view(3, 1, 1)                              # 또는 mean[:, None, None]
out = (img - m) / std.view(3, 1, 1)
print(out.shape)                                    # (4, 3, 8, 8)
print(out[:, 0].mean().item(), out[:, 1].mean().item())
```

**확인할 것:** 여기 쓴 `mean`/`std` 숫자는 ImageNet 통계값입니다. 7주차에서 "왜 이 값을 쓰는가"를 다룹니다. 오늘은 브로드캐스팅 예제로만 씁니다.

**셀 4-3** — 브로드캐스팅은 메모리를 안 쓴다는 걸 확인합니다.

```python
big = torch.zeros(1000, 1000)
row = torch.zeros(1000)

print("row 실제 크기 :", row.numel() * row.element_size() / 1024**2, "MiB")
print("expand 후 shape:", tuple(row.expand(1000, 1000).shape))
print("expand 후 stride:", row.expand(1000, 1000).stride())   # (0, 1)
```

**확인할 것:** stride 첫 값이 `0`입니다. "행이 바뀌어도 메모리는 안 움직인다" = 같은 한 줄을 1000번 다시 읽는다는 뜻입니다. 브로드캐스팅이 공짜인 이유가 이 `0`입니다.

### Step 5 (12분) · CPU↔GPU 이동 비용과 VRAM

GPU가 없으면 이 Step은 읽고 넘어갑니다.

**셀 5-1** — 이동 시간을 제대로 재기.

```python
import time
n = 4096
x_cpu = torch.randn(n, n)                 # 64 MiB (float32)

torch.cuda.synchronize()
t = time.time(); x_gpu = x_cpu.to("cuda"); torch.cuda.synchronize()
print(f"CPU→GPU : {time.time()-t:.4f}s")

torch.cuda.synchronize()
t = time.time(); _ = x_gpu.cpu(); torch.cuda.synchronize()
print(f"GPU→CPU : {time.time()-t:.4f}s")

torch.cuda.synchronize()
t = time.time(); _ = x_gpu @ x_gpu; torch.cuda.synchronize()
print(f"GPU matmul : {time.time()-t:.4f}s")
```

**확인할 것:** `synchronize()`가 왜 필요한지는 1일차에 봤습니다(CUDA 호출은 비동기). 여기서는 **이동 시간과 계산 시간을 비교**하는 게 목적입니다. 4096×4096 곱셈은 원소 하나당 4096번의 곱셈-덧셈을 하는데, 그것과 단순 복사가 비슷한 시간대에 있다면 데이터 이동이 얼마나 비싼지 알 수 있습니다.

**셀 5-2** — 나눠 옮기기 vs 한 번에 옮기기.

```python
chunks = [torch.randn(512, 512) for _ in range(64)]     # 합계는 위와 동일한 64 MiB
one    = torch.randn(4096, 4096)

torch.cuda.synchronize()
t = time.time()
for c in chunks: c.to("cuda")
torch.cuda.synchronize()
print(f"64번 나눠 이동 : {time.time()-t:.4f}s")

torch.cuda.synchronize()
t = time.time(); one.to("cuda"); torch.cuda.synchronize()
print(f"1번에 이동     : {time.time()-t:.4f}s")
```

**확인할 것:** 총 바이트 수는 같은데 시간이 다릅니다. 호출마다 붙는 고정 비용이 있기 때문입니다. **2주차 4일 DataLoader에서 배치 단위로 옮기는 이유가 이겁니다.**

**셀 5-3** — VRAM을 얼마나 쓰는지 직접 셉니다.

```python
torch.cuda.empty_cache()
base = torch.cuda.memory_allocated() / 1024**2
batch = torch.randn(32, 3, 224, 224, device="cuda")     # 흔한 학습 배치 한 개
used = torch.cuda.memory_allocated() / 1024**2 - base

print(f"텐서 1개      : {used:.2f} MiB")
print(f"직접 계산     : {32*3*224*224*4/1024**2:.2f} MiB")
print(f"reserved      : {torch.cuda.memory_reserved()/1024**2:.2f} MiB")
print(f"내 VRAM       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB")
```

**확인할 것:** 계산식은 `배치 × 채널 × 높이 × 너비 × 4바이트`입니다. 이 배치 하나가 약 18 MiB인데, 내 VRAM은 16 GiB입니다. 그러면 배치를 800개 넣어도 되나요? **아닙니다.** 학습 중에는 이 입력 말고도 (1) 모델 파라미터, (2) 그래디언트, (3) 옵티마이저 상태, (4) 중간 활성값이 함께 올라갑니다. 보통 (4)가 가장 큽니다. 이 계산을 제대로 하는 건 7주차 이후 과제고, 오늘은 **입력 텐서 크기를 손으로 셀 수 있다**까지가 목표입니다.

`memory_allocated`와 `memory_reserved`가 다르다는 것도 봐두세요. PyTorch는 OS에 매번 요청하지 않고 미리 크게 받아 쪼개 씁니다. `nvidia-smi`에 보이는 값은 `reserved` 쪽입니다.

**셀 5-4** — 정리.

```python
del batch, x_gpu, one, chunks
torch.cuda.empty_cache()
print(f"allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MiB")
```

> **커널 재시작 지점.** 위 출력이 0에 가깝지 않으면 참조가 남아 있는 것입니다. 노트북에서는 셀 출력이나 `_` 같은 변수가 텐서를 붙잡고 있는 경우가 흔합니다. 굳이 추적하지 말고 커널을 재시작하세요. Step 6은 재시작 후에 진행해도 됩니다.

### Step 6 (6분) · 스스로 확인

아래를 각각 한 줄로 작성해 보고 결과 shape를 예측한 뒤 실행하세요. 예측과 다르면 그 자리에서 stride를 찍어봅니다.

```python
# 1. (8, 1, 28, 28) 이미지 배치를 (8, 784)로 편다
# 2. (16, 10) 로짓에서 각 행의 최댓값 위치를 뽑는다  (힌트: argmax의 dim 인자)
# 3. (4, 3) 텐서와 (4,) 텐서를 원소별로 곱해 (4, 3)을 만든다
# 4. (2, 3, 4) 텐서를 (4, 3, 2)로 축만 뒤집는다
```

정답 코드를 노트북 마지막 셀에 남기세요. 4번은 `view`로는 안 됩니다. 왜 안 되는지 한 줄 주석으로 적어두면 좋습니다.

---

## 3교시 (20분) · 실패 케이스

오늘 다 겪지 않아도 됩니다. 읽어두면 나중에 검색 시간을 아낍니다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `view size is not compatible ... use .reshape()` | `.t()`/`.permute()` 뒤라 연속이 아님 | `.reshape()`을 쓰거나 `.contiguous().view()` |
| `Expected all tensors to be on the same device` | 한쪽만 `.to(dev)` 했음. `.to()`는 복사라 원본은 남는다 | 모델과 데이터 모두 같은 device로. 대입을 잊지 않기(`x = x.to(dev)`) |
| `expected scalar type Long but found Float` (또는 반대) | 정수 리스트로 만든 텐서는 `int64`. `CrossEntropyLoss`는 입력은 float, 정답 라벨은 long을 요구 | 입력 `.float()`, 라벨 `.long()` |
| `mean(): could not infer output dtype` | `int64` 텐서에 `mean()`을 부름. 정수의 평균은 정수가 아니니 PyTorch가 거부 | `.float()` 후 호출 |
| 에러 없이 결과 shape가 커짐 — `(5,)+(5,1)`이 `(5,5)` | 브로드캐스팅 규칙상 정상. 아무도 안 막아줌 | 연산 전후 `shape`를 assert로 박아둔다 |
| 원본 텐서 값이 저절로 바뀜 | `view`/슬라이싱이 메모리를 공유. in-place 연산(`add_`, `x[0]=`)이 원본까지 고침 | 독립 사본이 필요하면 `.clone()` |
| numpy로 바꿨는데 값이 같이 변함 | `torch.from_numpy` / `.numpy()`는 CPU에서 **메모리를 공유** | 사본이 필요하면 `torch.tensor(arr)` 또는 `.copy()` |
| `CUDA out of memory` | 배치나 해상도가 큼. 또는 노트북이 이전 셀 텐서를 붙잡고 있음 | `del` + `empty_cache()`, 안 되면 커널 재시작. 배치 축소 |
| GPU가 CPU보다 느리게 나옴 | 텐서가 너무 작아 이동·호출 비용이 계산보다 큼 | 정상이다. 작은 연산은 CPU가 이긴다 |

**직접 재현해 볼 것 두 개** (5분)

```python
# ① 조용히 잘못되는 브로드캐스팅
pred   = torch.randn(5)
target = torch.randn(5, 1)          # 실수로 (5,1)로 만들어진 라벨
loss = ((pred - target) ** 2).mean()
print(loss.item(), (pred - target).shape)   # 에러 없이 (5,5)에서 평균이 나온다
```

```python
# ② 메모리 공유
x = torch.arange(6.)
y = x[:3]            # 슬라이싱도 view다
y += 100
print(x)             # tensor([100., 101., 102., 3., 4., 5.])
```

①이 실전에서 무서운 이유는, 손실값이 그럴듯하게 나오고 학습도 돌아간다는 점입니다. 성능만 이상하게 안 오릅니다. **6주차 3일 학습 디버깅 체크리스트의 첫 항목이 shape 확인인 이유입니다.**

---

## 마무리 (15분)

### 커밋 과제

```
week01/d2_tensor.ipynb
```

Step 1~6 셀이 위에서 아래로 실행되는 상태로 저장합니다. 실행 출력을 지우지 마세요 — 3개월 뒤 본인 GPU에서 나온 수치가 남아 있어야 비교가 됩니다.

커밋 메시지:

```
week01 d2: tensor shape/dtype/device, stride & view, broadcasting, H2D transfer cost
```

Step 5-3에서 확인한 배치 크기 계산(18 MiB / 16 GiB)과 Step 5-2의 이동 시간 비교값은 `NOTES.md` 환경 실측값에 남깁니다. 7주차 배치 크기 산정에서 다시 씁니다.

### 오늘의 통과 기준

아래 네 개가 모두 만족되면 끝입니다. 더 진도 나가지 마세요.

1. `xt.view(12)`가 `RuntimeError`를 내고, `xt.reshape(12)`는 성공하는 것을 노트북에서 확인했다
2. `(5,) + (5,1)`이 에러 없이 `(5,5)`가 되는 것을 직접 출력했다
3. `(4, 3, 8, 8)` 이미지에 채널별 mean/std를 브로드캐스팅으로 적용해 shape가 유지됨을 확인했다
4. Step 5-3에서 `memory_allocated` 증가분과 손계산 값이 일치했다 (GPU 없으면 손계산만)

### 다음 세션 예고

**1주차 3일 · autograd.** 오늘 만든 텐서에 `requires_grad=True`를 붙이는 순간 PyTorch가 계산 과정을 기록하기 시작합니다. 오늘보다 개념 밀도는 높지만 코드는 짧습니다. 오늘의 `view`/메모리 공유 이야기가 "그래디언트가 왜 누적되는가"로 이어집니다.

---

## 퀴즈

정답은 채팅에서 공개합니다. 먼저 풀어보세요.

**1.** `x = torch.arange(12).reshape(3, 4)`, `y = x.t()` 일 때 `y.view(12)`는 실패한다. 가장 정확한 이유는?

- (a) `y`의 원소 수가 12가 아니기 때문
- (b) `y`는 메모리 순서가 shape 순서와 어긋나 있어서, stride만 고쳐서는 1차원으로 표현할 수 없기 때문
- (c) 전치 연산이 데이터를 복사했기 때문에 원본과 연결이 끊겼기 때문
- (d) `view`는 2차원 이상만 지원하기 때문

**2.** 다음 코드의 마지막 출력은?

```python
a = torch.zeros(2, 3)
b = a.view(6)
b[0] = 7
c = a.t().reshape(6)
c[1] = -5
print(a[0, 0].item(), a[1, 0].item())
```

- (a) `7.0 -5.0`
- (b) `7.0 0.0`
- (c) `0.0 -5.0`
- (d) `0.0 0.0`

**3.** `torch.zeros(4, 1) + torch.zeros(3)`의 결과 shape는?

- (a) 에러
- (b) `(4, 3)`
- (c) `(4, 1)`
- (d) `(3, 4)`

**4.** `labels = torch.tensor([0, 1, 2, 1])`을 만들고 `labels.mean()`을 부르면 에러가 난다. 이유는?

- (a) 원소가 4개뿐이라 통계 연산이 거부됨
- (b) `dtype`이 `int64`인데 정수들의 평균은 정수로 표현할 수 없어 PyTorch가 출력 타입을 정하지 못함
- (c) `mean()`은 2차원 이상에서만 동작함
- (d) `labels`가 CPU에 있어서

**5.** 64 MiB 데이터를 CPU에서 GPU로 옮길 때, 512×512 텐서 64개로 나눠 보내는 것이 4096×4096 텐서 하나로 보내는 것보다 느렸다. 가장 그럴듯한 설명은?

- (a) 작은 텐서는 `float64`로 저장되기 때문
- (b) 전송 호출 한 번마다 붙는 고정 비용이 있어서, 호출 횟수가 많아질수록 총 시간이 늘기 때문
- (c) GPU가 512×512 크기를 처리하지 못해 내부적으로 패딩하기 때문
- (d) `torch.cuda.synchronize()`가 텐서 개수만큼 반복 실행되어 느려진 것이며 실제 전송은 같은 속도이기 때문
