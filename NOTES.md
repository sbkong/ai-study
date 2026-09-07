# AI 학습 진행 노트

## 현재 상태
- 마지막 완료 세션: 1주차 1일 · PyTorch 설치 + GPU 인식 확인
- 다음 세션: 1주차 2일 · 텐서 조작 (2시간)
- 미해결 항목:
  - VRAM 용량 미확인 (4060 Ti는 8GB / 16GB 두 종류). 배치 크기 산정에 필요하므로 다음 세션 도입부에 `nvidia-smi`로 확인
  - 드라이버 버전 미기록

## 환경 실측값
| 항목 | 값                             |
|---|-------------------------------|
| GPU | NVIDIA GeForce RTX 4060 Ti    |
| VRAM | 15.99560546875 GiB            |
| 드라이버 / nvidia-smi CUDA | 595.97 (13.2)                 |
| torch | 2.14.0+cu132                  |
| torch.version.cuda | 13.2                          |
| Compute Capability | (8, 9) = sm_89 · Ada Lovelace |
| Python | 3.11.8                        |

**벤치마크 기준값** (4096×4096 matmul, `torch.cuda.synchronize()` 적용)

| 장치 | 시간 |
|---|---|
| GPU | 0.0698s |
| CPU | 0.1424s |

> 초회 실행이라 CUDA 컨텍스트 초기화가 포함된 수치다. GPU/CPU 격차가 2배로 작게 나온 이유이며, 실제 격차는 이보다 크다. 워밍업 후 재측정은 5주차 4일(학습 안정화)에서 다룬다.

## 개발 환경
| 항목 | 값 |
|---|---|
| IDE | PyCharm (unified, Jupyter 무료 티어 포함) |
| venv | `<project>/.venv` — PyCharm Project venv |
| 노트북 | `jupyter` 설치 완료, PyCharm 내장 노트북 사용 |
| 리포지토리 | `.gitignore`에 `.venv/`, `.idea/` 포함 |

> 커리큘럼 문서에는 venv 경로가 `~/venv/ai-study`로 적혀 있으나, 실제로는 PyCharm 프로젝트 내부 `.venv`를 사용한다. 이후 세션의 패키지 설치는 모두 이 환경에 누적된다.

---

## 세션 로그

### 1주차 1일 · PyTorch 설치 + GPU 인식 확인
- 통과 기준: ✅ 충족 — `is_available()` True, GPU matmul이 CPU보다 빠름
- 커밋: `week01/d1_check_env.py` + 실행 출력
- 막힌 점: 없음
- 퀴즈: 4/4
- 메모:
  - **`get_arch_list()`에 `sm_89`가 없는데 정상 동작함.** 목록은 `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`. CUDA 바이너리는 같은 major 아키텍처 안에서는 하위 minor로 컴파일된 코드가 상위 minor 장치에서 실행된다(sm_86 → sm_89). 즉 "내 sm 번호가 목록에 정확히 있어야 한다"가 아니라 "같은 8.x 계열 중 내 minor 이하 번호가 있으면 된다"가 정확한 규칙이다. 퀴즈 3번 해설을 이 케이스로 보정해 둘 것
  - 최신 torch(2.14.0 / cu13.2)라 Blackwell(sm_120)까지 포함된 휠. 버전 문제로 막힐 여지는 당분간 없음
  - PyCharm 환경 및 Jupyter 노트북 세팅을 이 세션에서 함께 완료 (원래 커리큘럼에는 없던 항목)
