# DefectLab

소량의 defect 예시 이미지로 라벨링 → 증강 → YOLO11 학습 → ONNX export까지 하는 데스크톱 앱입니다.
요구사항은 [PRD.md](PRD.md), 구현 순서는 [PROMPTS.md](PROMPTS.md)를 참고하세요.

지원 OS: **Linux (Ubuntu 22.04+)**, **Windows 10/11 (64bit)**, **macOS 13+ (Apple Silicon: M1 이후)**. Python 3.10 이상 (3.12 권장).

## 설치

PyTorch는 OS와 GPU에 맞는 빌드를 **먼저** 설치한 뒤 앱을 설치합니다.
CUDA 버전별 명령은 [pytorch.org](https://pytorch.org/get-started/locally/)에서 확인하세요 (아래 `cu128`은 예시).

### Linux

```bash
cd "defect inspection"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 또는 CPU 전용
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Qt 실행에 필요한 시스템 라이브러리가 없으면 설치합니다:
`sudo apt install libegl1 libgl1 libxkbcommon0 libxcb-cursor0`

### Windows (PowerShell)

```powershell
cd "defect inspection"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 또는 CPU 전용
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

`Activate.ps1` 실행이 막히면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 실행합니다.

### macOS (Apple Silicon)

```bash
cd "defect inspection"
python3.12 -m venv .venv          # Homebrew: brew install python@3.12
source .venv/bin/activate
python -m pip install --upgrade pip

# PyPI의 macOS용 PyTorch에 Mac GPU(MPS) 지원이 들어 있습니다 (CUDA 없음)
pip install torch torchvision

pip install -e ".[dev]"
python -m defect_inspection.app --self-check   # "계산 장치: GPU: Apple MPS → … 'mps'"가 보이면 정상
```

- 장치 "자동"은 Mac에서 **MPS(Mac GPU)** 로 학습·추론합니다. Train 탭 장치 목록에 "Apple GPU (MPS)"가 따로 있습니다.
  MPS가 지원하지 않는 일부 연산은 자동으로 CPU에서 실행됩니다(`PYTORCH_ENABLE_MPS_FALLBACK=1`, 앱이 설정).
- 단축키의 Ctrl은 Mac에서 ⌘(Command)입니다 (예: ⌘S 저장, ⌘Z 실행 취소).
- 한글 폰트는 macOS 기본 "Apple SD Gothic Neo"를 씁니다.
- 창·탭 설정은 `~/Library/Preferences/com.defectlab.DefectLab.plist`, 증강 프리셋은 `~/Library/Application Support/DefectLab/`, 캐시(썸네일·가중치·matplotlib 폰트)는 `~/Library/Caches/DefectLab/`에 저장됩니다.
- Export 탭에서 ONNX 외에 **CoreML(.mlpackage)** 로도 내보낼 수 있습니다 (macOS 전용, Neural Engine 사용, 컬러 모델만).
- 실제 M1에서 확인할 항목은 [docs/mac_checklist.md](docs/mac_checklist.md)에 있습니다.

#### 가속 측정 (MPS · CoreML · MLX)

```bash
pip install -e ".[bench]"                       # MLX
python tools/bench_apple.py --json bench.json   # 10~20분: 학습 epoch CPU vs MPS, 추론 PyTorch/ONNX Runtime/CoreML, 학습 1 step MLX vs MPS
python tools/bench_apple.py --quick             # 동작 확인용 (몇 분)
```

마지막 줄의 **판단**이 "MLX가 MPS보다 1.5배 이상 빠름"이면 MLX 학습 엔진을 별도 단계로 계획하고, 아니면 MPS를 그대로 씁니다
(PRD 5.2). 결과 표(또는 `bench.json`)를 공유해 주세요.

### uv 사용 시 (모든 OS 공통)

```bash
uv venv .venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # 또는 cu128 (macOS는 --index-url 없이)
uv pip install -e ".[dev]"
```

### SAM 가중치 (자동 라벨 제안)

SAM 클릭 보조(단축키 S)를 처음 쓸 때 `mobile_sam.pt`(약 40MB)를 사용자 캐시 폴더로 내려받습니다
(Linux `~/.cache/DefectLab/weights/`, Windows `%LOCALAPPDATA%\cache\DefectLab\weights\`, macOS `~/Library/Caches/DefectLab/weights/`).
인터넷이 안 되는 PC는 다른 PC에서 받은 `.pt` 파일을 **설정 > SAM 모델 > 파일 선택**으로 지정하세요.

## 실행

```bash
python -m defect_inspection.app
# 또는 설치된 명령
defectlab
# 설치 상태 확인 (화면 없이: 라이브러리, 모델 파일, ONNX Runtime, 한글 폰트, 메인 창)
python -m defect_inspection.app --self-check
```

리포트 그래프에 한글 폰트가 필요합니다. Windows는 기본 "맑은 고딕"을 쓰고, Linux는 Noto Sans CJK KR 또는
나눔고딕을 씁니다 (`sudo apt install fonts-noto-cjk`). 앱이 OS에 맞는 폰트를 자동으로 고릅니다.

## 사용 가이드

화면 왼쪽은 이미지 목록과 라벨링 캔버스, 오른쪽은 `Augment / Train / Result / Compare / Export` 탭입니다.

1. **프로젝트**: `새 프로젝트`로 폴더를 만들고 task(detect = BBox, seg = Polygon)와 색 모드(컬러/그레이)를 고른 뒤
   `이미지 폴더 추가`로 PNG/JPEG/BMP/TIFF(16bit 포함)를 넣습니다. 이미지는 프로젝트 `images/`로 복사됩니다.
2. **라벨링**: BBox `B`, Polygon `P`, 선택 `V`, 지우개 `E`, 실행 취소 `Ctrl+Z`. 결함이 없는 이미지는 `O` 키나 목록 우클릭
   `OK 이미지 (결함 없음) 표시/해제`로 OK로 표시합니다(배경 학습용, 라벨이 있으면 표시할 수 없고 라벨을 그리면 자동 해제). SAM 보조 `S`는 클릭 한 번으로 영역 제안을 만들고, 제안은 `Enter`(수락)/`Backspace`(거절)로
   확정합니다. 라벨은 30초마다, 이미지를 바꿀 때, `Ctrl+S`에 자동 저장됩니다.
3. **Crop ROI (선택)**: `C` 도구로 이 이미지 ROI 또는 "모든 이미지에 적용"(공통 ROI, 고정 지그)을 그립니다.
4. **Augment 탭**: 프리셋(약함/보통/강함/금속 표면/PCB)이나 변환별 강도·확률을 정하고 미리보기 9장으로 확인합니다.
   아래 "Crop 데이터셋"에서 crop 방식(수동 ROI / 결함 중심 / 배경 / 슬라이딩 타일, ATT-YOLO 1280 타일 프리셋)을 정합니다.
   빌드 버튼 옆에서 Full 또는 Crop을 골라 `데이터셋 빌드`. Train/Val은 원본 이미지 단위로 나뉩니다.
5. **Train 탭**: 모델(yolo11n/s/m, ATT-YOLO n/s), 데이터셋(Full/Crop), epoch·이미지 크기·batch 등을 정하고 `학습 시작`.
   loss/mAP 그래프가 실시간으로 그려지고, 중지 후 `재개`할 수 있습니다.
6. **Result 탭**: Val 예측 그리드(정답 파랑/예측 주황, conf 슬라이더), 오검·미검 이미지 더블클릭 → 캔버스에서 라벨 수정.
   이미지를 끌어다 놓으면 바로 추론합니다. Crop 모델은 자동으로 타일링 추론을 씁니다.
   `자동 라벨 제안 생성`으로 미라벨 이미지에 제안을 만들 수 있습니다.
7. **Compare 탭**: `Full + Crop 비교 학습`(두 데이터셋 빌드 → 같은 설정으로 Full, Crop 학습 → 비교) 또는 끝난 run을
   2개 이상 체크하고 `비교`. 모든 모델을 같은 Val 원본 이미지, 같은 conf/IoU로 다시 평가해 지표 표·그래프·예측 비교를 보여주고
   추천 모델을 고릅니다(기본 mAP50-95, 1%p 이내면 빠른 쪽; 우선 지표 변경 가능).
   `리포트` 탭에서 메모를 적고 `HTML 리포트 저장` → `reports/<YYYYMMDD_HHMM>_full_vs_crop.html`
   (그래프·이미지가 들어 있는 파일 하나, 다른 PC 브라우저에서 바로 열림). 리포트 이력에서 열기/삭제/같은 설정으로 재실험.
8. **Export 탭**: run을 골라 ONNX로 내보냅니다. 기본 이름은 `<프로젝트>_<full|crop>_<모델>_<날짜>_<시각>.onnx`이고
   바꿀 수 있습니다. `.metadata.json`(입력 크기·채널·전처리·Crop 모델의 타일 크기/overlap)과 `.classes.txt`가 같이 저장되고,
   onnxruntime 결과가 PyTorch와 같은지 자동 검증합니다.

## 테스트

```bash
python -m pytest                                  # 전체
python -m pytest tests/test_platform_utils.py -v  # 파일 하나
python -m pytest "tests/test_app.py::test_main_window_opens"  # 테스트 하나
```

테스트는 Qt를 화면 없이(`QT_QPA_PLATFORM=offscreen`) 실행합니다. GitHub Actions에서 Ubuntu, Windows, macOS 14(Apple Silicon, CPU만)로 돌아갑니다 (`.github/workflows/ci.yml`).

실제 학습·export가 들어간 느린 테스트는 기본으로 빠지고, 따로 실행합니다 (CPU에서 몇 분):

```bash
python -m pytest -m slow                          # 실제 학습/ONNX 테스트 전체
python -m pytest -m slow tests/test_integration.py  # 합성 결함 5장 → Full/Crop 각 3 epoch 학습 → ONNX → HTML 리포트
```

M1(라벨링 툴) 수동 확인 항목은 [docs/M1_checklist.md](docs/M1_checklist.md)에 있습니다.

## 설치본 만들기 (PyInstaller)

OS마다 그 OS에서 빌드합니다 (크로스 빌드 불가). 앱과 같은 가상환경에 PyInstaller를 추가로 설치합니다.
설치본에는 그 환경의 PyTorch가 들어가므로 GPU용 설치본은 CUDA 빌드 torch 환경에서, CPU용은 CPU 빌드 환경에서 만듭니다
(CUDA torch가 들어가면 수 GB가 됩니다).

| | Linux | Windows 10/11 | macOS (Apple Silicon) |
| --- | --- | --- | --- |
| 준비 | `pip install -e ".[package]"` | `pip install -e ".[package]"` | `pip install -e ".[package]"` |
| 빌드 | `packaging/build_linux.sh` | `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1` | `packaging/build_mac.sh` |
| spec | `packaging/defectlab_linux.spec` | `packaging/defectlab_windows.spec` | `packaging/defectlab_mac.spec` |
| 결과 | `dist/DefectLab/DefectLab` + `dist/DefectLab-linux-<arch>.tar.gz` | `dist\DefectLab\DefectLab.exe` + `DefectLab-worker.exe` + `dist\DefectLab-windows-x64.zip` | `dist/DefectLab-<버전>.app` + `dist/DefectLab-<버전>-macos-arm64.dmg` (빌드 전 `tools/bump_version.py`로 버전 올림) |

- 빌드 스크립트는 끝에 설치본의 `--self-check`를 실행해 빠진 파일이 없는지 확인합니다.
- Windows 설치본은 창 앱 `DefectLab.exe`와 학습용 콘솔 프로그램 `DefectLab-worker.exe` 두 개로 되어 있습니다
  (학습 진행 상황을 표준 출력으로 받기 때문). 학습 중에 콘솔 창은 뜨지 않습니다. Linux는 `DefectLab` 하나가 `--train-worker`로 학습도 합니다.
- GitHub Actions의 `Package` 워크플로(수동 실행)가 Linux·Windows 설치본을 빌드해 artifact로 올립니다 (`.github/workflows/package.yml`).
- **Mac 설치본과 Gatekeeper**: Apple 개발자 인증서 없이 만든 설치본은 ad-hoc 서명만 되어 있어, 다른 Mac에서 처음 열 때
  "확인되지 않은 개발자" 경고가 나옵니다. `.dmg`를 열어 DefectLab을 Applications로 끌어다 놓은 뒤:
  - macOS 13·14: Finder에서 DefectLab을 **Control+클릭 → 열기 → 열기**
  - macOS 15 이후: 한 번 실행해 경고가 뜨면 **시스템 설정 → 개인정보 보호 및 보안 → "그래도 열기"**
  - 또는 터미널에서 `xattr -dr com.apple.quarantine /Applications/DefectLab.app`

  배포용으로 경고를 없애려면 `DEVELOPER_ID="Developer ID Application: …" NOTARY_PROFILE=… packaging/build_mac.sh`로
  서명·공증합니다 (Apple Developer Program 필요).
- 라이선스: Ultralytics는 AGPL-3.0입니다. 설치본을 외부에 배포하려면 AGPL 조건(소스 공개) 또는 Ultralytics 상용 라이선스가 필요합니다.

## 설치본 실행 확인 체크리스트

Linux와 Windows **각각** 깨끗한 PC(개발 환경·Python 없음)에서 확인합니다 (Mac 설치본은 [docs/mac_checklist.md](docs/mac_checklist.md) 8장). 경로에 공백과 한글이 들어간 폴더
(예: `D:\검사 프로젝트\라인 A`)를 프로젝트 위치로 써 봅니다.

| # | 확인 항목 | Linux | Windows |
| --- | --- | --- | --- |
| 1 | 압축을 풀고 실행 파일이 뜬다 (Linux `./DefectLab`, Windows `DefectLab.exe` 더블클릭, SmartScreen 경고 시 "추가 정보 → 실행") | ☐ | ☐ |
| 2 | 설치 확인: Linux `QT_QPA_PLATFORM=offscreen ./DefectLab --self-check`, Windows `cmd /c "DefectLab.exe --self-check 2>&1"` → "모든 확인 통과" | ☐ | ☐ |
| 3 | 한글·공백 경로에 새 프로젝트 → JPEG·PNG 이미지 폴더 추가 → 썸네일·캔버스 표시 | ☐ | ☐ |
| 4 | BBox/Polygon 라벨, 실행 취소, 저장 후 다시 열기 | ☐ | ☐ |
| 5 | Augment 미리보기 9장, Full·Crop 데이터셋 빌드 | ☐ | ☐ |
| 6 | Train: yolo11n 3 epoch 학습이 끝까지 진행되고 그래프가 그려진다 (Windows: 콘솔 창이 뜨지 않는다) | ☐ | ☐ |
| 7 | 학습 중 `중지` → `재개`가 된다 | ☐ | ☐ |
| 8 | Result: Val 그리드, 이미지 드래그&드롭 추론, Crop run의 타일링 표시 | ☐ | ☐ |
| 9 | Compare: Full + Crop 비교 학습 → 지표 표·그래프 → HTML 리포트 저장 → 브라우저에서 한글·그래프가 보인다 | ☐ | ☐ |
| 10 | Export: ONNX 저장(기본 이름에 날짜·시간), 검증 PASS, metadata.json·classes.txt 생성 | ☐ | ☐ |
| 11 | GPU PC: 장치 표시가 "GPU: …"이고 학습이 GPU로 진행된다 | ☐ | ☐ |
| 12 | 앱을 닫고 다시 열면 창 크기와 패널 배치가 유지된다 | ☐ | ☐ |

문제가 있으면 `--self-check` 출력과, 학습 문제라면 프로젝트의 `runs/<run_id>/events.jsonl`·`status.json`을 함께 확인합니다.

## 크로스플랫폼 규칙

- OS별로 달라지는 코드는 `defect_inspection/platform_utils.py`에만 둡니다. 계산 장치(CUDA / MPS / CPU)도 `resolve_device()` 한 곳에서 정합니다.
- 이미지 읽기/쓰기는 `cv2.imread`/`cv2.imwrite` 대신 `imread_unicode`/`imwrite_unicode`를 씁니다 (Windows 한글 경로 대응).
- 경로는 `pathlib`, 텍스트 파일은 UTF-8로 다룹니다.
