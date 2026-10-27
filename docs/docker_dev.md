# Windows에서 Docker(Linux)로 DefectLab 개발 이어 하기

Mac에서 만든 DefectLab을 Windows PC의 Docker(Linux 컨테이너)에서 이어서 개발하는 방법입니다.
컨테이너 안에 Python 3.12, PyTorch(CUDA 또는 CPU), PySide6가 돌아가는 데 필요한 라이브러리, 한글 글꼴,
Claude Code CLI가 들어 있습니다. 소스 폴더는 컨테이너에 연결(bind mount)되므로 코드를 고치면 바로 반영됩니다.

| 파일 | 역할 |
|---|---|
| `docker/Dockerfile` | 개발 이미지 (`/opt/venv`에 패키지를 편집 모드로 설치) |
| `compose.yaml` | `dev`(NVIDIA GPU)와 `dev-cpu` 두 서비스, 소스·캐시·프로젝트 폴더 연결, WSLg 화면 연결 |
| `.devcontainer/devcontainer.json` | VS Code "Reopen in Container"용 |
| `.gitattributes` | Windows에서 받아도 줄바꿈을 LF로 유지 (CRLF면 셸 스크립트가 컨테이너에서 깨짐) |

## 1. 준비 (한 번만)

1. **Windows 11**: WSLg(Linux 앱 창을 Windows 화면에 띄우는 기능)가 기본으로 들어 있습니다.
   Windows 10에서는 앱 창을 띄울 수 없고, 테스트·학습·빌드만 됩니다.
2. **WSL2 + Ubuntu**: PowerShell(관리자)에서 `wsl --install`을 실행하고 재부팅합니다.
3. **Docker Desktop**을 설치합니다. Settings → General에서 "Use the WSL 2 based engine"을 켭니다.
   Settings → Resources → WSL integration에서 Ubuntu를 켭니다.
4. **NVIDIA GPU를 쓸 때**: Windows용 최신 NVIDIA 드라이버만 설치하면 됩니다. WSL 안에 CUDA를 따로 설치하지 않습니다.
   확인: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`

## 2. 소스 가져오기

**Ubuntu(WSL) 터미널에서** 작업합니다. Windows 드라이브(`/mnt/c/...`)에 두면 파일 접근이 몇 배 느리므로,
WSL 홈 폴더에 둡니다.

```bash
# GitHub에서 받기
git clone https://github.com/terrythkim-web/defect_training_m1.git ~/defect-inspection

# 또는 받은 압축 파일에서 (git 이력 포함)
unzip /mnt/c/Users/<이름>/Downloads/DefectLab-dev-<버전>.zip -d ~    # → ~/defect-inspection
cd ~/defect-inspection
git pull                                     # 압축 이후의 변경 받기 (origin = GitHub)
```

## 3. 이미지 만들기와 실행

```bash
cd ~/defect-inspection
docker compose build dev          # NVIDIA GPU용 (처음 한 번 10~20분, CUDA 이미지는 수 GB)
docker compose run --rm dev       # 컨테이너 셸
# GPU가 없으면: docker compose build dev-cpu && docker compose run --rm dev-cpu
```

컨테이너 셸 안에서는 다음을 실행할 수 있습니다.

```bash
python -m pytest                                    # 전체 테스트 (QT_QPA_PLATFORM=offscreen 자동)
python -m defect_inspection.app --self-check        # 환경 점검 (GPU면 "CUDA" 표시)
python -m defect_inspection.app                     # 앱 실행: 창이 Windows 화면에 뜸 (WSLg)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
claude                                              # Claude Code로 이어 개발 (처음에 로그인)
```

- **프로젝트 데이터**: 저장소의 `projects/` 폴더(git에서 제외됨)가 컨테이너의 `/projects`에 연결됩니다.
  다른 폴더를 쓰려면 저장소 폴더에 `.env` 파일을 만들고 `DEFECTLAB_PROJECTS=/home/<이름>/defect-projects`처럼 적습니다.
  Mac에서 쓰던 프로젝트 폴더를 그대로 복사해 오면 됩니다. 경로는 프로젝트 기준 상대 경로라 OS가 바뀌어도 열립니다.
- **모델 가중치·썸네일 캐시**: Docker 볼륨 `defectlab-cache`에 저장됩니다. 이미지를 다시 만들어도 남습니다.
- **의존성을 바꿨을 때**(`pyproject.toml`): `docker compose build dev`로 이미지를 다시 만듭니다. 코드만 바꿨을 때는 필요 없습니다.

### 앱 창이 안 뜰 때 (WSLg)
- Docker Desktop은 WSLg를 `/run/desktop/mnt/host/wslg`로 연결합니다(기본값).
  Docker Desktop 대신 **WSL 안에 Docker Engine을 직접 설치**했다면 `.env`에 `WSLG_ROOT=/mnt/wslg`를 적습니다.
- WSL 터미널에서 `ls /mnt/wslg/.X11-unix`에 `X0`가 보여야 합니다. 없으면 `wsl --update` 후 `wsl --shutdown`을 실행합니다.
- 그래도 안 되면 `QT_QPA_PLATFORM=wayland python -m defect_inspection.app`로 실행해 봅니다.

## 4. VS Code로 개발 (선택)
1. Windows VS Code에 "WSL"과 "Dev Containers" 확장을 설치합니다.
2. WSL 터미널의 저장소 폴더에서 `code .`를 실행합니다.
3. 명령 팔레트에서 "Dev Containers: Reopen in Container"를 실행합니다.
GPU가 없으면 `.devcontainer/devcontainer.json`의 `"service"`를 `"dev-cpu"`로 바꿉니다.
Python 해석기는 `/opt/venv/bin/python`이고, Claude Code 확장도 설치됩니다.

## 5. Linux 실행 파일 만들기 (선택)
컨테이너 안에서 다음을 실행합니다.

```bash
python tools/bump_version.py        # 빌드마다 버전 올리기 (pyproject.toml + __init__.py)
packaging/build_linux.sh            # → dist/DefectLab/DefectLab, dist/DefectLab-<버전>-linux-x86_64.tar.gz
```

결과물은 이 이미지(Debian 12, glibc 2.36)와 같거나 더 새로운 Linux에서 실행됩니다. 빌드 끝에서 설치본
self-check가 자동으로 돌아갑니다.

## 6. Mac과 다른 점
- 학습 장치: Mac은 MPS, 컨테이너는 CUDA(`dev`) 또는 CPU(`dev-cpu`)입니다. Train 탭의 "자동"이 알아서 고릅니다.
  CUDA에서는 batch "자동"이 Ultralytics AutoBatch로 GPU 메모리를 재서 정합니다.
- CoreML 내보내기는 macOS에서만 됩니다(Export 탭에서 꺼짐). ONNX 내보내기는 그대로 됩니다.
- 설정(창 배치 등)은 컨테이너 안 `~/.config` 아래에 저장되어, `run --rm`으로 띄운 컨테이너가 끝나면 초기화됩니다.
  프로젝트 데이터(`/projects`)와 캐시 볼륨은 남습니다.
- CI(GitHub Actions)의 `docker` 작업이 이 컨테이너(CPU)를 실제로 빌드하고, 그 안에서 전체 테스트를 실행합니다.
