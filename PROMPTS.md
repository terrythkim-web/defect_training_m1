# DefectLab 구현 프롬프트 (20단계)

각 단계는 Claude Code에 그대로 붙여 넣어 쓰는 프롬프트다. 모두 `PRD.md`를 기준으로 하며, 단계마다 테스트를 통과한 뒤 다음 단계로 넘어간다.

**모든 단계 공통**: 앱은 Linux와 Windows 10/11에서 같은 기능으로 동작해야 한다. PRD 5.1(크로스플랫폼 요구사항)을 지키고, 경로는 `pathlib`, 이미지 IO는 한글 경로 대응 함수, 텍스트 파일은 UTF-8을 쓴다. OS 의존 코드가 필요하면 `defect_inspection/platform_utils.py` 한 곳에 모은다.

## Phase A — 기반 (M1)

1. **프로젝트 골격**
   `PRD.md`를 읽고 6장 소스 구조대로 `defect_inspection` 패키지 골격을 만들어줘. `pyproject.toml`에 PySide6, ultralytics, albumentations, opencv-python, tifffile, pyqtgraph, onnx, onnxruntime, jinja2, matplotlib, pydantic, pyyaml, pytest를 넣고, `python -m defect_inspection.app`으로 빈 메인 창이 뜨게 해줘. 가상환경 생성 방법과 실행 방법을 README에 적어줘. Linux와 Windows에서 모두 설치·실행되도록 README에 OS별 설치 명령(PyTorch CUDA/CPU 포함)을 적고, `platform_utils.py`(한글 경로 대응 `imread_unicode`/`imwrite_unicode`, OS별 설정 경로, 폰트 선택)를 만들어줘. GitHub Actions에 `ubuntu-latest`·`windows-latest` 매트릭스로 pytest(QT_QPA_PLATFORM=offscreen)를 돌리는 워크플로를 추가해줘.

2. **프로젝트 모델**
   PRD F1을 기준으로 `core/project.py`를 구현해줘. pydantic으로 `project.yaml`(클래스 목록·색상, 이미지 목록, Train/Val 분할 비율 기본 8:2, 수동 지정, OK 이미지 플래그, 태스크 모드 detect/seg, 색 모드 color/grayscale)을 읽고 저장한다. 그레이스케일 모드(F1.6)면 가져오는 이미지를 비트 깊이를 유지한 채 1채널로 변환하고, 기존 프로젝트를 그레이스케일로 바꾸는 변환 기능(전후 용량 반환)도 넣는다. 분할은 seed 고정으로 재현 가능하게 하고 pytest를 작성해줘.

3. **라벨 데이터 구조**
   PRD F2.6~F2.7을 기준으로 `core/annotation.py`를 구현해줘. BBox/Polygon을 원본 픽셀 좌표로 담는 구조, 이미지별 JSON 저장/로드, YOLO detect txt와 seg txt 변환(Polygon→외접 BBox, BBox→4점 polygon)을 만든다. 왕복 변환 테스트를 포함해줘.

4. **메인 화면 레이아웃**
   PRD 3장 화면 구성대로 `ui/main_window.py`를 만들어줘. 툴바(프로젝트 열기, 폴더 추가, 저장, BBox, Polygon, Crop ROI, 선택, 지우개, Undo/Redo), 좌측 이미지 리스트와 클래스 목록, 가운데 캔버스, 우측 탭 5개(Augment/Train/Result/Compare/Export, 지금은 빈 패널), 상태바. QSplitter 비율은 QSettings에 저장한다.

5. **이미지 로드와 캔버스**
   PRD F1.2, F1.3, F2.3을 구현해줘. PNG/JPEG(.jpg/.jpeg)/BMP/TIFF(.tif/.tiff, 8/16bit, 그레이스케일)를 확장자 대소문자 구분 없이 로드하고(`platform_utils.is_supported_image` 사용) 16bit는 표시용으로만 정규화한다. QGraphicsView 캔버스에 휠 줌, 스페이스+드래그 팬, Fit, 픽셀 좌표 표시, 밝기·대비 임시 조정을 넣는다. 썸네일 리스트에 라벨 완료 상태와 필터, A/D 이미지 이동을 넣는다. 4K 이미지에서 100ms 이내 응답을 확인해줘. 이미지 로드는 반드시 `platform_utils`의 한글 경로 대응 함수를 쓰고, 공백·한글이 들어간 경로로 테스트해줘. 그레이스케일 프로젝트의 1채널 이미지도 정상 표시되게 해줘.

6. **BBox / Polygon 마킹 도구**
   PRD F2.1, F2.2, F2.5를 구현해줘. BBox 드래그, Polygon 클릭(더블클릭 종료), 선택·이동·꼭짓점 편집, 삭제, 숫자키 1~9 클래스 지정, B/P/V 도구 단축키. 라벨 색은 클래스 색을 따르고 좌측 라벨 목록과 선택이 동기화되게 해줘. 자동 라벨 제안(PRD F2.8)도 표시한다: `status: suggested`인 도형은 점선·반투명으로 출처와 score를 함께 그리고, 선택 후 Enter로 수락, Backspace로 거절, 수정·클래스 변경 후 수락, "score ≥ X 모두 수락", "이 이미지 제안 모두 거절"을 지원해줘(`ImageAnnotations.accept/reject/accept_above` 사용).

7. **Undo/Redo, 자동저장, 클래스 관리, SAM 보조**
   PRD F2.2, F2.4를 구현해줘. 라벨 편집의 Undo/Redo(Ctrl+Z/Y)는 6단계에서 `LabelSession` + 공용 QUndoStack으로 이미 들어가 있으니(이미지별 이력) 빠진 동작이 없는지만 확인하고, 이미지 전환과 일정 주기마다 JSON과 YOLO txt를 자동 저장한다. 클래스 추가·이름변경·색상변경 UI를 만든다. 여기까지 M1 완료 기준으로 수동 확인 체크리스트를 정리해줘. SAM 클릭 보조 도구(단축키 S, PRD F2.8 ②)도 구현해줘: `core/sam_assist.py`에서 Ultralytics `SAM`(기본 mobile_sam.pt, 옵션 sam2.1_b.pt, 로컬 경로 지정 가능)으로 포함/제외 점 또는 박스 프롬프트를 받아 마스크를 만들고, 마스크를 단순화한 폴리곤(seg) 또는 외접 박스(detect)로 바꿔 `source="sam"` 제안으로 추가한다. SAM 추론은 UI가 멈추지 않게 백그라운드 스레드에서 돌리고, 마스크→폴리곤 변환 테스트를 작성해줘. 수락/거절도 Undo 대상에 넣고, 이미지 리스트에 "제안 대기" 상태와 필터를 추가해줘.

## Phase B — 증강과 데이터셋 (M2)

8. **증강 파이프라인**
   PRD F3.1, F3.3, F3.4, F3.6을 기준으로 `core/augment.py`를 구현해줘. 설정(dict/pydantic)에서 Albumentations Compose를 만들고 bbox와 polygon(keypoints)을 함께 변환한다. 프리셋(약함/보통/강함/금속 표면/PCB)과 사용자 프리셋 저장, 최소 면적·가시비율 미만 라벨 제거를 넣는다. 변환 후 좌표 정합성 테스트를 작성해줘. 그레이스케일 프로젝트에서는 색조/채도 같은 색 변환을 끄고 1채널 이미지가 그대로 1채널로 나오게 해줘.

9. **Full 데이터셋 빌드**
   PRD F3.2, F1.5, F7.5를 기준으로 `dataset/full/` 빌드를 구현해줘. 원본 단위로 Train/Val을 먼저 나누고 Train만 N배(기본 20) 증강한다. OK 이미지는 빈 라벨로 넣고, data.yaml을 생성한다. Copy-Paste(defect 패치를 정상 이미지에 합성) 옵션도 넣어줘. data.yaml에는 프로젝트의 `channels`(그레이스케일이면 1)를 넣어줘.

10. **Augment 탭 UI**
    PRD F3.3~F3.5를 기준으로 Augment 탭을 만들어줘. 변환별 체크박스·강도 슬라이더·확률, 프리셋 선택/저장, 선택 이미지의 증강 샘플 9개를 라벨 오버레이와 함께 보여주는 미리보기, 데이터셋 빌드 버튼(QThread, 진행바)을 넣는다.

## Phase C — 학습과 결과 (M3, M4)

11. **학습 워커**
    PRD F4.1~F4.4, F4.7을 기준으로 `core/trainer.py`를 구현해줘. Ultralytics YOLO11 학습을 별도 프로세스에서 실행하고, on_train_epoch_end/on_fit_epoch_end 콜백에서 epoch·loss·metrics를 stdout JSON line으로 보낸다. 중지, resume, run별 설정·결과를 `runs/<run_id>/`에 남긴다. 워커가 죽어도 UI는 살아 있어야 한다. 워커는 QProcess로 `sys.executable -m defect_inspection.worker`를 실행하고, Windows spawn 방식에 맞게 `__main__` 가드를 둔다. 중지는 graceful stop 후 시간 초과 시 kill로 처리하고, Windows에서는 DataLoader workers 기본값을 낮게 잡아줘. 모델은 반드시 `core/model_zoo.make_model(key, task, nc, channels, run_dir, weights_dir)`로 만들고(yolo11n/s/m, att-yolo-n/s, PRD F4.8), ATT-YOLO면 반환된 TransferReport(전이된 층)를 run 로그에 남겨줘.

12. **Train 탭 UI**
    PRD F4.1~F4.6을 기준으로 Train 탭을 만들어줘. 모델(yolo11n/s/m, detect/seg), epochs, imgsz, batch, lr0, patience, device(자동감지), online augmentation 토글(오프라인 증강과 중복 시 경고), 데이터셋 선택(full/crop) 폼. QProcess로 워커를 실행하고 진행바·ETA·로그, pyqtgraph로 loss/Precision/Recall/mAP50/mAP50-95를 실시간으로 그린다. run 이력 목록도 넣어줘. 그레이스케일 프로젝트는 1채널 입력 모델로 학습되는지 확인하고, 사전학습 가중치 전이 결과를 로그로 남겨줘. 모델 콤보는 `model_zoo.MODEL_CHOICES`에서 채우고, 모델을 바꾸면 기본 imgsz·epochs를 그 모델 기본값으로 바꾸고 ATT-YOLO 안내 문구(`note`)를 보여줘. "논문 하이퍼파라미터"(`PAPER_HYP`) 체크박스도 넣어줘.

13. **Result 탭**
    PRD F5를 구현해줘. 선택한 run의 Val 예측 결과 그리드(GT와 Pred 색 구분), Confusion matrix와 PR curve 이미지, 드래그&드롭 즉시 추론과 conf/IoU 슬라이더, 오검/미검 목록 클릭 시 좌측 캔버스로 해당 이미지를 열어 라벨 수정으로 이동한다. "자동 라벨 제안 생성"(PRD F2.8 ①, F5.5)도 넣어줘: 선택한 run의 모델로 미라벨 이미지에 예측해 `source="model"` 제안(score 포함)으로 라벨 JSON에 추가하고, 모델이 없을 때는 확정 라벨만으로 yolo11n을 짧게 학습하는 "제안용 빠른 학습" 버튼을 제공해줘.

## Phase D — ONNX Export (M5)

14. **Exporter 코어**
    PRD F6.1, F6.2, F6.4를 기준으로 `core/exporter.py`를 구현해줘. best.pt를 ONNX로 export(imgsz, opset, dynamic, simplify, FP16, NMS 옵션)하고, onnxruntime으로 같은 이미지를 추론해 PyTorch 결과와 박스 IoU 0.99 이상·클래스 일치를 검사한다. metadata.json과 classes.txt를 만든다. 파일명 생성 함수 `default_export_name(pattern, project, mode, model, now)`를 분리하고 테스트를 작성해줘. ONNX 입력 채널 수를 프로젝트 color_mode에 맞추고 metadata에 기록해줘. ATT-YOLO(6개 헤드, attention 포함)도 ONNX export와 onnxruntime 검증이 되는지 확인하고, attention 때문에 입력 크기는 학습 imgsz로 고정해줘(dynamic 끔).

15. **Export 탭과 파일명 규칙**
    PRD F6.3을 구현해줘. 저장 대화상자에 기본 이름 `<프로젝트명>_<모드>_<모델>_<YYYYMMDD_HHMMSS>.onnx`(export 시점)를 미리 채우고 이름과 폴더를 바꿀 수 있게 한다. 금지 문자(`\ / : * ? " < > |`) 입력 차단, `.onnx` 자동 부여, 덮어쓰기 확인, metadata/classes 파일도 같은 이름을 쓴다. 설정에서 `{project}{mode}{model}{date}{time}` 패턴을 바꿀 수 있게 하고, 검증 결과를 탭에 표시해줘. Windows 예약어(CON, PRN, AUX, NUL, COM1 등)와 260자 경로 길이도 검사해줘.

## Phase E — Crop 학습 (M6)

16. **Crop ROI 도구**
    PRD F7.1을 구현해줘. `ui/canvas/roi_tool.py`에 단축키 C로 쓰는 ROI 사각형 도구를 만든다. 이미지별 다수 ROI, 이동/크기 조절, 고정 크기(320/640px)·정사각 옵션, "모든 이미지에 같은 ROI 적용", 라벨과 분리된 점선 레이어, `crops.json` 저장을 넣어줘.

17. **Crop 데이터셋 빌드**
    PRD F7.2~F7.5를 기준으로 `core/crop.py`를 구현해줘. 수동 ROI crop, defect 중심 crop(위치 jitter), 배경 crop(기본 1:1), 슬라이딩 타일(overlap 기본 20%)을 지원한다. 원본→crop 좌표 변환, 경계 라벨 clip, 가시비율 40% 미만 제외를 넣는다. 원본 단위 Train/Val 분할 후 crop하고 Train만 증강해 `dataset/crop/`과 data.yaml을 만든다. Augment 탭에 crop 설정 UI를 추가하고 좌표 왕복·clip 테스트를 작성해줘. 슬라이딩 타일 overlap은 %와 px 둘 다 받게 하고, "ATT-YOLO (1280 타일, overlap 32px)" 프리셋을 넣어줘.

18. **타일링 추론**
    PRD F7.6, F6.4를 구현해줘. Crop 모델로 전체 이미지를 타일링 추론하고 결과를 원본 좌표로 복원한 뒤 타일 간 NMS로 병합한다. PyTorch와 ONNX 모델 모두 지원한다. Result 탭 테스트 추론에서 crop 모델이면 자동으로 타일링을 쓰고, crop 모델 export metadata에 타일 크기와 overlap을 넣는다. 좌표 복원 테스트를 작성해줘.

## Phase F — 비교 리포트와 마무리 (M7)

19. **Full vs Crop 비교 실험**
    PRD F8.1~F8.4를 기준으로 `core/compare.py`와 Compare 탭을 구현해줘. 버튼 하나로 같은 모델·하이퍼파라미터·seed로 full과 crop을 순차 학습하거나, 기존 run 2개 이상을 골라 비교한다. 두 모델을 같은 원본 전체 이미지 Val 셋에 같은 conf/IoU로 다시 평가해(crop은 타일링) Precision, Recall, F1, mAP50, mAP50-95(클래스별 포함), 크기별 Recall, OK/NG 판정 정확도, 미검/과검 수, GPU/CPU 추론 시간, 학습 시간, ONNX 크기를 계산한다. loss/mAP·PR 곡선 겹침, 지표 막대, 두 모델 예측 나란히 보기를 넣어줘.

20. **HTML 리포트, 통합 테스트, 패키징**
    PRD F8.5, F8.6, 10장을 구현해줘. `core/report.py`에서 Jinja2 + matplotlib으로 그래프·이미지를 base64로 넣은 HTML 단일 파일을 `reports/<YYYYMMDD_HHMM>_full_vs_crop.html`에 저장한다. 구성은 요약(추천 방식 + 근거, 기본 mAP50-95 우선·1%p 이내면 빠른 쪽, 우선 지표 변경 가능), 실험 조건, 지표 표, 그래프, 미검/과검 사례, 메모다. 리포트 이력(열기/삭제/같은 설정 재실험)을 Compare 탭에 넣는다. 합성 defect 5장으로 full/crop 각 epochs=3 학습 → ONNX export → HTML 리포트까지 도는 통합 테스트를 만들고, PyInstaller 패키징 스크립트와 README 사용 가이드를 정리해줘. 리포트 그래프의 한글 폰트는 OS별로 자동 선택한다. PyInstaller 빌드는 Linux와 Windows 각각의 spec/스크립트로 만들고, 두 OS에서 설치본이 실행되는지 확인하는 체크리스트를 README에 넣어줘. macOS 설치본은 22단계에서 만든다.

## Phase G — macOS (Apple Silicon) (M8)

21. **macOS 지원**
    PRD 5.2를 기준으로 macOS(Apple Silicon)를 3번째 OS로 추가해줘. `platform_utils`에 macOS 분기(한글 폰트 Apple SD Gothic Neo, 기본 DataLoader workers, MPS 감지)를 넣고, "자동" 장치 선택이 Mac에서 `mps`가 되도록 `core/trainer.train_args`, `core/evaluate.Predictor`, `core/exporter` 검증, SAM 보조의 장치 선택을 한곳에서 정하게 바꿔줘(Ultralytics는 지정하지 않으면 MPS를 쓰지 않음). 학습 워커 환경에 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 넣는다. CI에 `macos-14` 작업을 추가하고(CPU 기능 테스트, GitHub 러너는 MPS 없음), README에 Mac 설치 방법을, `docs/mac_checklist.md`에 실제 M1에서 확인할 항목(설치, 라벨링·SAM, MPS 학습, 결과·export, 단축키 ⌘)을 정리해줘.

22. **Mac 패키징과 가속 측정**
    PyInstaller로 `.app`과 `.dmg`를 만드는 빌드 스크립트를 추가하고(서명·공증이 없으면 Gatekeeper 경고 안내), Export 탭에 macOS 전용 CoreML(.mlpackage) 내보내기 옵션을 넣어줘. `tools/bench_apple.py`를 만들어 같은 데이터로 CPU vs MPS 학습 epoch 시간, CPU / ONNX Runtime / CoreML 추론 시간, 그리고 YOLO11n 학습 1 step(forward+backward)을 MLX로 구현한 마이크로벤치를 MPS와 비교해 표로 출력한다. MLX 마이크로벤치의 수치(출력)는 Linux CPU MLX에서 PyTorch와 일치하는지 테스트하고, 사용자가 M1에서 실행한 결과로 MLX 학습 엔진 진행 여부(기준: MPS 대비 1.5배 이상)를 결정한다.
