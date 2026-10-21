# PRD — Defect Inspection Trainer (가칭: DefectLab)

## 0. Context
소량의 defect 예시 이미지만으로 검사용 딥러닝 모델을 빠르게 만들고 싶다. 현재는 라벨링 툴, augmentation 스크립트, YOLO 학습, ONNX 변환이 각각 분리되어 있어 반복 작업이 번거롭다.
본 제품은 **이미지 로드 → 마킹(라벨링) → Augmentation → YOLO11 학습 → ONNX Export**를 하나의 데스크톱 앱(PySide6)에서 처리한다.
화면은 **좌측: 이미지/마킹 영역**, **우측: 학습 설정·진행·결과 영역**으로 구성한다.
작업 디렉터리 `/home/terry/defect inspection`은 현재 비어 있음 → 신규 프로젝트.

확정 사항: GUI = PySide6 데스크톱 / 라벨 = BBox + Polygon / 엔진 = Ultralytics YOLO11

---

## 1. 목표 / 비목표
**목표**
- 비전문가도 30분 이내에 "이미지 10~50장 → 동작하는 ONNX 모델"을 얻는다.
- 소량 데이터를 augmentation으로 보강하여 실사용 가능한 수준의 mAP 확보.
- 결과 ONNX를 생산 라인 추론 SW(OpenCV DNN / ONNX Runtime)에 바로 투입.

**비목표 (v1)**
- 다중 사용자/서버 협업, 클라우드 학습
- 분류(classification)·이상탐지(anomaly, 비지도) 모델
- 실시간 카메라 연동 검사 (v2 후보)

## 2. 사용자 / 시나리오
- **주 사용자**: 제조 공정 품질/비전 엔지니어 (Python 비숙련 가능)
- **시나리오**: 스크래치·찍힘·이물 예시 이미지 20장 확보 → 앱에서 폴더 열기 → 각 defect를 BBox/Polygon으로 마킹하고 클래스 지정 → augmentation 프리셋 선택 → 학습 시작 → 우측에서 loss/mAP 그래프와 예측 결과 확인 → ONNX export → 라인 PC에 배포.

## 3. 화면 구성 (UI)
```
┌──────────────────────────────── 메뉴/툴바 ─────────────────────────────────┐
│ [프로젝트 열기][이미지 폴더 추가][저장]  [BBox][Polygon][Crop ROI][선택][지우개] [Undo/Redo]│
├───────────────┬───────────────────────────┬────────────────────────────────┤
│ 이미지 리스트  │   좌측: 캔버스 (마킹)       │   우측: 학습 패널 (탭)           │
│ (썸네일, 라벨  │   - 줌/팬, 그리드, 밝기      │   [Augment][Train][Result][Compare][Export]│
│  완료 여부)    │   - BBox/Polygon 그리기      │   - 설정 폼                      │
│               │   - 라벨 목록/클래스 지정    │   - 진행바, 로그                  │
│ 클래스 목록    │                            │   - 그래프(loss, mAP)             │
│ (색상, 단축키) │                            │   - 예측 결과 이미지 그리드        │
└───────────────┴───────────────────────────┴────────────────────────────────┘
│ 상태바: 이미지 n/N · 라벨 수 · 좌표 · GPU 상태                                 │
```
- QSplitter로 좌/우 비율 조절 가능, 레이아웃 저장.

## 4. 기능 요구사항

### F1. 프로젝트 / 이미지 관리
- F1.1 프로젝트 생성/열기 (폴더 단위, `project.yaml`)
- F1.2 이미지 파일/폴더 불러오기: PNG, JPEG(`.jpg`/`.jpeg`), BMP, TIFF(`.tif`/`.tiff`, 8/16bit, 그레이스케일 포함). 확장자는 대소문자 구분 없음(`.JPG`, `.JPEG` 포함)
- F1.3 이미지 리스트: 썸네일, 라벨 완료 상태, 필터(미라벨/라벨됨)
- F1.4 Train/Val 분할 비율 설정(기본 8:2), 수동 지정 가능
- F1.5 "정상(OK) 이미지" 지정 → 배경(negative) 샘플로 학습에 포함
- F1.6 **그레이스케일 모드** (`project.yaml`의 `color_mode: color | grayscale`, 기본 color): 컬러 이미지라도 1채널 그레이로 바꿔 데이터 양을 줄인다
  - 프로젝트 생성 시 선택, 또는 기존 프로젝트를 "그레이스케일로 변환"(프로젝트 안 `images/` 복사본만 변환, 원본 파일은 그대로). 변환 전후 용량을 표시
  - 그레이스케일 프로젝트는 이미지를 가져올 때 자동 변환. 비트 깊이 유지(16bit TIFF는 16bit 그레이)
  - 용량 효과: PNG/BMP/TIFF는 약 1/3로 줄어듦. JPEG는 원래 색 정보를 많이 압축하므로 줄어드는 폭이 작음(약 10%), 재인코딩은 품질 95
  - 이후 단계 전체가 1채널 기준: 데이터셋(`data.yaml`의 `channels: 1`), 증강, 학습 모델 입력, ONNX 입력(1×1×H×W)

### F2. 마킹(라벨링) 툴 — 좌측
- F2.1 도구: BBox(드래그), Polygon(클릭 포인트, 더블클릭 종료), 선택/이동/꼭짓점 편집, 삭제
- F2.2 클래스 관리: 추가/이름변경/색상/숫자키 단축키(1~9)
- F2.3 줌(휠)/팬(스페이스+드래그)/Fit, 픽셀 좌표 표시, 밝기·대비 임시 조정(라벨에 영향 없음)
- F2.4 Undo/Redo, 자동저장
- F2.5 단축키: B/P/V/C/S(도구, S = SAM 보조), A/D(이전/다음 이미지), Del, Ctrl+S, Ctrl+Z/Y
- F2.6 저장 포맷: 내부 JSON(원본 정밀 좌표) + YOLO txt(detect: bbox / seg: polygon) 자동 생성
- F2.7 태스크 모드: **Detect**(Polygon은 외접 BBox로 변환) / **Segment**(BBox는 4점 polygon으로 변환)
- F2.8 **자동 라벨 제안 + 사용자 선택**: 제안은 확정 라벨과 따로 표시되고, 사용자가 수락한 것만 라벨이 된다
  - **제안 방식 ① 학습된 모델**: 선택한 run의 모델로 미라벨(또는 선택한) 이미지에 예측을 실행해 제안으로 표시. confidence 임계값 슬라이더(기본 0.25)
    - 처음에는 모델이 없으므로 "제안용 빠른 학습" 버튼 제공: 지금까지 확정된 라벨로 yolo11n을 짧게 학습(예: 30 epoch) 후 바로 제안 생성
    - 라벨을 수락할수록 재학습 → 제안 품질 향상 (반복 루프)
  - **제안 방식 ② SAM 클릭 보조** (단축키 S): defect를 클릭(좌클릭 = 포함, 우클릭 = 제외 점)하거나 대충 박스를 그리면 SAM이 마스크를 만들고, 이를 Polygon(seg) 또는 외접 BBox(detect)로 제안. 학습 없이 첫 이미지부터 사용 가능
    - 모델: 기본 `mobile_sam.pt`(가볍고 CPU에서도 빠름), 정밀 옵션 `sam2.1_b.pt`. 가중치 자동 다운로드 또는 로컬 경로 지정(오프라인 라인 PC 대응)
    - 마스크 → 폴리곤 변환 시 꼭짓점 단순화(허용 오차 설정), 작은 조각 제거
  - **표시**: 제안은 점선 + 반투명, 출처(모델/SAM)와 score 표시. 확정 라벨은 실선
  - **선택 동작**: 하나씩 수락(Enter)/거절(Backspace), 수정 후 수락, 클래스 변경 후 수락, "score ≥ X 모두 수락", "이 이미지 제안 모두 거절". 모든 동작은 Undo 가능
  - **저장**: 제안은 라벨 JSON에 `status: suggested`(+ `source`, `score`)로 저장되어 앱을 다시 열어도 유지. YOLO txt와 학습에는 수락된 라벨만 사용
  - **이미지 리스트**: "제안 대기" 상태 표시와 필터 (F1.3)

### F3. Augmentation
- F3.1 라이브러리: **Albumentations** (bbox/polygon 좌표 동기 변환)
- F3.2 오프라인 증강: Train 원본 1장마다 원본 + 증강본 N장(기본 20) 생성 → `dataset/full/`에 저장. Val 셋은 증강하지 않음(누수 방지)
  - 포함 대상: 확정 라벨이 있는 이미지 + OK 이미지(빈 라벨 = 배경). **미라벨 이미지와 제안만 있는 이미지는 제외**(표시 안 된 결함을 배경으로 학습하게 되므로). 제외 목록은 빌드 결과에 표시
  - Train/Val 분할은 결함 이미지와 OK 이미지를 각각 나눔(Val에 항상 실제 결함이 들어가게). 라벨된 결함 이미지가 2장 이상 필요
  - 학습용 픽셀: 8bit, 프로젝트 채널 수(그레이 1 / 컬러 3). **16bit는 값 >> 8로 변환**(모든 이미지에 같은 기준이라 이미지 간 밝기 비율 유지, 화면 표시용 늘이기와 다름)
  - 원본이 변환 없이 쓸 수 있는 PNG/JPEG면 그대로 복사(재인코딩 없음). 증강본은 원본이 JPEG면 JPEG(품질 95), 그 외 PNG
  - 증강 설정은 `project.yaml`(`augment`)에 저장되고, 빌드마다 `build_info.yaml`에 설정·seed·개수 기록. 같은 설정이면 같은 데이터셋
  - 임시 폴더에 만든 뒤 교체하므로 취소·오류 시 이전 데이터셋이 그대로 남음
- F3.3 지원 변환 (체크박스 + 강도 슬라이더 + 확률):
  - 기하: 수평/수직 Flip, 회전(90°/임의), 스케일, 이동, 원근(Perspective)
  - 광학: 밝기/대비, 감마, 색조/채도(그레이스케일 프로젝트에서는 비활성), CLAHE
  - 노이즈/블러: Gaussian noise, Motion blur, Gaussian blur, JPEG 압축
  - 환경: 조명 불균일(그라디언트 그림자), 반사광(Specular), 초점 이탈
  - 고급(옵션): Copy-Paste — Train 결함 이미지의 결함(라벨 모양대로, 가장자리 부드럽게)을 Train OK 이미지의 기존 라벨과 겹치지 않는 위치에 합성하고 라벨도 추가. Val 이미지의 픽셀은 쓰지 않음
- F3.4 프리셋: "약함/보통/강함", "금속 표면", "PCB" 등, 사용자 프리셋 저장
- F3.5 **미리보기**: 선택 이미지에 증강 9개 샘플 그리드를 라벨 오버레이와 함께 표시
- F3.6 증강 후 잘려나가거나 너무 작아진 라벨 처리 규칙(최소 면적/가시비율 임계값). 기본: 보이는 비율 40% 미만 또는 16px² 미만이면 제거, 나머지는 이미지 경계에서 잘라냄
  - BBox 라벨은 임의 회전·원근 후 회전된 모서리를 감싸는 박스가 되어 헐거워짐. Polygon 라벨은 정확히 따라가므로 detect 프로젝트라도 회전 증강을 많이 쓸 때는 Polygon으로 라벨링 권장
  - 회전·원근으로 생긴 빈 영역은 검은색으로 채움(반사 채우기는 라벨 없는 가짜 결함을 복제할 수 있어 사용하지 않음)

### F4. 학습 (YOLO11) — 우측
- F4.1 모델 선택: yolo11n/s/m 또는 ATT-YOLO n/s (detect 또는 seg, F4.8), 사전학습 가중치 자동 다운로드/로컬 경로 지정. 그레이스케일 프로젝트는 1채널 입력 모델로 학습. 기본 모델은 yolo11s
- F4.2 하이퍼파라미터: epochs, imgsz(기본 640), batch(auto), lr0, patience(early stop), device(GPU/CPU 자동감지)
- F4.3 Ultralytics 내장 online augmentation on/off (오프라인 증강과 중복 시 경고)
- F4.4 학습은 **별도 프로세스(QProcess/multiprocessing)** 에서 실행 → UI 프리징 방지, 중지/재개(resume) 지원
  - 중지 = 일시정지: 현재 배치가 끝나면 멈추고 마지막으로 끝난 epoch의 `last.pt`로 재개 가능(Ultralytics의 정상 종료 처리는 last.pt의 optimizer를 지워 재개를 막으므로 그 단계를 건너뜀). 20초 안에 멈추지 않으면 강제 종료
  - 워커가 비정상 종료돼도 UI는 유지되고 run 상태에 "failed"와 종료 코드가 기록됨
  - run 폴더 `runs/<날짜_시간_모델>/`: config.yaml(설정 전체), status.json(상태·진행 epoch·지표), events.jsonl(epoch별 loss·지표, 학습 후 best.pt 최종 지표), train/(가중치·그래프)
- F4.5 진행 표시: epoch 진행바, ETA, 실시간 로그
- F4.6 실시간 그래프: train/val box·seg·cls loss, Precision, Recall, mAP50, mAP50-95 (pyqtgraph)
- F4.7 학습 이력: run별 설정·결과 보관, 비교 가능
- F4.8 **ATT-YOLO** (Wang et al., *Scientific Reports* 13:7062, 2023)를 Ultralytics용으로 재구성한 모델 선택지 (`core/att_yolo.py`)
  - 논문 구조 유지: YOLOv5s 기반, P2~P7 6스케일 백본·피처 피라미드와 6개 검출 헤드(stride 4~128), P7 뒤에 self-attention 1개(Conv+SPP+C3 다음 C3TR = multi-head attention + MLP)
  - 다른 점: 헤드가 anchor-free(Detect/Segment)라 논문의 GA auto-anchor는 필요 없음(논문이 해결한 anchor BPR 문제가 생기지 않음). seg도 지원
  - 사전학습: 백본 P2~P5가 YOLOv5와 층 단위로 같아 `yolov5su.pt`/`yolov5nu.pt` 가중치를 그 부분에만 전이, P6·P7·attention·neck·head는 새로 학습 (논문은 1만 4천 개 결함으로 처음부터 학습)
  - 크기(측정): s 22.4M params(논문 23.1M), 33.5 GFLOPs@640(논문 21.8G — Ultralytics의 분리형 헤드가 stride 4 레벨에서 더 무거움), n 6.1M / 11.6G
  - 기본값: imgsz 1280, epochs 300. "논문 하이퍼파라미터" 프리셋(lr0 0.0032, lrf 0.12, momentum 0.843, weight_decay 0.00036, mosaic 1.0, mixup 0.243) 선택 가능
  - 선택 시 안내: 이미지가 적으면 사전학습된 yolo11이 더 나을 수 있으니 Compare 탭(F8.1 "기존 run 2개 선택")으로 비교 권장
  - 논문 코드는 가져오지 않고 논문 본문·Fig. 7·Table 8로 구조만 재구성 (Ultralytics AGPL 범위 그대로)

### F5. 결과 확인 — 우측 Result 탭
- F5.1 Val 이미지 예측 결과 그리드 (정답 파랑 / 예측 주황 + 점수). 낮은 conf로 한 번 예측해 두고 conf 슬라이더는 즉시 다시 거름. 맞음·오검·미검은 같은 클래스 + IoU ≥ 0.5 기준(점수 높은 예측부터 짝지음), 필터: 오검·미검 / 미검 / 오검 / 정상
- F5.2 Confusion matrix, PR curve 표시 (Ultralytics 산출물 활용)
- F5.3 임의 이미지 드래그&드롭 → 즉시 추론 테스트, confidence/IoU 슬라이더
- F5.4 오검/미검 이미지 목록 → 클릭 시 좌측 캔버스에서 라벨 수정으로 이동 (반복 개선 루프)
- F5.5 "자동 라벨 제안 생성" 버튼: 선택한 run으로 미라벨 이미지(확정 라벨 없음, OK 아님)에 제안을 만들고 좌측 캔버스에서 검토 (F2.8 ①). 같은 이미지의 이전 모델 제안은 교체, SAM 제안과 확정 라벨은 유지. "제안용 빠른 학습"은 데이터셋 다시 빌드 → yolo11n 30 epoch → 제안 생성을 이어서 실행

### F6. ONNX Export
- F6.1 best.pt → ONNX (`model.export(format="onnx")`), 옵션: imgsz, opset(기본 12+), dynamic, simplify, FP16, NMS 포함 여부
- F6.2 Export 후 **ONNX Runtime으로 검증 추론** 실행, PyTorch 결과와 일치 여부 표시
  - 검증: Val 이미지 최대 3장을 PyTorch와 onnxruntime으로 같은 전처리·후처리로 추론해, 모든 검출이 같은 클래스·박스 IoU ≥ 0.99로 짝지어지고 개수가 같으면 통과
  - 변환은 CPU에서 수행(모든 모델 가능, onnxruntime-gpu 교체 방지). FP16만 GPU 필요(없으면 FP32로 바꾸고 안내)
  - 입력 크기 기본값 = 학습 imgsz. ATT-YOLO는 attention 때문에 학습 크기로 고정, dynamic 끔
  - Ultralytics의 실행 중 패키지 자동 설치는 끔(오프라인 PC 대응). 필요한 onnxslim은 의존성에 포함
- F6.3 **저장 파일명**: 저장 대화상자에 기본 이름이 미리 채워지고, 사용자가 이름과 저장 폴더를 바꿀 수 있음
  - 기본 이름: `<프로젝트명>_<모드>_<모델>_<YYYYMMDD_HHMMSS>.onnx` (예: `pcb_crop_yolo11s_20261013_143025.onnx`). 날짜·시간은 export 시점 기준
  - 파일명에 쓸 수 없는 문자(`\ / : * ? " < > |`)는 입력 시 막고, `.onnx` 확장자는 자동으로 붙임
  - 같은 이름 파일이 있으면 덮어쓰기 확인
  - 함께 저장되는 metadata 등은 같은 이름을 따름 (예: `<이름>.metadata.json`, `<이름>.classes.txt`)
  - 기본 이름 패턴은 설정에서 변경 가능 (`{project}`, `{mode}`, `{model}`, `{date}`, `{time}` 토큰, 메뉴 [설정 > 내보내기 파일명 패턴])
  - 기본 이름의 프로젝트명 등은 금지 문자 → `_`, 공백 → `_`로 바꿈(사용자가 직접 입력한 이름은 그대로). 이름을 고치지 않았다면 날짜·시간은 [ONNX 내보내기]를 누른 시점으로 다시 계산
  - Windows 예약어(CON, PRN, AUX, NUL, COM1~9, LPT1~9), 공백·마침표로 끝나는 이름, 260자 넘는 경로(metadata 파일 포함 가장 긴 경로 기준)는 내보내기 버튼을 막고 이유 표시
- F6.4 함께 출력: `classes.txt`/`metadata.json`(클래스명, 입력크기, 입력 채널 수(1 또는 3)·색 순서, 전처리 정보, 임계값 권장치). Crop 모델은 crop 크기·ROI·타일링 추론 파라미터(타일 크기, overlap) 포함
  - Crop 모델 metadata `crop`: `tiling`(tile_size, overlap_px, 병합 기준), 타일링 방법 설명(`how`), crop 크기, crop 방식, 공통 ROI 좌표. ONNX 검증은 crop 모델이면 Val crop 이미지로 함
  - 입력 색 순서: 컬러는 **RGB**(OpenCV BGR 이미지는 바꿔서 입력), 그레이는 1채널. 전처리: 16bit는 >> 8, letterbox(비율 유지, 패딩 114, 가운데 정렬), 0~1 정규화, NCHW. 출력 형식(NMS 포함 여부), 학습 지표, 검증 결과도 기록
- F6.5 (옵션) 간단한 Python/C++ 추론 예제 코드 동봉

### F7. Crop 기반 학습 — 좌측 마킹 + 우측 Augment/Train
전체 이미지에서 defect가 작거나 검사 대상 영역이 일부일 때, 이미지를 잘라(crop) 학습할 수 있게 한다.
- F7.1 **수동 ROI Crop 도구** (좌측 캔버스, 단축키 `C`): 샘플 이미지 위에 crop 영역(사각형)을 그린다.
  - 이미지별 다수 ROI 가능, 이동/크기 조절, 고정 크기 옵션(예: 320/640 px), 정사각 고정 옵션
  - "모든 이미지에 같은 ROI 적용" (고정 지그 검사 대응)
  - 라벨과 별도 레이어(점선 표시)로 표시, 라벨 편집에 영향 없음
  - ROI는 2종류: **이미지별 ROI**(보라 점선)와 **공통 ROI**(청록 점-선, 모든 이미지에 같은 위치 적용, 작은 이미지에서는 이미지 안쪽만 사용). 고정 크기는 클릭 한 번으로 그 지점 중심에 생성(이미지 밖으로 나가지 않게 밀어 넣음), 고정 크기 ROI는 크기 조절 불가
  - ROI 편집은 `C` 도구에서만 가능하고 다른 도구에서는 클릭이 라벨로 통과. 추가/이동/크기/삭제/공통 지정은 실행 취소 가능, 변경 즉시 `crops.json`에 저장(그리기 옵션도 프로젝트별 저장)
- F7.2 **자동 Crop 생성** (Augment 탭):
  - Defect 중심 crop: 각 라벨 주변을 crop 크기로 자르고, 위치를 무작위로 이동(jitter)해 defect가 항상 중앙에 오지 않게 함
  - 배경(negative) crop: defect 없는 영역에서 비율 지정(기본 defect crop : 배경 = 1 : 1)
  - 슬라이딩 타일: 타일 크기 + overlap(기본 20%, px 단위 지정도 가능)로 전체 이미지를 분할. 프리셋 "ATT-YOLO (1280 타일, overlap 32px)" — 논문의 5000×5000 원본 처리 방식
  - 설정(Augment 탭 "Crop 데이터셋" 그룹, `project.yaml`의 `crop`): 4가지 방식(수동 ROI / 결함 중심 / 배경 / 슬라이딩 타일)을 각각 켜고 끔. crop 크기(기본 640), 라벨당 crop 수(기본 2), 위치 흔들기(기본 50%), 배경 비율, 타일 크기·overlap(%/px), Train crop당 증강 수(기본 5). 프리셋: 기본(결함 중심 640), ATT-YOLO(1280 타일, overlap 32px), 슬라이딩 타일(640, 20%), 수동 ROI만. 빌드 버튼 옆에서 Full/Crop 선택
  - crop 위치는 프로젝트 seed + 이미지 경로로 정해져 같은 설정이면 다시 빌드해도 같은 crop이 나옴. 이미지보다 큰 crop은 이미지 전체를 사용
- F7.3 Crop 라벨 처리: 원본 좌표 → crop 좌표로 변환, crop 경계에 걸린 라벨은 clip. clip 후 보이는 비율이 임계값(기본 40%) 미만이면 제외 (F3.6 규칙과 동일)
- F7.4 Crop 데이터셋 생성 후 증강(F3) 적용. **Train/Val 분할은 원본 이미지 단위로 먼저 하고 그 다음 crop** → 같은 원본에서 나온 crop이 Train/Val에 섞이지 않게 함(누수 방지)
- F7.5 데이터셋 모드: `full`(전체 이미지) / `crop`(ROI 또는 자동 crop). 각각 `dataset/full/`, `dataset/crop/`으로 따로 생성되며 동시에 보관
- F7.6 Crop 모델 추론: 전체 이미지 입력 시 **타일링 추론 + 결과 좌표 원본 복원 + 타일 간 NMS 병합**을 지원 (Result 탭 테스트·비교 평가에서 사용)
  - 타일 크기 = 학습한 crop 크기, overlap = 데이터셋의 타일 overlap(타일 없이 만든 crop은 20%). 학습 시작 시 run에 기록되어 나중에 Crop 데이터셋을 다시 빌드해도 그 run의 추론 방식은 그대로
  - 병합: 같은 클래스끼리 점수 높은 순으로 IoU ≥ 0.5, 또는 작은 박스 기준 겹침(IoS) ≥ 0.7이면 낮은 쪽 제거 (타일 경계에서 잘린 조각 제거)
  - Result 탭은 crop run이면 자동으로 타일링 추론을 쓰고(검증 그리드는 Val 원본 이미지 기준) 화면에 타일 크기·overlap을 표시. ONNX 파일도 metadata의 타일 정보로 같은 방식 추론 가능

### F8. Full vs Crop 학습 결과 비교 리포트 — 우측 Compare 탭
- F8.1 **비교 실험 실행**: "Full + Crop 비교 학습" 버튼 한 번으로 두 데이터셋을 같은 모델/하이퍼파라미터/seed로 순차 학습. 기존 run 2개 이상을 골라 비교할 수도 있음(예: yolo11s vs ATT-YOLO s, 같은 데이터셋)
- F8.2 **공정 비교 평가**: 각 run 자체 val 지표 외에, **같은 원본 Val 이미지(전체 이미지)** 로 두 모델을 다시 평가
  - Full 모델: 일반 추론 / Crop 모델: 타일링 추론(F7.6)
  - conf/IoU 임계값을 동일하게 맞춤
- F8.3 비교 항목
  - 정확도: Precision, Recall, F1, mAP50, mAP50-95 (전체 + 클래스별)
  - 크기별 성능: defect 크기 구간(small/medium/large)별 Recall
  - 이미지 단위 판정: OK/NG 판정 정확도, 미검(FN)/과검(FP) 이미지 수
  - 속도: 이미지 1장 추론 시간(ms, 타일링 포함), GPU/CPU 각각
  - 학습: 학습 시간, best epoch, 데이터셋 크기(이미지/라벨 수)
  - 모델: ONNX 파일 크기, 입력 크기
  - 평가 기준(구현): 같은 Val 원본 이미지 = 프로젝트의 원본 단위 분할(Full·Crop 데이터셋 공통)의 Val 전체(결함 + OK). 박스 기준(seg도 마스크의 박스로 비교). P/R/F1은 선택한 conf에서 IoU ≥ 0.5·같은 클래스, mAP는 conf 0.001까지의 예측으로 COCO 101점 AP(클래스 평균). 크기 구간은 COCO 기준(작음 < 32², 중간 < 96², 큼). 이미지 판정은 라벨이 있으면 NG, conf 이상 예측이 하나라도 있으면 NG로 판정. 추론 시간은 이미지 5장의 중앙값(첫 장 예열 제외), CPU와 GPU(있을 때) 각각. ONNX 크기는 그 run을 내보낸 파일이 있을 때만(없으면 "내보내기 전"). 각 run 자체 val 지표는 참고로만 표시(데이터셋이 달라 직접 비교 불가)
  - Full + Crop 비교 학습: Full·Crop 데이터셋을 현재 설정으로 다시 빌드 → Train 탭 설정으로 Full 학습 → 같은 설정(데이터셋만 다름)으로 Crop 학습 → 두 run 자동 비교. 단계별 진행 상태 표시, 어느 단계든 실패하면 멈추고 이유 표시
- F8.4 시각화: loss/mAP 곡선 겹쳐 그리기, PR curve 겹쳐 그리기, 지표 막대그래프, 같은 이미지에 두 모델 예측 결과를 나란히 표시(차이 나는 이미지 우선)
- F8.5 **리포트 문서 저장**
  - 형식: **HTML 단일 파일** (그래프·이미지를 base64로 포함, 다른 PC에서 브라우저로 바로 열람 가능)
  - 저장 위치: `reports/<YYYYMMDD_HHMM>_full_vs_crop.html`
  - 구성: ① 요약(추천 방식 + 근거 지표) ② 실험 조건(데이터셋, crop 설정, 증강, 하이퍼파라미터, 환경 GPU/버전) ③ 지표 비교표 ④ 그래프 ⑤ 예측 결과 비교 이미지(미검/과검 사례) ⑥ 결론/메모(사용자 입력란)
  - 추천 규칙: 기본은 mAP50-95 우선, 차이가 1%p 이내이면 추론 속도가 빠른 쪽. 사용자가 우선 지표(예: Recall 우선) 변경 가능
  - 구현: Compare 탭 "리포트"에서 메모를 적고 저장. 같은 분에 두 번 저장하면 `_2`가 붙음. ⑥ 메모 칸은 브라우저에서도 고칠 수 있음(다른 이름으로 저장). 실험 정보(run, 학습 설정, 결과 요약)를 HTML 안에 JSON으로 넣어 이력·재실험에 씀. 지표 표는 Compare 탭과 같은 정의를 씀
- F8.6 비교 리포트 이력: 리포트 목록에서 다시 열기/삭제, run 설정을 그대로 불러와 재실험 가능
  - 재실험: 리포트에 기록된 run들의 학습 설정 그대로, 필요한 데이터셋(Full/Crop)을 현재 설정으로 다시 빌드 → 차례로 학습 → 자동 비교

## 5. 비기능 요구사항
- 플랫폼: **Linux(Ubuntu 22.04+)와 Windows 10/11(64bit) 모두 동일 기능 지원**. 한 코드베이스로 두 OS에서 실행·학습·export가 되어야 한다 (5.1 참고)
- 플랫폼(추가 예정): **macOS 13+ (Apple Silicon M1 이상)** — 같은 코드베이스에 3번째 OS로 추가, 20단계 이후 Phase G(5.2 참고)
- 환경: Python 3.10+, PyTorch(CUDA), NVIDIA GPU 권장, CPU 학습도 동작(느림 경고)
- 성능: 4K 이미지 로드/줌 시 UI 응답 < 100ms, 200장 프로젝트 원활. 이미지 디코딩은 UI 스레드에서 하지 않음(4K PNG 디코딩만 약 150ms): 이웃 이미지는 미리 읽어 두고, 캐시에 없는 이미지는 "불러오는 중" 표시 후 백그라운드에서 읽어 표시. 썸네일은 백그라운드 생성 + 디스크 캐시
- 안정성: 학습 프로세스 크래시가 UI를 종료시키지 않음, 라벨 자동저장
- 라이선스: Ultralytics는 **AGPL-3.0** — 사내 배포/상용 시 Enterprise 라이선스 검토 필요
- 국제화: UI 한국어 기본, 영어 지원 구조(Qt tr)

### 5.1 크로스플랫폼 요구사항 (Linux / Windows)
- 경로: 모든 경로는 `pathlib.Path`로 처리, 하드코딩된 `/`·`\` 구분자 금지. **공백·한글이 포함된 경로**(예: `C:\Users\홍길동\defect inspection`)에서 동작해야 함
- 이미지 IO: Windows에서 `cv2.imread/imwrite`는 한글 경로를 못 읽으므로 `np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `tofile`로 통일
- 텍스트 파일: JSON/YAML/txt는 항상 `encoding="utf-8"`, 줄바꿈 `\n`으로 저장
- 프로세스: 학습 워커는 QProcess로 `sys.executable -m defect_inspection.worker` 실행. Windows의 spawn 방식에 맞춰 모든 진입점에 `if __name__ == "__main__":` 가드. 중지는 graceful stop 신호 후 시간 초과 시 `kill()`
- DataLoader: Windows에서는 `workers` 기본값을 낮게(0~2) 두고 설정에서 변경 가능
- GPU: CUDA 지원 PyTorch는 두 OS 모두 지원. GPU가 없으면 CPU로 자동 전환. onnxruntime은 CUDA EP 우선, 없으면 CPU EP
- 폰트: 리포트 그래프(matplotlib)의 한글 폰트를 OS별로 자동 선택 (Windows: Malgun Gothic, Linux: Noto Sans CJK / NanumGothic, 없으면 경고)
- 설정·캐시 위치: Qt `QStandardPaths`로 OS별 표준 위치 사용 (Linux `~/.config`, Windows `%APPDATA%`)
- 파일명: Windows 금지 문자·예약어(CON, PRN, AUX, NUL, COM1 등) 차단, 경로 길이 260자 초과 시 경고
- 단축키·UI: Ctrl 기반 단축키 동일, HiDPI(Windows 125/150% 배율) 정상 표시
- 배포: PyInstaller로 OS별 설치본을 각각 빌드 (Linux: 폴더/AppImage, Windows: 폴더/.exe 설치본)
  - 구현: `packaging/defectlab_{linux,windows}.spec` + `build_linux.sh` / `build_windows.ps1` (폴더 + tar.gz/zip). Windows는 창 앱 `DefectLab.exe` + 학습용 콘솔 `DefectLab-worker.exe`(창 앱은 표준 출력이 없을 수 있어서, QProcess가 콘솔 창 없이 실행). 설치본 확인은 `--self-check`(라이브러리·모델 YAML·ONNX Runtime·한글 폰트·메인 창)와 README 체크리스트
- CI: GitHub Actions `ubuntu-latest` + `windows-latest` 매트릭스로 pytest 실행 (GPU 없는 CPU 기준, Qt는 offscreen)

### 5.2 macOS 요구사항 (Apple Silicon, Phase G)
별도 앱이 아니라 같은 코드에 macOS를 추가한다. 앱이 쓰는 라이브러리는 Apple Silicon에서 그대로 동작한다.
- 학습 가속: PyTorch **MPS**(Mac GPU). Ultralytics는 장치를 지정하지 않으면 MPS를 고르지 않으므로, 앱의 "자동" 장치 선택이 Mac에서 `mps`를 명시한다. 미지원 연산은 `PYTORCH_ENABLE_MPS_FALLBACK=1`로 CPU 대체(속도 편차 가능). DataLoader workers는 0(Ultralytics가 MPS에서 자동 적용)
- 추론: Result 탭 예측·export 검증·SAM 보조도 MPS 사용. ONNX Runtime CoreML 실행 모드(선택)
- Export 탭: macOS에서 CoreML(.mlpackage) 내보내기 옵션 추가(Neural Engine 활용, macOS 전용)
- 한글 폰트: Apple SD Gothic Neo. 설정·캐시 위치: `QStandardPaths`(~/Library/Application Support, ~/Library/Caches)
- 단축키: Qt가 Ctrl을 ⌘(Command)로 자동 매핑
- 배포: PyInstaller로 `.app`/`.dmg`. 개발자 서명·공증이 없으면 첫 실행 시 Gatekeeper 경고(우클릭 → 열기)
- 테스트: GitHub Actions `macos-14`(Apple Silicon)로 기능 테스트(CPU). GitHub 러너는 MPS(GPU)를 쓸 수 없으므로 속도·화면은 실제 M1에서 확인(`docs/mac_checklist.md`, `tools/bench_apple.py`)
- 구현(21단계): 장치는 `platform_utils.resolve_device(요청, 용도)` 한 곳에서 정한다. "자동" → CUDA면 `0`, Mac이면 `mps`, 없으면 `cpu`. 학습·Result 예측·비교·SAM이 이 함수를 쓰고, ONNX 검증만 PyTorch와 ONNX Runtime을 같은 조건으로 비교하려고 항상 CPU. Train 탭 장치 목록은 Mac에서 "Apple GPU (MPS)", FP16 export는 CUDA 전용. Mac 키보드에는 Delete 키가 없어 ⌫가 (거절할 제안이 없으면) 선택한 라벨·ROI를 삭제. 버튼 툴팁의 단축키도 OS 표기(⌘0)로 표시. 설정 `~/Library/Application Support/DefectLab`, 캐시 `~/Library/Caches/DefectLab`. 실제 M1 확인 항목은 `docs/mac_checklist.md`
- 구현(22단계): `packaging/defectlab_mac.spec` + `build_mac.sh` → `DefectLab.app`(ad-hoc 서명, `DEVELOPER_ID`/`NOTARY_PROFILE`이 있으면 서명·공증) + `.dmg`. Export 탭 형식에 CoreML(.mlpackage, macOS에서만 보임, 컬러 모델만, Core ML로 검증 IoU ≥ 0.9). `tools/bench_apple.py`: 학습 epoch CPU vs MPS, 추론(네트워크만) PyTorch CPU/MPS·ONNX Runtime CPU/CoreML EP·CoreML, 학습 1 step MLX vs PyTorch → 판단 줄 출력. MLX 네트워크(`tools/mlx_yolo.py`)는 PyTorch와 출력·loss·gradient가 일치함을 Linux CPU MLX로 테스트. **MLX 진행 여부는 사용자의 M1 측정 결과로 결정(미정)**
- **MLX 평가 기준**: Ultralytics에는 MLX 지원이 없어 MLX 학습은 모델·loss·학습 루프 재구현 수준의 작업이다. M1에서 MPS 학습 속도를 측정하고, YOLO11n 학습 1 step(forward+backward)을 MLX로 만든 마이크로벤치와 비교해 **MLX가 의미 있게 빠를 때만(예: 1.5배 이상)** MLX 학습 엔진을 별도 단계로 계획한다. MLX는 Linux CPU 빌드도 있어 수치 정합성은 Linux에서도 검증 가능

## 6. 기술 스택 / 아키텍처
| 영역 | 선택 |
|---|---|
| GUI | PySide6 (QGraphicsView 기반 캔버스) |
| 그래프 | pyqtgraph |
| 이미지 IO | OpenCV, tifffile |
| Augmentation | Albumentations |
| 학습/Export | Ultralytics YOLO11 |
| 검증 추론 | onnxruntime(-gpu) |
| 리포트 | Jinja2(HTML 템플릿), matplotlib(리포트용 그래프) |
| 설정 | YAML (pydantic 검증) |

**프로젝트 폴더 구조 (사용자 데이터)**
```
my_project/
  project.yaml          # 클래스, 분할, 증강/학습 설정
  images/               # 원본
  labels/               # 내부 JSON
  crops.json            # 수동 ROI crop 정의, 자동 crop 설정
  dataset/full/         # 전체 이미지 YOLO 데이터셋 (images/{train,val}, labels/{train,val}, data.yaml)
  dataset/crop/         # crop 데이터셋 (동일 구조)
  runs/<run_id>/        # 학습 결과, weights, 그래프
  exports/              # <프로젝트>_<모드>_<모델>_<YYYYMMDD_HHMMSS>.onnx + .metadata.json (이름 변경 가능)
  reports/              # <YYYYMMDD_HHMM>_full_vs_crop.html
```

**소스 구조**
```
defect_inspection/
  app.py                     # 진입점
  worker.py                  # 학습 워커 진입점 (QProcess로 실행)
  platform_utils.py          # OS 의존 코드 (한글 경로 이미지 IO, 설정 경로, 폰트)
  ui/main_window.py          # 좌/우 스플리터, 툴바
  ui/canvas/                 # QGraphicsView, BBox/Polygon item, 도구
  ui/panels/                 # augment_panel, train_panel, result_panel, export_panel
  core/project.py            # 프로젝트/이미지/클래스 모델
  core/annotation.py         # 라벨 데이터 구조, YOLO 변환
  core/augment.py            # Albumentations 파이프라인, 데이터셋 빌드
  core/trainer.py            # 학습 워커 프로세스, 콜백 → 진행 이벤트(JSON) 전송
  core/exporter.py           # ONNX export + onnxruntime 검증
  core/crop.py               # ROI/자동 crop, 좌표 변환, 타일링 추론 + NMS 병합
  core/compare.py            # 공통 Val 셋 재평가, 지표 계산
  core/report.py             # HTML 리포트 생성
  ui/panels/compare_panel.py # 비교 탭
  ui/canvas/roi_tool.py      # Crop ROI 도구
tests/
```
- 학습 워커 ↔ UI 통신: Ultralytics 콜백(on_fit_epoch_end 등)에서 metrics를 stdout JSON line 또는 Queue로 전달 → UI Signal로 그래프 갱신.

## 7. 마일스톤
| 단계 | 내용 | 산출물 |
|---|---|---|
| M1 | 프로젝트/이미지 로드, BBox·Polygon 마킹, SAM 클릭 보조, 제안 수락/거절, YOLO 라벨 저장 | 라벨링 툴 단독 사용 가능 |
| M2 | Augmentation 파이프라인 + 미리보기 + 데이터셋 빌드 | data.yaml 생성 |
| M3 | 학습 워커, 실시간 그래프/로그, 중지/재개 | best.pt |
| M4 | Result 탭(예측 그리드, 테스트 추론, 오검 → 재라벨 루프), 모델 기반 자동 라벨 제안 | 제안 → 수락 → 재학습 루프 |
| M5 | ONNX export + 검증 + metadata | 날짜·시간 이름의 .onnx |
| M6 | Crop ROI 도구, 자동 crop, crop 데이터셋, 타일링 추론 | dataset/crop, crop 모델 |
| M7 | Full vs Crop 비교 학습/평가, Compare 탭, 리포트(HTML), Linux·Windows 각각 PyInstaller 패키징 | 비교 리포트, OS별 설치본 |
| M8 | macOS(Apple Silicon) 지원: MPS 학습·추론, CoreML export, Mac .app/.dmg, 가속 측정과 MLX 진행 여부 결정 | Mac 설치본, 측정 결과 |

## 8. 성공 지표 (수용 기준)
- 샘플 이미지 20장 + 증강으로 학습 시 val mAP50 ≥ 0.8 (단순 defect 기준)
- ONNX vs PyTorch 예측 박스 IoU ≥ 0.99, 클래스 일치
- 버튼 한 번으로 Full/Crop 학습 → 공통 Val 셋 평가 → 리포트 저장까지 사용자 개입 없이 완료
- 리포트 하나만 보고 어느 방식을 쓸지 판단 가능 (요약 + 근거 지표 + 실험 조건)
- 라벨링 → ONNX까지 전체 플로우를 문서 없이 처음 사용자가 완료 가능
- 학습 중 UI 프레임 드랍/프리징 없음

## 9. 리스크 / 오픈 이슈
- 소량 데이터 과적합 → 증강 강도 가이드, Val 분리 강제, early stopping
- 매우 작은 defect(수 픽셀) → 타일링(SAHI 방식) 학습/추론 옵션 검토 (v1.1)
- AGPL 라이선스 이슈 → 배포 형태 결정 필요
- macOS: MPS 미지원 연산의 CPU 대체로 속도 편차, GitHub macOS 러너는 GPU 없음(속도는 실제 Mac 측정), 서명·공증 없는 배포는 보안 경고. MLX는 Ultralytics 미지원이라 도입 시 별도 학습 엔진 유지 부담
- ATT-YOLO는 P2~P7 전체를 쓸 사전학습 가중치가 없어(백본 일부만 전이) 이미지 수십 장 규모에서는 yolo11보다 낮을 수 있음 → 기본 모델은 yolo11s 유지, 비교 리포트로 판단
- ATT-YOLO 1280 입력은 GPU 메모리·CPU 학습 시간이 크게 늘어남(s 기준 137 GFLOPs@1280) → 배치 자동 조정, CPU 경고
- ATT-YOLO의 attention(C3TR)은 ONNX에서 입력 크기가 고정됨(dynamic 미보장) → export 시 학습 imgsz로 고정
- SAM 가중치 다운로드 필요(mobile_sam 약 40MB, sam2.1_b 약 160MB) → 오프라인 PC는 로컬 경로 지정, 설치본에 mobile_sam 포함 검토
- SAM은 자연 이미지로 학습되어 미세한 스크래치·저대비 defect에서는 마스크가 부정확할 수 있음 → 박스 프롬프트·제외 점으로 보정, 결과는 항상 사용자 수락을 거침
- Crop 모델은 전체 이미지에서 타일링 추론이 필요해 속도가 느려질 수 있음 → 리포트에 속도를 함께 표시
- Crop 크기가 너무 작으면 주변 정보가 사라져 과검이 늘 수 있음 → 배경 crop 비율 조절, 크기 실험 권장
- (결정) 16bit 학습 이미지는 값 >> 8로 8bit 변환 (F3.2)
- 1채널 학습 시 사전학습 가중치의 첫 conv 층(3채널)이 그대로 옮겨지지 않을 수 있음 → 11단계에서 확인, 성능이 낮으면 그레이를 3채널로 복제해 3채널 모델로 학습하는 옵션 검토
- Windows 한글 경로·긴 경로·프로세스 종료 방식 차이 → 5.1 규칙 준수, CI에서 한글·공백 경로 테스트
- PyTorch CUDA 버전이 OS별로 설치 방법이 다름 → README에 OS별 설치 명령 명시

## 10. 검증 방법 (구현 후)
- 단위 테스트: annotation ↔ YOLO txt 변환 왕복, 증강 후 bbox/polygon 좌표 정합성(pytest)
- 통합 테스트: 샘플 이미지(합성 defect 5장)로 epochs=3 학습 → ONNX export → onnxruntime 추론 성공 확인
- Crop 테스트: crop↔원본 좌표 변환 왕복, 경계 clip 규칙, 타일링 추론 결과가 원본 좌표에 맞게 복원되는지(pytest)
- 비교 리포트 테스트: 합성 데이터로 Full/Crop 각 epochs=3 학습 → HTML 생성 및 지표 값 일치 확인
- ONNX 파일명 테스트: 기본 이름에 export 시점 날짜·시간 포함, 이름 변경·금지 문자 처리·덮어쓰기 확인 동작
- 크로스플랫폼 테스트: CI에서 Ubuntu·Windows 양쪽 pytest 통과, 공백·한글이 포함된 경로로 이미지 로드·데이터셋 빌드·export·리포트 생성 확인
- 수동 테스트: Linux와 Windows 각각에서 `python -m defect_inspection.app` 실행 → 라벨링·증강 미리보기·학습 그래프·export·Crop 학습·비교 리포트 전체 플로우 확인, PyInstaller 설치본으로도 동일 확인
