# macOS (Apple Silicon) 확인 체크리스트

Phase G(PRD 5.2, PROMPTS 21단계) 확인용이다. GitHub Actions의 `macos-14` 작업은 기능 테스트(CPU)만 하고 Mac GPU(MPS)는 쓰지 못하므로,
**실제 M1 이후 Mac**에서 아래 항목을 확인한다. 테스트 폴더는 공백과 한글이 들어간 경로를 쓴다 (예: `~/Documents/검사 데이터/샘플 이미지`).
확인이 끝나면 각 섹션의 메모 칸에 기록해 공유한다 (특히 5장의 학습 시간).

실행: `python -m defect_inspection.app` (설치는 README "macOS (Apple Silicon)")

## 1. 설치
- [ ] `pip install torch torchvision` 후 `python -c "import torch; print(torch.backends.mps.is_available())"`가 `True`
- [ ] `pip install -e ".[dev]"`가 오류 없이 끝남
- [ ] `python -m defect_inspection.app --self-check` → 9개 모두 `[OK]` (Mac 전용 "Mac (MPS·CoreML)" 포함), "계산 장치: GPU: Apple MPS → … 'mps'", "한글 폰트: Apple SD Gothic Neo"
- [ ] `python -m pytest`가 통과 (실패가 있으면 테스트 이름과 메시지를 기록)
- [ ] 앱 상태바 오른쪽에 "GPU: Apple MPS" 표시

메모 (macOS 버전, 칩, 메모리, Python·torch 버전):

## 2. 화면과 단축키 (⌘)
- [ ] 창·메뉴·탭의 한글이 깨지지 않고 Retina 화면에서 또렷함
- [ ] 메뉴에 단축키가 ⌘로 표시됨 (⌘N 새 프로젝트, ⌘O 열기, ⌘S 저장, ⌘I 이미지 폴더 추가, ⌘Z/⇧⌘Z 실행 취소/다시 실행)
- [ ] ⌘0 화면 맞춤, ⌘1 실제 크기, 버튼 툴팁에도 ⌘0 / ⌘1로 표시
- [ ] 도구 단축키 B / P / C / S / V / E, A/D 이전·다음 이미지, Delete(⌫ 또는 fn+⌫)로 라벨 삭제
- [ ] 트랙패드: 두 손가락 스크롤로 확대/축소, 스페이스+드래그로 이동
- [ ] 창 크기·패널 배치를 바꾸고 다시 열면 유지됨 (창·탭 설정은 `~/Library/Preferences/com.defectlab.DefectLab.plist`, 증강 프리셋은 `~/Library/Application Support/DefectLab/`)

## 3. 라벨링과 SAM 보조
- [ ] 한글·공백 경로에 새 프로젝트, JPEG·PNG·16bit TIFF 폴더 추가, 썸네일 표시 (`~/Library/Caches/DefectLab/thumbnails`)
- [ ] BBox / Polygon / Crop ROI 그리기, 실행 취소, 저장 후 다시 열기
- [ ] SAM 보조(S): 처음 선택 시 `mobile_sam.pt` 다운로드(`~/Library/Caches/DefectLab/weights/`) → 이미지 준비 시간 ____ 초
- [ ] SAM 클릭 → 제안 표시까지 ____ 초 (Linux CPU 기준 약 0.3~0.6 초). 제안 수락 Enter / 거절 ⌫
- [ ] SAM 결과 모양이 Linux/Windows와 비슷함 (MPS에서 마스크가 깨지거나 비어 있지 않음)

## 4. 증강·데이터셋
- [ ] Augment 미리보기 9장이 몇 초 안에 표시
- [ ] Full / Crop 데이터셋 빌드 완료, 진행 표시와 취소 동작

## 5. MPS 학습 (가장 중요)
- [ ] Train 탭 장치 목록: "자동 (GPU: Apple MPS)", "CPU", "Apple GPU (MPS)"
- [ ] 장치 "자동"으로 yolo11n, imgsz 640, 10 epoch 학습이 끝까지 진행 (로그에 `mps` 장치 표시)
- [ ] 같은 데이터로 장치 "CPU" 10 epoch도 학습해 시간 비교:

| 설정 | epoch당 시간(초) | 최종 mAP50-95 |
| --- | --- | --- |
| yolo11n · MPS | | |
| yolo11n · CPU | | |
| yolo11s · MPS | | |
| att-yolo-n · MPS (선택) | | |

- [ ] 학습 중 다른 탭 사용이 끊기지 않음, Activity Monitor의 GPU 기록에서 GPU 사용 확인
- [ ] 중지 → 재개가 됨, 재개 후 그래프가 이어짐
- [ ] 로그에 "MPS fallback"(CPU로 대체된 연산) 경고가 있으면 연산 이름을 기록:
- [ ] Crop 데이터셋으로도 학습 완료

## 6. 결과·비교·Export
- [ ] Result 탭: Val 그리드, 이미지 끌어다 놓기 추론, Crop run 타일링 추론 표시
- [ ] 자동 라벨 제안 생성 동작
- [ ] Compare 탭: 두 run 비교 → 지표 표의 "추론 시간 GPU" 행이 나옴 (MPS), CPU 대비 ____ 배
- [ ] HTML 리포트 저장 → Safari에서 한글·그래프 정상
- [ ] Export: ONNX 저장, 검증 PASS (검증은 일부러 CPU에서 함), metadata.json·classes.txt 생성
- [ ] FP16 옵션은 비활성(CUDA 전용)

## 7. CoreML 내보내기 (22단계)
- [ ] Export 탭 "형식"에 "CoreML (.mlpackage, macOS · Neural Engine)"이 있음 (Linux/Windows에는 없음)
- [ ] CoreML로 바꾸면 이름이 `.mlpackage`로, 버튼이 "CoreML 내보내기"로 바뀌고 opset·dynamic·simplify가 꺼짐
- [ ] 내보내기 완료, 검증 ✓ PASS (Core ML vs PyTorch, IoU ≥ 0.9), `.metadata.json`·`.classes.txt` 생성
- [ ] NMS 포함 / FP16 가중치 옵션으로도 내보내기 성공
- [ ] Xcode가 있으면 `.mlpackage`를 열어 Performance 탭에서 Neural Engine 사용 확인 (선택)
- [ ] 그레이스케일 프로젝트는 "컬러(3채널) 모델만" 안내가 나옴

## 8. Mac 설치본 (.app / .dmg, 22단계)
- [ ] `pip install -e ".[package]"` 후 `packaging/build_mac.sh` → 마지막에 self-check 9개 `[OK]`, `dist/DefectLab-macos-arm64.dmg` 생성
- [ ] 다른 Mac(또는 다른 사용자 계정)에 .dmg 복사 → 열기 → DefectLab을 Applications로 끌어다 놓기
- [ ] 첫 실행 시 Gatekeeper 경고 → README 안내대로 열림 (macOS 버전: ____)
- [ ] 설치본에서 1·3·5·6장의 핵심 동작 (프로젝트, 라벨링, MPS 학습 3 epoch, 비교 리포트, ONNX·CoreML 내보내기)
- [ ] 학습 중 Dock에 아이콘이 하나 더 생기지 않음 (학습 프로세스는 같은 실행 파일의 `--train-worker`)
- [ ] 앱 크기: ____ GB, .dmg 크기: ____ GB

## 9. 가속 측정 → MLX 진행 여부 (22단계)
- [ ] `pip install -e ".[bench]"` 후 `python tools/bench_apple.py --json bench.json` (10~20분, 다른 앱은 닫고 전원 연결)
- [ ] 출력된 표 3개와 마지막 **판단** 줄을 그대로 공유 (또는 `bench.json`)
- [ ] 판단 기준: 학습 1 step에서 MLX가 MPS보다 **1.5배 이상** 빠르면 MLX 학습 엔진을 별도 단계로 계획, 아니면 MPS 유지

## 10. 알려진 제한 (확인만)
- GitHub의 Mac 러너에는 MPS가 없어 자동 테스트는 CPU로만 돈다 — 속도·MPS 동작은 이 체크리스트로 확인한다.
- MPS가 지원하지 않는 연산은 CPU로 대체돼 느려질 수 있다 (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- MLX 마이크로벤치는 네트워크 forward+backward(같은 대체 loss)만 비교한다. Ultralytics의 실제 detection loss·학습 루프는
  MLX로 옮기지 않았다 — 그 작업을 할 가치가 있는지 판단하려는 측정이다.
- CoreML 변환·검증은 macOS에서만 된다 (Linux/Windows 테스트는 변환기를 가짜로 바꿔 주변 흐름만 확인).
