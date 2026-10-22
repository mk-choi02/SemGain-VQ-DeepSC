# 새로운 독립 GitHub 저장소로 등록하기

새 저장소 추천 이름: `SemGain-VQ-DeepSC`. `EHeum/DeepSC`의 fork로 올리지 않아도 됩니다. 코드가 독립 패키지이며, 원본 DeepSC는 **로컬 sibling dependency**로 연동합니다.

## 0. 업로드 전 검증 (Linux / macOS / WSL)

1. 배포 zip을 다운로드하고 압축을 풀어 `SemGain-VQ-DeepSC` 폴더로 이름을 바꿉니다.
2. 해당 폴더에서 터미널을 실행합니다. `pwd`와 `ls -a`에서 `README.md`, `pyproject.toml`, `.gitignore`, `semgain/`, `scripts/`, `tests/`가 보이는지 확인합니다.
3. `python -m pip install -e '.[dev]'`를 실행합니다.
4. `python -m pytest -q`에서 테스트가 통과하는지 확인합니다.
5. `python scripts/run_semgain.py --mode toy --snrs 6 --trials 20 --group-size 4 --out results/smoke.json`에서 결과가 생성되는지 확인합니다.
6. `git status` 전에 `.gitignore`가 `checkpoints/`, `results/`, `.venv/`를 제외하는지 확인합니다. **데이터셋과 모델 가중치를 GitHub에 올리지 않습니다.**

Windows PowerShell에서도 `python -m pip ...`, `git ...`는 동일하게 사용합니다. `PYTHONPATH` 설정만 README의 PowerShell 명령으로 바꿉니다.

## 1. GitHub.com에서 빈 저장소 만들기

1. https://github.com 로그인 → 우측 상단 `+` → `New repository`.
2. Repository name: `SemGain-VQ-DeepSC`.
3. 공개 여부는 연구 중이라면 `Private` 권장. 공개해도 데이터/체크포인트는 제외합니다.
4. **Add a README file / Add .gitignore / Choose a license를 모두 선택하지 않습니다.** 로컬 프로젝트에 README와 gitignore가 이미 있습니다. License는 권리 검토 후 별도 추가하면 됩니다.
5. `Create repository` 클릭. 생성 화면의 HTTPS URL 복사.

## 2. 로컬 프로젝트에서 첫 커밋

```bash
cd ~/workspace/SemGain-VQ-DeepSC
# 아래 두 값은 본인 GitHub 계정의 이름/이메일로 바꿉니다.
git config --global user.name "YOUR_GITHUB_NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"

git init
git branch -M main
git add .
git status --short
# checkpoint(.pt/.pth), 원본 데이터(.json 대용량), .venv 등이 있는지 검토
git commit -m "Initial SemGain-VQ-DeepSC research prototype"
```

## 3. GitHub 원격 저장소에 연결 후 업로드

```bash
git remote add origin https://github.com/YOUR_USERNAME/SemGain-VQ-DeepSC.git
git remote -v
git push -u origin main
```

`YOUR_USERNAME`을 본인 계정명으로 바꿉니다. 이때 GitHub는 인증을 요구할 수 있습니다. 일반 GitHub 계정 비밀번호를 HTTPS Git 인증용으로 쓰지 마세요. 시스템에 설치된 Git Credential Manager의 브라우저 인증, `gh auth login`, 또는 인증된 SSH 키를 사용하세요.

**대안: GitHub CLI (`gh`) 설치 및 로그인 후**

```bash
gh auth login
gh repo create SemGain-VQ-DeepSC --private --source=. --remote=origin --push
```

이 방법은 GitHub 웹사이트에서 빈 저장소를 먼저 만들 필요가 없습니다.

## 4. 업로드 확인

1. `https://github.com/YOUR_USERNAME/SemGain-VQ-DeepSC` 접속.
2. 최상단 README가 렌더링되는지 확인.
3. `semgain/`, `scripts/`, `tests/`, `.github/workflows/ci.yml`이 보이는지 확인.
4. `Actions` 탭에서 테스트 workflow가 실행되고 성공하는지 확인(저장소 정책상 허용된 경우).
5. `checkpoints/`, private corpus 및 로컬 가상환경이 커밋되지 않았는지 확인.

## 5. 이후 수정 내용을 반영할 때

```bash
git status
git add .
git commit -m "Improve channel-aware risk evaluation"
git push
```

## 6. 별도 DeepSC 연동 예시

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/EHeum/DeepSC.git
# 업로드한 본인 신규 저장소 복제
git clone https://github.com/YOUR_USERNAME/SemGain-VQ-DeepSC.git
cd SemGain-VQ-DeepSC
python -m pip install -e '.[dev]'
export PYTHONPATH="$(pwd):$(pwd)/../DeepSC"
python -m pytest -q
# 이어서 README의 verify_environment → codebook → finetune → importance → risk → evaluate 순서 실행
```

## 자주 발생하는 오류

- `fatal: remote origin already exists`: `git remote -v` 확인 후 `git remote set-url origin https://github.com/YOUR_USERNAME/SemGain-VQ-DeepSC.git`.
- `src refspec main does not match any`: `git status`에서 초기 커밋이 존재하는지 확인하고 `git branch -M main` 및 `git commit` 재실행.
- `Repository not found`: 원격 URL의 소유자/저장소명과 본인의 접근 권한 확인.
- `Permission denied (publickey)`: SSH를 사용 중이라면 키를 등록하거나 HTTPS+브라우저 인증을 사용.
- `ModuleNotFoundError: deepsc`: `PYTHONPATH`에 옆 폴더 `../DeepSC`가 포함되어 있는지 확인.
- `Vocabulary/order mismatch`: 원본 checkpoint와 정확히 동일한 tokenizer 및 vocabulary 순서를 사용해야 합니다.
- `No teacher samples`: `--start-sent`가 학습 split 범위를 넘었는지 확인; 임의로 test 데이터를 편입하지 마세요.

공식 문서: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
