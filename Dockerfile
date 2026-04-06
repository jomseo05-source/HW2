# 빌드 속도 및 용량 최적화를 위해 TensorFlow가 잘 지원되는 Python 3.10 slim 버전을 사용합니다.
FROM python:3.10-slim

# 보안 및 권한 관리를 위해 컨테이너 내부의 작업 디렉토리를 /app 으로 설정
WORKDIR /app

# 파이썬 출력을 버퍼링 없이 바로 볼 수 있도록 설정 (로그 확인에 유리)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OpenCV가 리눅스 환경에서 의존하는 시스템 패키지 설치 및 임시 파일 삭제로 이미지 크기 최소화
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 패키지 설치용 파일만 먼저 복사하여 Docker 레이어 캐싱의 이점을 극대화합니다
COPY requirements.txt .

# No-cache-dir을 사용해 pip 캐시 폴더를 남기지 않음으로써 이미지 용량을 줄임
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스코드 전체 복사
COPY . .

# FastAPI 포트 노출
EXPOSE 8000

# 서버 실행 (수정 시 재시작이 필요 없도록 --reload 제외, production 수준)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
