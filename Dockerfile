# 예시: Python FastAPI 애플리케이션을 위한 Dockerfile
FROM python:3.10

WORKDIR /app

# 요구사항 파일 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사 (main.py 포함)
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
