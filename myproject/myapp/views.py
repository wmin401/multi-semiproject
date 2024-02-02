
# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_GET, require_POST
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from io import BytesIO
import base64
import matplotlib

matplotlib.use('Agg')  # Matplotlib이 메인 스레드 이외에서도 사용될 수 있도록 설정

@require_GET
def get_api_view(request):
    # 여기에서 GET 요청을 처리하고 응답을 생성합니다.!!
    result = {'message': 'GET 요청이 성공적으로 처리되었습니다.'}
    return JsonResponse(result)


@require_POST
def post_api_view(request):
    try:
        # POST 요청으로 전달된 JSON 데이터 파싱
        data = json.loads(request.body.decode('utf-8'))



        # 여기에서 POST 요청을 처리하고 응답을 생성합니다.
        result = {'message': 'POST 요청이 성공적으로 처리되었습니다.', 'data': data}
        return JsonResponse(result)
    except json.JSONDecodeError as e:
        return JsonResponse({'error': '잘못된 JSON 형식입니다.'}, status=400)


def visualize_data(request):
    # CSV 파일 읽기
    csv_file_path = '/Users/james/my_project/multi-semiproject/myproject/myapp/dummy_data.csv'
    df = pd.read_csv(csv_file_path)

    # Matplotlib 및 Seaborn을 사용하여 시각화 생성
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 예시: 데이터에 따른 Seaborn 시각화
    sns.barplot(x='category', y='value', data=df)

    # 그래프를 이미지로 변환
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # 이미지를 Base64로 인코딩
    encoded_image = base64.b64encode(image_stream.read()).decode("utf-8")

    # 시각화된 이미지를 템플릿에 전달
    return render(request, 'visualization.html', {'plot_image': encoded_image})


