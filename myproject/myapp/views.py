from django.shortcuts import render

# Create your views here.
# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_GET, require_POST
import json

@require_GET
def get_api_view(request):
    # 여기에서 GET 요청을 처리하고 응답을 생성합니다.
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


