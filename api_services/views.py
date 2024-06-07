from django.views.generic.base import View
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from .services import woa_selection_service

# декоратор для скасування csrf захисту
@method_decorator(csrf_exempt, name='dispatch')
# класове представлення для обробки запитів від /api/woa-selection
class WoaSelectionView(View):
    def post(self, request):
        # Перевірка на наявність файлу у запиті
        print(f"FILES={request.FILES}")

        if "csv_file" not in request.FILES:
            return JsonResponse({"status": "failure", "error": "Missed file."}, status=400)

        # Отримання значень з полів введення
        research_title = request.POST.get("research_title", "")
        whales_num = request.POST.get("whales_num", "")
        iter_num = request.POST.get("iter_num", "")

        # Перевірка поля research_title
        if not research_title or len(research_title) <= 5:
            return JsonResponse({"status": "failure",
                                 "error": "No correct research_title."}, status=400)

        try:
            whales_num = int(whales_num)
            iter_num = int(iter_num)
            if whales_num < 5:
                return JsonResponse({"status": "failure",
                                     "error": "Whales number value must be equal or greater than 5."}, status=400)
            if iter_num < 10:
                return JsonResponse({"status": "failure",
                                     "error": "Iter number value must be equal or greater than 10."}, status=400)
        except ValueError:
            return JsonResponse({"status": "failure",
                                 "error": "Not correct number values."}, status=400)
        try:
            # Збереження файлу
            file = request.FILES['csv_file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            uploaded_file_url = fs.path(filename)

            response_data = woa_selection_service(uploaded_file_url, whales_num, iter_num)
        except Exception as e:
            return JsonResponse({"status": "failure",
                                 "error": str(e)}, status=400)

        return JsonResponse({"status": "success", "data": {"research_title": research_title, **response_data}}, status=201)
