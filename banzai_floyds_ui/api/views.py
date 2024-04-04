from rest_framework.response import Response
from rest_framework.decorators import api_view


@api_view(['GET'])
def printHelp(request):
    return Response({'message': 'Welcome to BANZAI-FLOYDS reprocessing API.'})
