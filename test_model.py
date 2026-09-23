from PIL import Image
from app.model import predict_image

image = Image.new("RGB", (224, 224), color="white")

result = predict_image(image)

print(result)